import torch
import torch.nn as nn
import copy
from transformers import PretrainedConfig, PreTrainedModel
from mamba_ssm import Mamba

class MambaConfig(PretrainedConfig):
    model_type = "mamba"
    
    def __init__(
        self,
        d_model=4096,
        n_layer=4,
        vocab_size=32000,
        d_state=16,
        d_conv=4,
        expand=2,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        **kwargs,
    ):
        self.d_model = d_model
        self.n_layer = n_layer
        self.vocab_size = vocab_size
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

class MambaDraftModel(PreTrainedModel):
    config_class = MambaConfig
    
    def __init__(self, config, load_emb=False, path=None, bias=True, total_tokens=50, depth=4, top_k=8, threshold=1.0):
        super().__init__(config)
        self.config = config
        self.total_tokens = total_tokens
        self.depth = depth
        self.top_k = top_k
        
        # Embedding layer (frozen as per MMModel design usually, but here we might need to project)
        # In MMModel, it uses the base model's embeddings. 
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, config.pad_token_id)
        if load_emb:
            import os
            import json
            from safetensors import safe_open
            try:
                with open(os.path.join(path, "model.safetensors.index.json"), "r") as f:
                    index_json = json.loads(f.read())
                    emb_path = index_json["weight_map"]["language_model.model.embed_tokens.weight"]
                with safe_open(os.path.join(path, emb_path),
                               framework="pt",
                               device="cpu") as f:
                    tensor_slice = f.get_slice("language_model.model.embed_tokens.weight")
                    vocab_size, hidden_dim = tensor_slice.get_shape()
                    tensor = tensor_slice[:, :hidden_dim].float()
            except:
                try:
                    with open(os.path.join(path, "pytorch_model.bin.index.json"), "r") as f:
                        index_json = json.loads(f.read())
                        emb_path = index_json["weight_map"]["model.embed_tokens.weight"]
                    weights = torch.load(os.path.join(path, emb_path))
                    tensor = weights["model.embed_tokens.weight"].float()
                except:
                     # Fallback or error
                     print("Warning: Could not load embeddings from path:", path)
                     tensor = torch.randn(config.vocab_size, config.d_model)

            self.embed_tokens.weight.data = tensor
            
        for param in self.embed_tokens.parameters():
            param.requires_grad = False 
        
        self.layers = nn.ModuleList()
        for i in range(config.n_layer):
            self.layers.append(
                Mamba(
                    d_model=config.d_model, # Model dimension d_model
                    d_state=config.d_state,  # SSM state expansion factor
                    d_conv=config.d_conv,    # Local convolution width
                    expand=config.expand,    # Block expansion factor
                    layer_idx=i,
                )
            )
        
        self.norm_f = nn.LayerNorm(config.d_model)
        # MMModel has a projection layer to fuse hidden_states and input_embeddings
        self.fc = nn.Linear(2 * config.d_model, config.d_model, bias=bias)
        # Head is usually passed in or we can have one. MMModel has no head in __init__ but uses one in forward?
        # MMModel has self.fc. 
        # MMModel uses the base model's head for token prediction in topK_genrate.
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Mamba state cache
        self.inference_params = None

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        hidden_states,
        input_ids=None,
        input_embeddings=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None, # Alias for input_embeddings
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        std=None,
        inference_params=None,
    ):
        if inputs_embeds is not None:
            input_embeddings = inputs_embeds
            
        # MMModel fusion logic:
        # hidden_states = self.fc(torch.cat((inputs_embeds, hidden_states), dim=-1))
        
        if input_embeddings is not None and hidden_states is not None:
            # Check shapes
            if input_embeddings.shape[1] != hidden_states.shape[1]:
                # If shapes mismatch (e.g. during tree drafting where hidden_states is [top_k, 1, d] and input_embeddings is [top_k, 1, d])
                # They should match.
                pass
                
            x = self.fc(torch.cat((input_embeddings, hidden_states), dim=-1))
        elif input_embeddings is not None:
             x = input_embeddings
        elif hidden_states is not None:
             x = hidden_states
        else:
            raise ValueError("Either hidden_states or input_embeddings must be provided")

        # Mamba forward pass
        # Mamba implementation usually takes (x, inference_params)
        
        all_hidden_states = () if output_hidden_states else None
        
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (x,)
            
            x = layer(x, inference_params=inference_params)
            
        x = self.norm_f(x)
        
        if output_hidden_states:
            all_hidden_states += (x,)
            
        return x, None # past_key_values dummy

    def reset_kv(self):
        self.inference_params = None
        
    def init_tree(self):
        self.tree_mask_init = torch.eye(self.top_k, self.top_k)
        self.tree_mask_init[:, 0] = 1
        self.tree_mask_init = self.tree_mask_init[None, None]
        self.position_ids = torch.arange(self.top_k)
        self.position_ids = self.position_ids[None]

    @torch.no_grad()
    def topK_genrate(self, hidden_states, input_ids, input_embeds, head, logits_processor):
        # Mamba Tree Drafting Implementation
        
        # 1. Initialization
        self.training = False
        
        # Robust slicing to align input_embeds with hidden_states
        if input_embeds.shape[1] > hidden_states.shape[1]:
            input_embeds = input_embeds[:, -hidden_states.shape[1]:, :]
            
        batch_size = input_ids.shape[0]
        # input_embeds: [batch, seq_len, d_model]
        # We assume input_embeds contains the context.
        
        # Mamba Inference Params
        if self.inference_params is None:
            try:
                from mamba_ssm.utils.generation import InferenceParams
                self.inference_params = InferenceParams(max_seqlen=2048, max_batch_size=2048)
            except ImportError:
                # Fallback if InferenceParams is not available (older versions or different structure)
                class InferenceParams:
                    def __init__(self, max_seqlen, max_batch_size):
                        self.max_seqlen = max_seqlen
                        self.max_batch_size = max_batch_size
                        self.seqlen_offset = 0
                        self.key_value_memory_dict = {}
                        self.lengths_per_sample = None
                self.inference_params = InferenceParams(max_seqlen=2048, max_batch_size=2048)
        
        inference_params = self.inference_params
        
        sample_token = input_ids[:, -1]
        # For Mamba, we pass the whole sequence to initialize state.
        # We need to ensure we don't double-process if state is already cached?
        
        out_hidden, _ = self(hidden_states, input_embeddings=input_embeds, inference_params=inference_params)
        
        # Prepare for tree search - use a copy of inference_params to avoid corrupting the trunk state
        tree_params = copy.deepcopy(inference_params)
        
        # Last hidden state
        last_hidden = out_hidden[:, -1]
        last_headout = head(last_hidden)
        
        # Top-k selection for first step
        top_k = self.top_k
        last_p = nn.LogSoftmax(dim=-1)(last_headout)
        top = torch.topk(last_p, top_k, dim=-1)
        topk_index, topk_p = top.indices, top.values
        
        scores = topk_p[0]
        
        # Lists to store tree structure
        scores_list = []
        parents_list = []
        ss_token = []
        
        scores_list.append(scores[None])
        parents_list.append(torch.zeros(1, dtype=torch.long, device=scores.device))
        ss_token.append(topk_index)
        
        # Prepare for loop
        input_ids = topk_index # [1, top_k]
        input_embeds = self.embed_tokens(input_ids) # [1, top_k, d_model]
        
        # Expand hidden state for the next step
        # MMModel: input_hidden = last_hidden[None].repeat(1, top_k, 1)
        input_hidden = last_hidden[None].repeat(1, top_k, 1) # [1, top_k, d_model]
        
        # Expand Mamba state
        # inference_params.key_value_memory_dict maps layer -> state
        # state shape: [batch, d_state, d_conv] or [batch, d_model, d_state] depending on implementation
        # We need to repeat the state `top_k` times.
        # Current batch size is 1. We want `top_k`.
        
        for layer_idx in tree_params.key_value_memory_dict:
            state = tree_params.key_value_memory_dict[layer_idx]
            # state: [1, ...]
            
            
            
            # Expand state
            # state: [1, d_inner, d_state] (example)
            if isinstance(state, tuple):
                tree_params.key_value_memory_dict[layer_idx] = tuple(s.repeat(top_k, 1, 1) for s in state)
            else:
                tree_params.key_value_memory_dict[layer_idx] = state.repeat(top_k, 1, 1)
            
        # Update sequence length offset in tree_params
        tree_params.seqlen_offset += input_embeds.shape[1] # +1?
        # Actually Mamba tracks seqlen_offset.
        
        # Reshape inputs for batch processing
        input_hidden = input_hidden.view(top_k, 1, -1)
        input_embeds = input_embeds.view(top_k, 1, -1)
        
        topk_cs_index = torch.arange(top_k, device=self.embed_tokens.weight.device)
        
        depth = self.depth
        total_tokens = self.total_tokens
        
        for i in range(depth):
            # Mamba Forward on batch of size top_k
            # We pass input_embeds.
            # In MMModel, `input_hidden` is passed as `hidden_states`.
            
            out_hidden, _ = self(input_hidden, input_embeddings=input_embeds, inference_params=tree_params)
            # out_hidden: [top_k, 1, d_model]
            
            # Calculate logits
            last_headout = head(out_hidden[:, 0]) # [top_k, vocab_size]
            last_p = nn.LogSoftmax(dim=-1)(last_headout)
            
            top = torch.topk(last_p, top_k, dim=-1)
            topk_index, topk_p = top.indices, top.values # [top_k, top_k]
            
            # Cumulative scores
            # scores: [top_k] (from previous step)
            # topk_p: [top_k, top_k]
            cu_scores = topk_p + scores[:, None] # [top_k, top_k]
            
            # Select top-k best paths globally from the top_k*top_k options
            topk_cs = torch.topk(cu_scores.view(-1), top_k, dim=-1)
            topk_cs_index, topk_cs_p = topk_cs.indices, topk_cs.values
            scores = topk_cs_p # [top_k]
            
            # Determine which parent branch each selection comes from
            out_ids = topk_cs_index // top_k # [top_k] indices into the batch
            
            # Update Mamba state to keep only the selected branches
            # We duplicate/select states based on out_ids
            for layer_idx in tree_params.key_value_memory_dict:
                state = tree_params.key_value_memory_dict[layer_idx]
                # state: [top_k, ...]
                if isinstance(state, tuple):
                    tree_params.key_value_memory_dict[layer_idx] = tuple(s[out_ids] for s in state)
                else:
                    tree_params.key_value_memory_dict[layer_idx] = state[out_ids]
                
            # Prepare inputs for next step
            # New input_ids
            # topk_index is [top_k, top_k] (parent, child)
            # We need the token IDs corresponding to topk_cs_index
            input_ids = topk_index.view(-1)[topk_cs_index][None] # [1, top_k]
            
            # Record tree structure
            bias1 = top_k if i > 0 else 0
            bias2 = max(0, i - 1)
            bias = 1 + top_k ** 2 * bias2 + bias1
            parents = (topk_cs_index + bias) # This might need adjustment if logic differs
            # MMModel logic: parents = (topk_cs_index + bias)
            # Let's stick to MMModel's logic for parents calculation as it relates to the flattened tree.
            
            # In first iter, `topk_cs_index` is `arange(top_k)`.
            
            parents = (topk_cs_index + bias) # topk_cs_index here is from prev step (or arange)
            parents_list.append(parents)
            
            ss_token.append(topk_index) # This stores ALL top-k candidates?
            # MMModel: ss_token.append(topk_index)
            # topk_index is [top_k, top_k]
            
            scores_list.append(cu_scores)
            
            # Update inputs for next iter
            input_embeds = self.embed_tokens(input_ids) # [1, top_k, d_model]
            input_embeds = input_embeds.view(top_k, 1, -1) # [top_k, 1, d_model]
            
            # Update input_hidden
            # out_hidden was [top_k, 1, d_model]
            # We select based on out_ids
            input_hidden = out_hidden[out_ids] # [top_k, 1, d_model]
            
            # Update tree_params seqlen
            tree_params.seqlen_offset += 1

        # Reconstruct Tree (Copied from MMModel)
        scores_list = torch.cat(scores_list, dim=0).view(-1)
        ss_token_list = torch.cat(ss_token, dim=0).view(-1)
        top_scores = torch.topk(scores_list, total_tokens, dim=-1)
        top_scores_index = top_scores.indices
        top_scores_index = torch.sort(top_scores_index).values

        draft_tokens = ss_token_list[top_scores_index]
        draft_tokens = torch.cat((sample_token, draft_tokens), dim=0)

        draft_parents = torch.cat(parents_list, dim=0)[top_scores_index // top_k].long()
        mask_index = torch.searchsorted(top_scores_index, draft_parents - 1, right=False)
        mask_index[draft_parents == 0] = -1
        mask_index = mask_index + 1
        mask_index_list = mask_index.tolist()

        tree_mask = torch.eye(total_tokens + 1).bool()
        tree_mask[:, 0] = True
        for i in range(total_tokens):
            tree_mask[i + 1].add_(tree_mask[mask_index_list[i]])

        tree_position_ids = torch.sum(tree_mask, dim=1) - 1
        tree_mask = tree_mask.float()[None, None]
        draft_tokens = draft_tokens[None]

        # Retrieve Indices
        max_depth = torch.max(tree_position_ids) + 1
        noleaf_index = torch.unique(mask_index).tolist()
        noleaf_num = len(noleaf_index) - 1
        leaf_num = total_tokens - noleaf_num

        retrieve_indices = torch.zeros(leaf_num, max_depth.item(), dtype=torch.long) - 1
        retrieve_indices = retrieve_indices.tolist()

        rid = 0
        position_ids_list = tree_position_ids.tolist()

        for i in range(total_tokens + 1):
            if i not in noleaf_index:
                cid = i
                depth = position_ids_list[i]
                for j in reversed(range(depth + 1)):
                    retrieve_indices[rid][j] = cid
                    cid = mask_index_list[cid - 1]
                rid += 1

        if logits_processor is not None:
            maxitem = total_tokens + 5
            def custom_sort(lst):
                sort_keys = []
                for i in range(len(lst)):
                    sort_keys.append(lst[i] if lst[i] >= 0 else maxitem)
                return sort_keys
            retrieve_indices = sorted(retrieve_indices, key=custom_sort)

        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
        tree_position_ids = tree_position_ids.to(hidden_states.device).unsqueeze(0)
        
        return draft_tokens, retrieve_indices, tree_mask, tree_position_ids
 

