import deepspeed
import os
import argparse
import wandb
import sys

# 1. Add the project root (`SpecVLA`) to the path.
# This allows imports like `from openvla.prismatic...` to work.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 2. Add the `openvla` subfolder to the path.
# This is necessary because the library itself uses absolute imports
# like `from prismatic...` assuming it's a top-level module.
openvla_path = os.path.join(project_root, 'openvla')
if openvla_path not in sys.path:
    # We insert at index 1 to keep the project root as the first priority
    sys.path.insert(1, openvla_path)
print(sys.path)
from torch import nn
from torch.utils.data import Dataset, DataLoader
from typing import Any, Dict, List
from accelerate import Accelerator
from accelerate.utils import set_seed
import torch
from openvla.specdecoding.model.mamba import MambaDraftModel, MambaConfig

# Argument parsing
parser = argparse.ArgumentParser(description='Mamba training script.')
parser.add_argument('--local_rank', type=int, default=1, help='local rank passed from distributed launcher')
parser = deepspeed.add_config_arguments(parser)
cmd_args = parser.parse_args()

# Configuration
basepath="/scratch/users/jjosh/spec/SpecVLA/backbone_models/openvla-7b-finetuned-libero-goal"
cpdir="ckpt_libero_goal_mamba_bf_h200_ckpt"
tmpdir="/scratch/users/jjosh/spec/SpecVLA/dataset/libero_goal_dataset"

train_config = {
    "lr": 5e-5,
    "bs": 4,
    "gradient_accumulation_steps": 1,
    "datapath": f"{tmpdir}",
    "is_warmup": True,
    "num_epochs": 200,
    "num_warmup_steps": 2000,
    "total_steps": 800000,
    "p_w": 0.1,
    "v_w": 1.0,
    "head_w": 0.1,
    "num_workers": 1,
    "embeding": True,
    "act": "No",
    "data_noise": True,
    "noise": "uniform",
    "mean": 0.0,
    "std": 0.2,
    "residual": "true,norm",
    "max_len": 2048,
    "b1": 0.9,
    "b2": 0.95,
    "grad_clip": 0.5,
    # Mamba specific
    "d_model": 4096,
    "n_layer": 2,
    "d_state": 16,
    "expand": 2
}

# Setup
torch.backends.cuda.matmul.allow_tf32 = True
set_seed(0)
accelerator = Accelerator(mixed_precision="bf16")

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '14756'
os.environ['WANDB_MODE'] = 'offline'
deepspeed.init_distributed()
rank = torch.distributed.get_rank()

if rank == 0:
    wandb.init(project="OpenVLA-Mamba", entity="ss_wang", config=train_config)

# Model Loading
# We need to load the VLA model to get the head weights, similar to original script
from transformers import AutoModelForVision2Seq
from openvla.prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from openvla.prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction

class FinetuneConfig:
    vla_path: str = "openvla/openvla-7b-finetuned-libero-goal"

cfg = FinetuneConfig()
AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
vla = AutoModelForVision2Seq.from_pretrained(
    cfg.vla_path,
    torch_dtype=torch.bfloat16,
    quantization_config=None,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)

lm = vla.language_model
vla_lm_head = lm.lm_head
vocab_size, hidden_dim = vla_lm_head.out_features, vla_lm_head.in_features
tensor = vla_lm_head.weight.data

# Head for training (frozen)
head = torch.nn.Linear(tensor.shape[1], tensor.shape[0], bias=False)
head.weight.data = tensor

#second iter
#for param in head.parameters():
 #   param.requires_grad = False

# Dataset Definitions (Copied from original script)
def list_files(path):
    datapath = []
    for root, directories, files in os.walk(path, followlinks=True):
        for file in files:
            file_path = os.path.join(root, file)
            datapath.append(file_path)
    return datapath

class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.0):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        tensor = data["hidden_state_big"]
        noise = torch.randn(tensor.size()) * self.std + self.mean
        noisy_tensor = tensor + noise
        data["hidden_state_big"] = noisy_tensor
        return data

class AddUniformNoise:
    def __init__(self, std=0.0):
        self.std = std

    def __call__(self, data):
        tensor = data["hidden_state_big"]
        noise = (torch.rand_like(tensor) - 0.5) * self.std * 512 / tensor.shape[1]
        noisy_tensor = tensor + noise
        data["hidden_state_big"] = noisy_tensor
        return data

class CustomDataset(Dataset):
    def __init__(self, datapath, transform=None):
        self.data = datapath
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = torch.load(self.data[index])
        new_data = {}
        hidden_state = torch.cat([item for item in data['hidden_state'][1]],dim=0)
        embedding_state = torch.cat([item for item in data['hidden_state'][0]],dim=0)
        target_tokens = torch.tensor(data['predicted_tokens'])
        input_ids = data['input_ids']
        loss_mask = data["loss_mask"]
        pixel_values = data["pixel_values"]
        length = data['hidden_state'][0][0].shape[0]-1
        attention_mask = [1] * (length)
        loss_mask = [0]*(length) + [1]*7
        loss_mask[-1] = 0
        input_ids_target = torch.cat([torch.tensor([0]*(length-1)),target_tokens,torch.tensor([0])])
        target = hidden_state[1:, :]
        embedding_state = embedding_state[1:, :]
        zeropadding = torch.zeros(1, target.shape[1])
        target = torch.cat((target, zeropadding), dim=0)
        embedding_state = torch.cat((embedding_state, zeropadding), dim=0)
        loss_mask[-1] = 0
        new_data["attention_mask"] = attention_mask
        new_data["loss_mask"] = loss_mask
        new_data["target"] = target
        new_data["hidden_state_big"] = hidden_state
        new_data["embedding_state"] = embedding_state
        new_data["input_ids"] = input_ids_target
        new_data['pixel_values'] = pixel_values

        if self.transform:
            new_data = self.transform(new_data)

        return new_data

class DataCollatorWithPadding:
    def paddingtensor1D(self, intensors, N):
        n = intensors.shape[0]
        if N>n:
            padding_tensor = torch.zeros(N - n, dtype=intensors.dtype)
            outtensors = torch.cat((intensors, padding_tensor), dim=0)
            return outtensors
        return intensors

    def paddingtensor2D(self, intensors, N):
        n, d = intensors.shape
        padding_tensor = torch.zeros(N - n, d, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=0)
        return outtensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max([item['hidden_state_big'].shape[0] for item in features])
        batch_input_ids = torch.cat([self.paddingtensor1D(item['input_ids'],max_length).unsqueeze(0) for item in features],dim=0)
        batch_hidden_states = torch.cat([self.paddingtensor2D(item['hidden_state_big'], max_length).unsqueeze(0) for item in features],dim=0)
        batch_embedding_states = torch.cat([self.paddingtensor2D(item['embedding_state'], max_length).unsqueeze(0) for item in features],dim=0)
        batch_target = torch.cat([self.paddingtensor2D(item['target'], max_length).unsqueeze(0) for item in features],dim=0)
        batch_loss_mask = torch.tensor([item['loss_mask'] + [0] * (max_length - len(item['loss_mask'])) for item in features])
        batch_attention_mask = torch.tensor(
            [item['attention_mask'] + [0] * (max_length - len(item['attention_mask'])) for item in features])
        batch_pixel_values = torch.cat([item['pixel_values'] for item in features])
        batch = {
            "input_ids": batch_input_ids,
            "hidden_states": batch_hidden_states,
            "embedding_states":batch_embedding_states,
            "target": batch_target,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
            'pixel_values':batch_pixel_values.to(torch.bfloat16)
        }
        return batch

def top_accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k)
        return res

#def compute_loss(target, target_p, predict, loss_mask):
#    out_head = head_engine(predict)
#    out_logp = nn.LogSoftmax(dim=2)(out_head)
#    plogp = target_p * out_logp
#    ploss = -torch.sum(torch.sum(loss_mask * plogp, 2)) / (loss_mask.shape[0] * loss_mask.shape[1])
#    vloss = criterion(predict, target.to(rank))
#    vloss = torch.sum(torch.mean(loss_mask * vloss, 2)) / (loss_mask.shape[0] * loss_mask.shape[1])
#    return vloss, ploss, out_head

def compute_loss(target, target_p, predict, loss_mask):
   
    # cast inputs to Float32. 
    predict_f32 = predict.float()
    target_f32 = target.to(rank).float()
    
    # calculate SmoothL1 in Float32
    vloss = criterion(predict_f32, target_f32)
    vloss = torch.sum(torch.mean(loss_mask * vloss, 2)) / (loss_mask.shape[0] * loss_mask.shape[1] + 1e-6)

    # head/classification Loss in Float32 as well
    out_head = head_engine(predict) # leave for now
    out_head_f32 = out_head.float() # Cast output to float for stable logsoftmax
    
    # Cast target probability to float
    target_p_f32 = target_p.float()

    out_logp = nn.LogSoftmax(dim=2)(out_head_f32)
    plogp = target_p_f32 * out_logp
    
    # Safety clamp to prevent NaN propagation if logp explodes
    if torch.isnan(plogp).any():
        plogp = torch.nan_to_num(plogp, nan=0.0)

    ploss = -torch.sum(torch.sum(loss_mask * plogp, 2)) / (loss_mask.shape[0] * loss_mask.shape[1] + 1e-6)
    
    # Return original out_head so accuracy calculation remains valid
    return vloss, ploss, out_head

# Data Loading
if train_config["data_noise"]:
    if train_config["noise"] == "uniform":
        aug = AddUniformNoise(std=train_config["std"])
    else:
        aug = AddGaussianNoise(mean=train_config["mean"], std=train_config["std"])
else:
    aug = None

datapath = list_files(train_config["datapath"])
print(f"train config index is {train_config['datapath']}")
traindatapath = datapath[:int(len(datapath) * 0.95)]
testdatapath = datapath[int(len(datapath) * 0.95):]
traindataset = CustomDataset(traindatapath, transform=aug)
testdataset = CustomDataset(testdatapath)

# Mamba Initialization
if rank == 0:
    if not os.path.exists(cpdir):
        os.makedirs(cpdir)

mamba_config = MambaConfig(
    d_model=train_config["d_model"],
    n_layer=train_config["n_layer"],
    vocab_size=vocab_size,
    d_state=train_config["d_state"],
    expand=train_config["expand"]
)

model = MambaDraftModel(mamba_config, path=basepath, load_emb=True)

criterion = nn.SmoothL1Loss(reduction="none")

num_epochs = train_config["num_epochs"]
model_engine, optimizer, train_loader, _ = deepspeed.initialize(args=cmd_args,
                                                                model=model,
                                                                model_parameters=model.parameters(),
                                                                training_data=traindataset,
                                                                collate_fn=DataCollatorWithPadding()
                                                                )

head_engine, _, test_loader, _ = deepspeed.initialize(args=cmd_args,
                                                      model=head,
                                                      model_parameters=head.parameters(),
                                                      training_data=testdataset,
                                                      collate_fn=DataCollatorWithPadding()
                                                      )
# second ag
for param in head.parameters():
    param.requires_grad = False

print('start training')
for epoch in range(num_epochs):
    top_3acc = [0 for _ in range(3)]
    correct = 0
    total = 0
    epoch_loss = 0
    num_batches = 0
    model.train()
    for batch_idx, data in enumerate(train_loader):
        model.zero_grad()
        # Mamba forward
        predict, _ = model_engine(
            hidden_states=data["hidden_states"].to(device=rank, dtype=torch.bfloat16), 
            input_ids=data["input_ids"].to(rank),
            input_embeddings=data['embedding_states'].to(device=rank, dtype=torch.bfloat16),
            attention_mask=data["attention_mask"].to(rank)
        )
        
        with torch.no_grad():
            target_head = head_engine(data["target"].to(device=rank, dtype=torch.bfloat16))
            target_p = nn.Softmax(dim=2)(target_head)
            target_p = target_p.detach()
        
        loss_mask = data["loss_mask"][:, :, None].to(rank)
        vloss, ploss, out_head = compute_loss(data["target"], target_p, predict, loss_mask)
        loss = train_config["v_w"] * vloss + train_config["p_w"] * ploss
        model_engine.backward(loss)
        model_engine.step()

        with torch.no_grad():
            _, predicted = torch.max(out_head, 2)
            _, target = torch.max(target_head, 2)
            ct = loss_mask.sum().item()
            cc = ((predicted == target) * loss_mask.squeeze()).sum().item()
            out_head = out_head.view(-1, target_head.shape[-1])[loss_mask.view(-1) == 1]
            target = target.view(-1)[loss_mask.view(-1) == 1]
            topkacc = top_accuracy(out_head, target, (1, 2, 3))
            for top_i in range(len(topkacc)):
                top_3acc[top_i] += topkacc[top_i]
            total += ct
            correct += cc
            
        if rank == 0 and ct != 0:
            logdict = {"train/lr": optimizer.optimizer.param_groups[0]["lr"], "train/vloss": vloss.item(),
                       "train/ploss": ploss.item(), "train/loss": loss.item(), "train/acc": cc / ct}
            for id, i in enumerate(top_3acc):
                logdict[f'train/top_{id + 1}_acc'] = topkacc[id].item() / ct
            wandb.log(logdict)

        del ploss, vloss
        epoch_loss += loss.item()
        num_batches += 1

    correct, total = torch.tensor(correct).cuda(), torch.tensor(total).cuda()
    correct, total = accelerator.gather_for_metrics((correct, total))
    correct, total = correct.sum().item(), total.sum().item()
    epoch_loss /= num_batches
    top_3acc = accelerator.gather_for_metrics(top_3acc)
    
    if accelerator.is_local_main_process:
        for id, i in enumerate(top_3acc):
            wandb.log({f'train/epochtop_{id + 1}_acc': i.sum().item() / total})
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss))
        print('Train Accuracy: {:.2f}%'.format(100 * correct / (total + 1e-5)))
        wandb.log({"train/epochacc": correct / (total + 1e-5), "train/epochloss": epoch_loss})

    model_engine.save_16bit_model(f"{cpdir}/state_{epoch}")
    if epoch % 10 == 0:
        deepspeed.DeepSpeedEngine.save_checkpoint(model_engine, save_dir=f"{cpdir}/state_{epoch}")
        print(f"checkpoint saved to {cpdir}/state_{epoch}")

