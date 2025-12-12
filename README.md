# SpecVLA

## Setup Guidance
#### Requirements & Installation

* Python >= 3.10
* Pytorch == 2.2.0 (tested with cuda == 12.1)
* Libero == 0.1.0
* ``pip install -r requirements-min.txt``
* ``cd openvla``
* ``pip install -e .``


```
SpecVLA
├── openvla
│   ├── experiments                # Scripts for conducting libero simulation benchmark and speedup test
│   ├── prismatic                  # Derived from the openvla
│   ├── scripts                    # Derived from the openvla
|   └── specdecoding               # SpecVLA implementation
├── dataset                        # Finetuning dataset 
└── backbone_models                # Finetuned OpenVLA models
```

#### Experiment Pipeline

#### Training data generation
First, run backbone_models/download_model.py and dataset/download_dataset.py to download HF models and datasets. Then,
```
python SpecVLA/openvla/specdecoding/train-scripts/ge_data_all_openvla_goal.py
```
#### Training Draft models
```
# Replicate SpecVLA Llama draft
export PYTHONPATH='/SpecVLA'
WANDB_MODE='offline' deepspeed --master_port 23333 --include=localhost:4,5,6,7 "SpecVLA/openvla/specdecoding/train-scripts/train_deepspeed_libero_goal.py" --deepspeed_config "/SpecVLA/openvla/specdecoding/scripts/ds_config.json"

# Train Mamba draft
python3 SpecVLA/openvla/specdecoding/train-scripts/train_deepspeed_libero_goal_mamba.py -deepspeed_config "/scratch/users/jjosh/spec/SpecVLA/openvla/specdecoding/train-scripts/mamba_ds_config.json" --local_rank 0
```
#### Testing on LIBERO simulation benchmark
Replicate SpecVLA Autoregressive Generation
```
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 python /SpecVLA/openvla/experiments/robot/libero/run_libero_goal_AR.py\
  --model_family openvla \
  --pretrained_checkpoint /openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --center_crop True
```
Replicate SpecVLA Speculative Decoding
```
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 python /SpecVLA/openvla/experiments/robot/libero/run_libero_goal_Spec.py \
    --model_family openvla \
    --pretrained_checkpoint /openvla-7b-finetuned-libero-goal \
    --task_suite_name libero_goal \
    --center_crop True
```

Mamba Speculative Decoding
```
# adjust draft model path in this file to Mamba model directory; add dataset_statistics.json and config.json to the model state folder
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 python3 SpecVLA/openvla/experiments/robot/libero/run_libero_goal_Spec.py  --model_family openvla   --task_suite_name libero_goal
```

Speculative Decoding with Relaxed Acceptance
```
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 python /SpecVLA/openvla/experiments/robot/libero/run_libero_goal_Spec_Relaxed.py \
    --model_family openvla \
    --pretrained_checkpoint /openvla-7b-finetuned-libero-goal \
    --task_suite_name libero_goal \
    --center_crop True
```

## Citing

Please kindly cite us if you find our papers or codes useful.
```
@article{wang2025spec,
  title={Spec-VLA: Speculative Decoding for Vision-Language-Action Models with Relaxed Acceptance},
  author={Wang, Songsheng and Yu, Rucheng and Yuan, Zhihang and Yu, Chao and Gao, Feng and Wang, Yu and Wong, Derek F},
  journal={arXiv preprint arXiv:2507.22424},
  year={2025}
}
```
