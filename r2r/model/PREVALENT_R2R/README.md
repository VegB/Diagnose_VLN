# Environment Setup

```bash
conda create -n prevalent python=3.6
conda activate prevalent
conda install torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
conda install caffe-gpu
pip install torch==1.10.0
pip install -r python_requirements.txt
pip install pytorch-transformers==1.2.0 
```

Follow [link](./original_README.md#train-agent-for-r2r) to download checkpoints and put them at `./snap/cvpr_agent`, `./snap/speaker`, and `./pretrained_hug_models/dicadd/checkpoint-12864`.

## Setup Matterport
See [R2R-EnvDrop Setup Matterport](../R2R-EnvDrop/README.md#setup-matterport).

# Commands

The following commands are used in our study:


## Default Setting

```bash
bash run/test_agent_default.bash
```


## Instruction Ablations

```bash
# Command Format: 
# bash [script] [cuda_device_id (default 0)] [setting] [repeat_time]

# -----Object---------
# mask
bash run/test_agent_mask_instr.bash 0 mask_object 1

# replace
bash run/test_agent_mask_instr.bash 0 replace_object 5

# controlled trial
bash run/test_agent_mask_instr.bash 0 random_mask_for_object 5


# ------Direction--------
# mask
bash run/test_agent_mask_instr.bash 0 mask_direction 1

# replace
bash run/test_agent_mask_instr.bash 0 replace_direction 5

# controlled trial
bash run/test_agent_mask_instr.bash 0 random_mask_for_direction 5


# ------Numeric--------
# numeric default
bash run/test_agent_mask_instr.bash 0 numeric_default 1

# mask
bash run/test_agent_mask_instr.bash 0 mask_numeric 1

# replace
bash run/test_agent_mask_instr.bash 0 replace_numeric 5

# controlled trial
bash run/test_agent_mask_instr.bash 0 random_mask_for_numeric 5
```

## Environment Ablations

```bash
# mask only foreground objects
bash run/test_agent_mask_env.bash 0 foreground

# mask objects except for wall/floor/ceiling
bash run/test_agent_mask_env.bash 0 all_visible

# controlled trial
bash run/test_agent_mask_env.bash 0 foreground_controlled_trial

# flip
bash run/test_agent_mask_env.bash 0 flip

```

## Dynamic Environment Object Masking

Please switch to the `dynamic` branch for the dynamic masking experiments.

```bash
# dynamically mask environment object instances mentioned in the instructions
bash run/test_agent_dynamic_mask_env.bash 0 dynamic

# controlled trial
bash run/test_agent_dynamic_mask_env.bash 1 dynamic_controlled_trial
```