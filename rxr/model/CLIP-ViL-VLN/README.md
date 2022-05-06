# Environment Setup

```bash
conda create -n clipvil python=3.6
conda activate clipvil
pip install -r python_requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

Download the checkpoints ([link](https://nlp.cs.unc.edu/data/vln_clip/models/)) and place them at `./snap/rxr_imagenet/agent_rxr_en_maxinput160_ml04` and `./snap/rxr_vit/agent_rxr_en_clip_vit_maxinput160_ml04`.

## Setup Matterport

```bash
cd ~/Diagnose_VLN/data_processing/Matterport3DSimulator/
mkdir build-clipvil && cd build-clipvil
cmake ..
make -j8
```

# Commands

The following commands are used in our study:


## Default Setting

```bash
# w/ CLIP-ViT features
bash run/test_rxr_vit_agent_default.bash

# w/ ImageNet features
bash run/test_rxr_image_agent_default.bash
```


## Instruction Ablations

```bash
# Command Format: 
# bash [script] [cuda_device_id (default 0)] [setting] [repeat_time]

# -----Object---------
# mask
bash run/test_rxr_vit_agent_mask_instr.bash 0 mask_object 1

# replace
bash run/test_rxr_vit_agent_mask_instr.bash 0 replace_object 5

# controlled trial
bash run/test_rxr_vit_agent_mask_instr.bash 0 random_mask_for_object 5


# ------Direction--------
# mask
bash run/test_rxr_vit_agent_mask_instr.bash 0 mask_direction 1

# replace
bash run/test_rxr_vit_agent_mask_instr.bash 0 replace_direction 5

# controlled trial
bash run/test_rxr_vit_agent_mask_instr.bash 0 random_mask_for_direction 5


# ------Numeric--------
# numeric default
bash run/test_rxr_vit_agent_mask_instr.bash 0 numeric_default 1

# mask
bash run/test_rxr_vit_agent_mask_instr.bash 0 mask_numeric 1

# replace
bash run/test_rxr_vit_agent_mask_instr.bash 0 replace_numeric 5

# controlled trial
bash run/test_rxr_vit_agent_mask_instr.bash 0 random_mask_for_numeric 5

```

## Environment Ablations

```bash
# ------ViT Features--------
# mask only foreground objects
bash run/test_rxr_vit_agent_mask_env.bash 0 foreground

# mask objects except for wall/floor/ceiling
bash run/test_rxr_vit_agent_mask_env.bash 0 all_visible

# controlled trial
bash run/test_rxr_vit_agent_mask_env.bash 0 foreground_controlled_trial

# flip
bash run/test_rxr_vit_agent_mask_env.bash 0 flip

# ------ImageNet ResNet-152 Features--------
# mask only foreground objects
bash run/test_rxr_imagenet_agent_mask_env.bash 0 foreground

# mask objects except for wall/floor/ceiling
bash run/test_rxr_imagenet_agent_mask_env.bash 0 all_visible

# controlled trial
bash run/test_rxr_imagenet_agent_mask_env.bash 0 foreground_controlled_trial

# flip
bash run/test_rxr_imagenet_agent_mask_env.bash 0 flip
```

## Dynamic Environment Object Masking

Please switch to the `dynamic` branch for the dynamic masking experiments.

```bash
# dynamically mask environment object instances mentioned in the instructions

```