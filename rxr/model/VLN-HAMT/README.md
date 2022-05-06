# Environment Setup

```bash
conda create --name vlnhamt python=3.8.5
conda activate vlnhamt
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

Follow [link](./original_README.md#installation) step 3 to download the checkpoint and place it at `./datasets/RxR/trained_models/vitbase.clip-finetune`.


## Setup Matterport

```bash
cd ~/Diagnose_VLN/data_processing/Matterport3DSimulator/
mkdir build-vlnhamt && cd build-vlnhamt
cmake ..
make -j8
```

# Commands

The following commands are used in our study:


## Default Setting

The working directory is at `./finetune_src/`.


```bash
bash scripts/test_rxr_default.sh
```


## Instruction Ablations

```bash
# Command Format: 
# bash [script] [cuda_device_id (default 0)] [setting] [repeat_time]

# -----Object---------
# mask
bash scripts/test_rxr_mask_instr.sh 0 mask_object 1

# replace
bash scripts/test_rxr_mask_instr.sh 0 replace_object 5

# controlled trial
bash scripts/test_rxr_mask_instr.sh 0 random_mask_for_object 5


# ------Direction--------
# mask
bash scripts/test_rxr_mask_instr.sh 0 mask_direction 1

# replace
bash scripts/test_rxr_mask_instr.sh 0 replace_direction 5

# controlled trial
bash scripts/test_rxr_mask_instr.sh 0 random_mask_for_direction 5


# ------Numeric--------
# numeric default
bash scripts/test_rxr_mask_instr.sh 0 numeric_default 1

# mask
bash scripts/test_rxr_mask_instr.sh 0 mask_numeric 1

# replace
bash scripts/test_rxr_mask_instr.sh 0 replace_numeric 5

# controlled trial
bash scripts/test_rxr_mask_instr.sh 0 random_mask_for_numeric 5
```

## Environment Ablations

```bash
# mask only foreground objects
bash scripts/test_rxr_mask_env.sh 2 foreground

# mask objects except for wall/floor/ceiling
bash scripts/test_rxr_mask_env.sh 2 all_visible

# controlled trial
bash scripts/test_rxr_mask_env.sh 2 foreground_controlled_trial

# flip
bash scripts/test_rxr_mask_env.sh 2 flip
```

## Dynamic Environment Object Masking

Please switch to the `dynamic` branch for the dynamic masking experiments.

```bash
# dynamically mask environment object instances mentioned in the instructions
bash scripts/test_rxr_mask_env.sh 0 dynamic

# controlled trial
bash scripts/test_rxr_mask_env.sh 7 dynamic_controlled_trial
```