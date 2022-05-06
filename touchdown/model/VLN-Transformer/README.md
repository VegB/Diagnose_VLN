# Environment Setup

Follow [original README.md](./original_README.md) to setup environment and download checkpoints.

After downloading the image features, modify the `IMG_FEAT_DIR` in [test_agent_default.bash](./touchdown/scripts/test_agent_default.bash) and [test_agent_mask_instr.bash](./touchdown/scripts/test_agent_mask_instr.bash) accordingly.


# Commands

The following commands are used in our study:


## Default Setting

The working directory is at `./touchdown/`.


```bash
# RCONCAT
bash scripts/test_agent_default.bash 1 rconcat

# ARC
bash scripts/test_agent_default.bash 1 arc

# VLN-TRANS
bash scripts/test_agent_default.bash 1 vlntrans
```


## Instruction Ablations - RCONCAT

The working directory is at `./touchdown/`.

```bash
# ===============================RCONCAT===============================
# -----Object---------
# mask
bash scripts/test_agent_mask_instr.bash 0 rconcat mask_object 1

# replace
bash scripts/test_agent_mask_instr.bash 0 rconcat replace_object 5

# controlled trial
bash scripts/test_agent_mask_instr.bash 0 rconcat random_mask_for_object 5


# ------Direction--------
# mask
bash scripts/test_agent_mask_instr.bash 0 rconcat mask_direction 1

# replace
bash scripts/test_agent_mask_instr.bash 0 rconcat replace_direction 5

# controlled trial
bash scripts/test_agent_mask_instr.bash 0 rconcat random_mask_for_direction 5


# ------Numeric--------
# numeric default
bash scripts/test_agent_mask_instr.bash 0 rconcat numeric_default 1

# mask
bash scripts/test_agent_mask_instr.bash 0 rconcat mask_numeric 1

# replace
bash scripts/test_agent_mask_instr.bash 0 rconcat replace_numeric 5

# controlled trial
bash scripts/test_agent_mask_instr.bash 0 rconcat random_mask_for_numeric 5
```


## Instruction Ablations - ARC

The working directory is at `./touchdown/`.

```bash
# ===============================ARC===============================
# -----Object---------
# mask
bash scripts/test_agent_mask_instr.bash 0 arc mask_object 1

# replace
bash scripts/test_agent_mask_instr.bash 0 arc replace_object 5

# controlled trial
bash scripts/test_agent_mask_instr.bash 0 arc random_mask_for_object 5


# ------Direction--------
# mask
bash scripts/test_agent_mask_instr.bash 0 arc mask_direction 1

# replace
bash scripts/test_agent_mask_instr.bash 0 arc replace_direction 5

# controlled trial
bash scripts/test_agent_mask_instr.bash 0 arc random_mask_for_direction 5


# ------Numeric--------
# numeric default
bash scripts/test_agent_mask_instr.bash 0 arc numeric_default 1

# mask
bash scripts/test_agent_mask_instr.bash 0 arc mask_numeric 1

# replace
bash scripts/test_agent_mask_instr.bash 0 arc replace_numeric 5

# controlled trial
bash scripts/test_agent_mask_instr.bash 0 arc random_mask_for_numeric 5
```


## Instruction Ablations - VLN-Transformer

The working directory is at `./touchdown/`.

```bash
# ===============================VLN-TRANS===============================
# -----Object---------
# mask
bash scripts/test_agent_mask_instr.bash 0 vlntrans mask_object 1

# replace
bash scripts/test_agent_mask_instr.bash 0 vlntrans replace_object 5

# controlled trial
bash scripts/test_agent_mask_instr.bash 0 vlntrans random_mask_for_object 5


# ------Direction--------
# mask
bash scripts/test_agent_mask_instr.bash 0 vlntrans mask_direction 1

# replace
bash scripts/test_agent_mask_instr.bash 0 vlntrans replace_direction 5

# controlled trial
bash scripts/test_agent_mask_instr.bash 0 vlntrans random_mask_for_direction 5


# ------Numeric--------
# numeric default
bash scripts/test_agent_mask_instr.bash 0 vlntrans numeric_default 1

# mask
bash scripts/test_agent_mask_instr.bash 0 vlntrans mask_numeric 1

# replace
bash scripts/test_agent_mask_instr.bash 0 vlntrans replace_numeric 5

# controlled trial
bash scripts/test_agent_mask_instr.bash 0 vlntrans random_mask_for_numeric 5
```
