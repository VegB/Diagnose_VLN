# Instructions for Touchdown


## Directory Structure

This directory should include the following subdirectories:
- The original instructions:
  - `default/`
- For object-related tokens ablations:
  - `mask_object/`
  - `replace_object/`
  - `random_mask_for_object/`: the controlled trial
- For direction-related tokens ablations:
  - `mask_direction/`
  - `replace_direction/`
  - `random_mask_for_direction/`
- For numeric tokens ablations:
  - `numeric_default/`: subsets of instructions that contains numeric tokens
  - `mask_numeric/`
  - `replace_numeric/`
  - `random_mask_for_numeric/`


For the object/direction/numeric token ablations, there should be 5 subdirectories under each setting, and corresponding directory name would be in the format of `{label}{mask_rate:.2f}_{repeat_idx}`. By default, `mask_rate = 1.00`, and `repeat_idx` spans from 0~4.


## Instruction Preparation

We provide the processed Touchdown instructions for the ablations covered in our study, which can be downloaded by the following script:

```bash
cd Diagnose_VLN/
python data_processing/download_data.py --download_instructions --instruction_dataset touchdown
```

Alternatively, you can follow the [readme](../../data_processing/process_instructions/README.md) and run the instruction processing scripts in `~/Diagnose_VLN/data_processing/process_instructions/` to generate the ablated instructions from scratch.

