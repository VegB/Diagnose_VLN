# Diagnose_VLN
This repository contains the code and data for our paper: [Diagnosing Vision-and-language Navigation: What Really Matters](https://arxiv.org/abs/2103.16561).

## Directory Structure
We cover three VLN datasets and nine agents in our study.

- [data_processing/](./data_processing/README.md)
  - [process_instructions/](./data_processing/process_instructions/): scripts to prepare/download instructions
  - [Matterport3DSimulator](./data_processing/Matterport3DSimulator/): a copy of Matterport simulator, and scripts to prepare/download R2R/RxR image features
- [r2r/](./r2r): Data and code for experiments on Room-to-room (R2R) for indoor VLN
  - [data/](./r2r/data/)
  - [model/](./r2r/model/)
    - [R2R-EnvDrop](r2r/model/R2R-EnvDrop/)
    - [FAST](r2r/model/FAST/)
    - [Recurrent-VLN-BERT](r2r/model/Recurrent-VLN-BERT/)
    - [PREVALENT_R2R](r2r/model/PREVALENT_R2R/)
- [rxr/](./rxr): Data and code for experiments on Room-across-room (RxR) for indoor VLN
  - [data/](./rxr/data/)
  - [model/](./rxr/model/)
    - [CLIP-ViL-VLN](rxr/model/CLIP-ViL-VLN/)
    - [VLN-HAMT](rxr/model/VLN-HAMT/)
- [touchdown/](./touchdown): Data and code for experiments on Touchdown for outdoor VLN
  - [data/](./touchdown/data/)
  - [model/](./touchdown/model/)
    - [RCONCAT](touchdown/model/VLN-Transformer/)
    - [ARC](touchdown/model/VLN-Transformer/)
    - [VLN-Transformer](touchdown/model/VLN-Transformer/)
## Installation

```bash
git clone --recursive https://github.com/VegB/Diagnose_VLN
```

We describe the detailed environment setup for each model in the corresponding directory.
For instance, guidance to setup R2R-EnvDrop can be found [here](r2r/model/R2R-EnvDrop/README.md).


## Data Preparation
- Prepare R2R data: [link](./r2r/data)
- Prepare RxR data: [link](./rxr/data)
- Prepare Touchdown data: [link](./touchdown/data)

## Acknowledgements
We thank the authors for [Matterport3DSimulator](https://github.com/peteanderson80/Matterport3DSimulator), [R2R-EnvDrop](https://github.com/airsplay/R2R-EnvDrop), [FAST](https://github.com/Kelym/FAST), [Recurrent-VLN-BERT](https://github.com/YicongHong/Recurrent-VLN-BERT), [PREVALENT_R2R](https://github.com/weituo12321/PREVALENT_R2R), [CLIP-ViL-VLN](https://github.com/clip-vil/CLIP-ViL), [VLN-HAMT](https://github.com/cshizhe/VLN-HAMT), [RCONCAT](https://github.com/lil-lab/touchdown), [ARC](https://github.com/szxiangjn), [VLN-Transformer](https://github.com/VegB/VLN-Transformer) for sharing their code!