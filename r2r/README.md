# Experiments on Room-to-room (R2R)

This directory contains the code and data for experiments on the [Room-to-room (R2R)](https://openaccess.thecvf.com/content_cvpr_2018/papers/Anderson_Vision-and-Language_Navigation_Interpreting_CVPR_2018_paper.pdf) dataset for indoor VLN.

We cover the following four models in our study:

- R2R-EnvDrop [[paper]](https://aclanthology.org/N19-1268.pdf), [[code]](https://github.com/airsplay/R2R-EnvDrop)
- FAST [[paper]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Ke_Tactical_Rewind_Self-Correction_via_Backtracking_in_Vision-And-Language_Navigation_CVPR_2019_paper.pdf), [[code]](https://github.com/Kelym/FAST)
- Recurrent-VLN-BERT [[paper]](https://arxiv.org/pdf/2011.13922.pdf), [[code]](https://github.com/YicongHong/Recurrent-VLN-BERT)
- PREVALENT_R2R [[paper]](https://arxiv.org/pdf/2002.10638.pdf), [[code]](https://github.com/weituo12321/PREVALENT_R2R)

Please refer to [`data/`](./data/README.md) to prepare the instructions and image features for R2R abalations, and refer to the corresponding directories in `model/` for detailed instructions on environment setup and the commands to reproduce the results in our ablation studies.