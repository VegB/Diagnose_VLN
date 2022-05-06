# Environment Setup
Please refer to the [original README.md](./original_README.md) to set up the environment.

Download the foreground bbox annotations and the annotations for all the visible objects and place them at xxx


# Commands

The following commands are used in our study:


## Precomputing ResNet-152 Features for R2R

We generate ResNet-152 features with Caffe. To replicate our approach, first download and save some Caffe ResNet-152 weights into the `models` directory. We experiment with weights pretrained on [ImageNet](https://github.com/KaimingHe/deep-residual-networks) for EnvDrop, FAST and PREVALENT, and also weights finetuned on the [Places365](https://github.com/CSAILVision/places365) dataset for Recurrent-VLN-BERT.

We use the following commands to generate the precompute image features for environment ablations:

```bash
# Install caffe-gpu
conda install caffe-gpu

CUDA_VISIBLE_DEVICES=0 python scripts/generate_img_features.py --image_feature [IMAGE_FEATURE] --mode [MODE]
```
- `IMAGE_FEATURE`: 
  - `imagenet` for ImageNet ResNet-152
  - `places365` for Places365 ResNet-152
- `MODE`:
  - `all_visible`: mask all visible objects in the environment
  - `foreground`: mask only foreground objects
  - `foreground_controlled_trial`: the controlled trial for foreground masking
  - `flip`: horizontally flipping the image at each viewpoint


Alternatively, skip the generation and run the following script to download and extract our generated tsv files into `~/Diagnose_VLN/r2r/data/img_features/`. The compressed file would be ~28.3G, and contains the ImageNet and Places365 ResNet-152 features in the above listed ablation settings.

```bash
cd Diagnose_VLN/
python data_processing/download_data.py --download_image_features --image_fearture_dataset r2r
```


## Precomputing CLIP-ViT Features for RxR

The CLIP-ViT features are used for CLIP-ViL-VLN and VLN-HAMT. 

We use the following commands to generate the precompute image features for environment ablations:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/generate_clip_img_features.py --mode [MODE]
```
- `MODE`: choose from `all_visible` / `foreground` / `foreground_controlled_trial` / `flip`

Alternatively, skip the generation and run the following script to download and extract our generated tsv files into `~/Diagnose_VLN/rxr/data/img_features/`. The compressed file would be ~2.4G, and contains the CLIP ViT-B/32 features in the above listed ablation settings.

```bash
cd Diagnose_VLN/
python data_processing/download_data.py --download_image_features --image_fearture_dataset rxr
```