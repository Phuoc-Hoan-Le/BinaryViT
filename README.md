# BinaryViT
This repository contains the training code of our work: "[BinaryViT: Pushing Binary Vision Transformers Towards Convolutional Models](https://openaccess.thecvf.com/content/CVPR2023W/ECV/html/Le_BinaryViT_Pushing_Binary_Vision_Transformers_Towards_Convolutional_Models_CVPRW_2023_paper.html)".

Vision transformers (ViTs) suffer a larger performance drop when directly applying convolutional neural network (CNN) binarization methods or existing binarization methods to binarize ViTs compared to CNNs on datasets with a large number of classes such as ImageNet-1k. Therefore, we propose BinaryViT, in which inspired by the CNN architecture, we include operations from the CNN architecture into a pure ViT architecture to enrich the representational capability of a binary ViT without introducing convolutions. These include an average pooling layer instead of a token pooling layer, a block that contains multiple average pooling branches, an affine transformation right before the addition of each main residual connection, and a pyramid structure. Experimental results on the ImageNet-1k dataset show the effectiveness of these operations that allow a fully-binary pure ViT model to be competitive with previous state-of-the-art binary (SOTA) CNN models.

An overview of our architectural modifications is illustrated below:
<div align=center>
<img src="https://github.com/Phuoc-Hoan-Le/BinaryViT/blob/main/overview.png"/>
</div>

## Run
### 1. Requirements:
* python 3.8.10, torch>=1.10.1, torchvision>=0.11.2, timm==0.6.12, transformers>=4.20.1
    
### 2. To run:
* To get the full-precision DeiT-S, either download it from Huggingface or train it from scratch using the script, "scripts/run_deit-small-patch16-224.sh".
* To get the ReActNet-DeiT-S, run "scripts/run_reactdeit-small-patch16-224.sh".
* To get the BinaryViT model, run "scripts/run_binaryvit-small-patch4-224.sh".
* To get the BinaryViT model with all patch embedding layers in full-precision, run "scripts/run_binaryvit-small-patch4-224-some-fp.sh".
* The other sh files in "scripts/" contains the settings to get the results of the 2nd, 3rd, and 4th row of Table 3 of the [paper](https://arxiv.org/abs/2306.16678).

* Note: The argument "enable-cls-token" and "disable-layerscale" only affects the ViT models that are in binary or quantized. "enable-cls-token" is only implemented for "modeling_qvit_extra_res.py". The argument "num-workers" should be set according to system specs

## Citation
If you find our work or this code useful, please cite our paper:
```
@InProceedings{Le_2023_CVPR,
    author    = {Le, Phuoc-Hoan Charles and Li, Xinlin},
    title     = {BinaryViT: Pushing Binary Vision Transformers Towards Convolutional Models},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2023},
    pages     = {4664-4673}
}
```
