# Spec-NeRF: Multi-Spectral Neural Radiance Fields
[Jiabao Li](https://github.com/TIMESTICKING), [Yuqi Li*](https://github.com/kylin-leo), Ciliang Sun, Chong Wang, and Jinhui Xiang

[Arxiv](https://arxiv.org/abs/2310.12987).

## Intro

Spec-NeRF jointly optimizes the degradation parameters and achieves high-quality multi-spectral image reconstruction results at novel views, which only requires a low-cost camera (like a phone camera but in RAW mode) and several off-the-shelf color filters. We also provide real scenarios and synthetic datasets for related studies.

## Video Demo
With recovered spectral information of the scenario, we can achieve several art effects, like

### Change the filters


https://github.com/CPREgroup/SpecNeRF-v2/assets/56912131/eae98987-3521-4f76-a756-9f6fe0115039


### Change the camera's SSF


https://github.com/CPREgroup/SpecNeRF-v2/assets/56912131/3d270945-bbe5-4281-9a81-eccc0f09bd00



### Change the ambient light source spectrum


https://github.com/CPREgroup/SpecNeRF-v2/assets/56912131/b230b331-8f53-46ac-8d8e-73232ed6f3a1



## Preliminaries

We conduct our experiments based on [TensoRF](https://apchenstu.github.io/TensoRF/), **please use the branch named `public` in our repository** and feel free to report issues, we'd really appreciate it!



#### Tested on Ubuntu 18 / Windows 11 + Pytorch 1.11 + cuda 11.3



## Dataset
Download the two types of datasets (real senario and synthetic one) from [google drive](https://drive.google.com/file/d/1wBux0JdsimjoDJfBvOuo8lJWi9ekk3cD/view?usp=drivesdk).

For the real dataset, please first separate the images into pose-based groups since all raw images are in the folder `RAW` by running `python split_allimg2filterfixed_classify.py --scene_dir <your full path>/multi-view-MSI/filter19/xjhdesk --filter_num 20 --legacy 0 --angle_num 9 --img_ext tiff` as administrator or root. 

This will generate nine folders named *pose?img*, each contains the images filtered by all color filters, which are the symbolic links point to the ones in `RAW` folder, so remember **do not** delete the `RAW` folder.


## Quick Start

Check what's the config file in start/start.bat file and execute `./start` | `start.bat` 

or try 

`python train.py --config ./configs/<your config file>.txt`


