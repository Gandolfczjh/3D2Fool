## Overview
This is the official repository for the paper [Physical 3D Adversarial Attacks against Monocular Depth Estimation in Autonomous Driving](http://arxiv.org/abs/2403.17301) accepted by CVPR2024.

## Abstract
Deep learning-based monocular depth estimation (MDE), extensively applied in autonomous driving, is known to be vulnerable to adversarial attacks. Previous physical attacks against MDE models rely on 2D adversarial patches, so they only affect a small, localized region in the MDE map but fail under various viewpoints. To address these limitations, we propose 3D Depth Fool (3D2Fool), the first 3D texture-based adversarial attack against MDE models. 3D2Fool is specifically optimized to generate 3D adversarial textures agnostic to model types of vehicles and to have improved robustness in bad weather conditions, such as rain and fog. Experimental results validate the superior performance of our 3D2Fool across various scenarios, including vehicles, MDE models, weather conditions, and viewpoints. Real-world experiments with printed 3D textures on physical vehicle models further demonstrate that our 3D2Fool can cause an MDE error of over 10 meters.

## Framework
![image-framework](https://github.com/Gandolfczjh/3D2Fool/blob/main/framework.png)

## Code
```
pip install -r requirements.txt
python attack_base.py
```
* data_loader_mde.py
  > class MyDataset: load training set
  > + data_dir: rgb background images path
  > + obj_name: car model path
  > + camou_mask: mask path (the texture area to attack)
  > + tex_trans_flag: TC flag
  > + phy_trans_flag: PA flag
  > + self.set_textures(self, camou): camou is texture seed
  > + camera_pos: camera relative position in carla
* attack_base.py
  > + camou_mask: camouflage texture mask path
  > + camou_shape: shape of camouflage texture
  > + obj_name: car model path
  > + train_dir: rgb background images
  > + log_dir: result save path

## training dataset
* [BaiduNetdisk Link](https://pan.baidu.com/s/1IiD0HYRKjoNOx-hIsamHbg?pwd=3D2F)
  > + ./rgb/*.jpg: background images
  > + ./ann.pkl: camera position matrix

## Acknowledgements
* MDE_Attack - [**Paper**](https://arxiv.org/pdf/2207.04718)
| [Source Code](https://github.com/Bob-cheng/MDE_Attack)

## Citation
```
@InProceedings{Zheng_2024_CVPR,
    author    = {Zheng, Junhao and Lin, Chenhao and Sun, Jiahao and Zhao, Zhengyu and Li, Qian and Shen, Chao},
    title     = {Physical 3D Adversarial Attacks against Monocular Depth Estimation in Autonomous Driving},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {24452-24461}
}
```
