## overview
This is the official repository for the paper [Physical 3D Adversarial Attacks against Monocular Depth Estimation in Autonomous Driving](http://arxiv.org/abs/2403.17301) accepted by CVPR2024.

## Abstract
Deep learning-based monocular depth estimation (MDE), extensively applied in autonomous driving, is known to be vulnerable to adversarial attacks. Previous physical attacks against MDE models rely on 2D adversarial patches, so they only affect a small, localized region in the MDE map but fail under various viewpoints. To address these limitations, we propose 3D Depth Fool (3D2Fool), the first 3D texture-based adversarial attack against MDE models. 3D2Fool is specifically optimized to generate 3D adversarial textures agnostic to model types of vehicles and to have improved robustness in bad weather conditions, such as rain and fog. Experimental results validate the superior performance of our 3D2Fool across various scenarios, including vehicles, MDE models, weather conditions, and viewpoints. Real-world experiments with printed 3D textures on physical vehicle models further demonstrate that our 3D2Fool can cause an MDE error of over 10 meters.

## Framework
![image-framework](https://github.com/Gandolfczjh/3D2Fool/blob/main/framework.png)
