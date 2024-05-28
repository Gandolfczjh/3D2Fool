import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import math
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
import torch.optim as optim

from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    OpenGLPerspectiveCameras,
    PointLights,
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    HardPhongShader,
    TexturesUV,
    materials
)

import networks
from utils import download_model_if_doesnt_exist
from data_loader_mde import MyDataset

os.environ["CUDA_VISIBLE_DEVICES"] = '5'


class DepthModelWrapper(torch.nn.Module):
    def __init__(self, encoder, decoder) -> None:
        super(DepthModelWrapper, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, input_image):
        features = self.encoder(input_image)
        outputs = self.decoder(features)
        disp = outputs[("disp", 0)]
        return disp


def disp_to_depth(disp,min_depth,max_depth):
# """Convert network's sigmoid output into depth prediction
# The formula for this conversion is given in the 'additional considerations'
# section of the paper.
# """
    min_disp=1/max_depth
    max_disp=1/min_depth
    scaled_disp=min_disp+(max_disp-min_disp)*disp
    depth=1/scaled_disp
    return scaled_disp,depth


def get_mean_depth_diff(adv_disp1, ben_disp2, scene_car_mask):
    scaler=5.4
    # print(disp_to_depth(torch.abs(adv_disp1),0.1,100)[1])
    # print(disp_to_depth(torch.abs(adv_disp1),0.1,100)[1].shape)
    # print(torch.max(disp_to_depth(torch.abs(adv_disp1),0.1,100)[1]))
    # print(torch.min(disp_to_depth(torch.abs(adv_disp1),0.1,100)[1]))
    # print(torch.max(disp_to_depth(torch.abs(ben_disp2),0.1,100)[1]))
    # print(torch.min(disp_to_depth(torch.abs(ben_disp2),0.1,100)[1]))
    # print(torch.sum(disp_to_depth(torch.abs(adv_disp1),0.1,100)[1]*scene_car_mask.unsqueeze(0))/torch.sum(scene_car_mask))
    # print(torch.sum(disp_to_depth(torch.abs(ben_disp2),0.1,100)[1]*scene_car_mask.unsqueeze(0))/torch.sum(scene_car_mask))
    dep1_adv=torch.clamp(disp_to_depth(torch.abs(adv_disp1),0.1,100)[1]*scene_car_mask.unsqueeze(0)*scaler,max=50)
    dep2_ben=torch.clamp(disp_to_depth(torch.abs(ben_disp2),0.1,100)[1]*scene_car_mask.unsqueeze(0)*scaler,max=50)
    # mean_depth_diff = torch.sum(torch.abs(dep1_adv-dep2_ben))/torch.sum(scene_car_mask)
    mean_depth_diff = torch.sum(dep1_adv-dep2_ben)/torch.sum(scene_car_mask)
    return mean_depth_diff


def get_affected_ratio(disp1, disp2, scene_car_mask):
    scaler=5.4
    dep1=torch.clamp(disp_to_depth(torch.abs(disp1),0.1,100)[1]*scene_car_mask.unsqueeze(0)*scaler,max=50)
    dep2=torch.clamp(disp_to_depth(torch.abs(disp2),0.1,100)[1]*scene_car_mask.unsqueeze(0)*scaler,max=50)
    ones = torch.ones_like(dep1)
    zeros = torch.zeros_like(dep1)
    affected_ratio = torch.sum(scene_car_mask.unsqueeze(0)*torch.where((dep1-dep2)>1, ones, zeros))/torch.sum(scene_car_mask)
    return affected_ratio


def loss_smooth(img):
    s1 = torch.pow(img[:, :, 1:, :-1] - img[:, :, :-1, :-1], 2)
    s2 = torch.pow(img[:, :, :-1, 1:] - img[:, :, :-1, :-1], 2)
    return torch.square(torch.sum(s1 + s2))
    

def loss_nps(img, color_set):
    # img: [batch_size, h, w, 3]
    # color_set: [color_num, 3]
    _, h, w, c = img.shape
    color_num, c = color_set.shape
    img1 = img.unsqueeze(1)
    color_set1 = color_set.unsqueeze(1).unsqueeze(1).unsqueeze(0)
    gap = torch.min(torch.sum(torch.abs(img1 - color_set1), -1), 1).values
    return torch.sum(gap)/h/w


def attack(args):
    model_name = "my_mono+stereo_1024x320"  # weights fine-tuned on Carla dataset
    download_model_if_doesnt_exist(model_name)
    encoder_path = os.path.join("models", model_name, "encoder.pth")
    depth_decoder_path = os.path.join("models", model_name, "depth.pth")

    # LOADING PRETRAINED MODEL
    encoder = networks.ResnetEncoder(18, False)
    depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict_enc = torch.load(encoder_path, map_location='cpu')
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)

    loaded_dict = torch.load(depth_decoder_path, map_location='cpu')
    depth_decoder.load_state_dict(loaded_dict)

    depth_model = DepthModelWrapper(encoder, depth_decoder).to(args.device)

    depth_model.eval()
    for para in depth_model.parameters():
        para.requires_grad_(False)

    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    input_resize = transforms.Resize([feed_height, feed_width])
    # keys = [("disp", 0), ("disp", 1), ("disp", 2), ("disp", 3)]
    # disp_size = [[192, 640], [96, 320], [48, 160], [24, 80]]
    
    H, W = args.camou_shape, args.camou_shape
    resolution = 8
    h, w = int(H/resolution), int(W/resolution)

    expand_kernel = torch.nn.ConvTranspose2d(3, 3, resolution, stride=resolution, padding=0).to(args.device)
    expand_kernel.weight.data.fill_(0)
    expand_kernel.bias.data.fill_(0)
    for i in range(3):
        expand_kernel.weight[i, i, :, :].data.fill_(1)

    color_set = torch.tensor([[0,0,0],[255,255,255],[0,18,79],[5,80,214],[71,178,243],[178,159,211],[77,58,0],[211,191,167],[247,110,26],[110,76,16]]).to(args.device).float() / 255

    # continuous color
    camou_para = torch.rand([1, h, w, 3]).float().to(args.device)
    camou_para.requires_grad_(True)
    optimizer = optim.Adam([camou_para], lr=args.lr)
    camou_para1 = expand_kernel(camou_para.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

    dataset = MyDataset(args.train_dir, args.img_size, args.obj_name, args.camou_mask, args.device)
    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        # num_workers=2,
    )
    # print(textures) # wjk tested
    dataset.set_textures(camou_para1)

    for epoch in range(15):
        print('-'*30 + 'epoch begin: ' + str(epoch) + '-'*30)
        tqdm_loader = tqdm(loader)
        for i, (index, total_img, total_img0, mask, img) in enumerate(tqdm_loader):
            
            input_image = input_resize(total_img)
            input_image0 = input_resize(total_img0)
            outputs = depth_model(input_image)
            # if i%3==0:
            #     total_img_np = total_img.data.cpu().numpy()[0] * 255
            #     total_img_np = Image.fromarray(np.transpose(total_img_np, (1,2,0)).astype('uint8'))
            #     total_img_np.save(os.path.join(args.log_dir, 'test_total.jpg'))
            #     total_img_np0 = total_img0.data.cpu().numpy()[0] * 255
            #     total_img_np0 = Image.fromarray(np.transpose(total_img_np0, (1,2,0)).astype('uint8'))
            #     total_img_np0.save(os.path.join(args.log_dir, 'test_total0.jpg'))

            outputs0 = depth_model(input_image0)
            mask = input_resize(mask)[:, 0, :, :]
            adv_loss = torch.sum(torch.pow(outputs*mask,2))/torch.sum(mask)
            tv_loss = loss_smooth(camou_para) * 1e-1
            nps_loss = loss_nps(camou_para, color_set) * 5
            loss = tv_loss + adv_loss + nps_loss

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            camou_para1 = expand_kernel(camou_para.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            camou_para1 = torch.clamp(camou_para1, 0, 1)
            dataset.set_textures(camou_para1)
        camou_png = cv2.cvtColor((camou_para1[0].detach().cpu().numpy()*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(args.log_dir+str(epoch)+'camou.png', camou_png)
        np.save(args.log_dir+str(epoch)+'camou.npy', camou_para.detach().cpu().numpy())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--camou_mask", type=str, default='./car/mask.jpg', help="camouflage texture mask")
    parser.add_argument("--camou_shape", type=int, default=1024, help="shape of camouflage texture")
    parser.add_argument("--obj_name", type=str, default='./car/lexus_hs.obj')
    parser.add_argument("--device", type=torch.device, default=torch.device("cuda:0"))
    parser.add_argument("--train_dir", type=str, default='/data/zjh/mde_carla/')
    parser.add_argument("--img_size", type=tuple, default=(320, 1024))
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=int, default=0.01)
    parser.add_argument("--log_dir", type=str, default='./res/')
    args = parser.parse_args()
    attack(args)
    
