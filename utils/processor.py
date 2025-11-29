import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import numpy as np
import matplotlib.pyplot as plt

import os
import ipdb
import sys
sys.path.append("/home/fujita/work/eipl")
from eipl.utils import normalization    # resize_img, 

sys.path.append("../")
from layer import FoveaProcessor


class Processor:
    def __init__(
        self, 
        img_bounds,
        vec_bounds,
        minmax=[0.0,1.0],
        show_sample=True
        ):
        self.img_bounds = img_bounds
        self.vec_bounds = vec_bounds
        self.minmax = minmax
        self.show_sample = show_sample
    def process_imgs(
        self,
        imgs,
        resize
        ):
        #_imgs = normalization(imgs, self.img_bounds, self.minmax)
        # H,W次元のみ正規化（B,S,C次元は保持）
        batch, seq, height, width, channels = imgs.shape
        _imgs = imgs.copy()
        for b in range(batch):
            for s in range(seq):
                for c in range(channels):
                    # H,W次元のみ正規化
                    _imgs[b,s,:,:,c] = normalization(
                        _imgs[b,s,:,:,c],  # (H, W) の2次元配列
                        self.img_bounds, 
                        self.minmax
                    )
        
        if not (_imgs.shape[2]==resize[0] and _imgs.shape[3]==resize[1]):     
            _imgs = resize_img(_imgs, resize)
        _imgs = _imgs.transpose(0, 1, 4, 2, 3)
        
        if self.show_sample:
            pass
        
        return _imgs
    
    def process_vecs(
        self,
        vecs,
        ):
        _vecs = normalization(vecs, self.vec_bounds, self.minmax)
        if self.show_sample:
            pass
        
        return _vecs
    
    def cos_interpolation(data, step=5):
        data = data.copy()
        points = np.diff(data)
        for i, p in enumerate(points):
            if p == 1:
                t = np.linspace(0.0, 1.0, step * 2)
            elif p == -1:
                t = np.linspace(1.0, 0.0, step * 2)
            else:
                continue
            x_latent = (1 - np.cos(t * np.pi)) / 2
            data[i - step + 1 : i + step + 1] = x_latent

        return data # np.expand_dims(data, axis=-1)


class Deprocessor:
    def __init__(self, img_bounds, vec_bounds, minmax, select_idxs=[0,1,2]):
        self.img_bounds = img_bounds
        self.vec_bounds = vec_bounds
        self.minmax = minmax
        self.select_idxs = select_idxs
        
        self.fovea_processor = FoveaProcessor()
    
    def deprocess_img(self, imgs_hat):
        _imgs_hat = imgs_hat.permute(0,1,3,4,2)
        _imgs_hat = _imgs_hat.detach().clone().cpu().numpy()
        
        # test2.pyと同じクリッピング処理を追加
        vmin, vmax = self.minmax
        _imgs_hat[np.where(_imgs_hat < vmin)] = vmin
        _imgs_hat[np.where(_imgs_hat > vmax)] = vmax
        
        #_imgs_hat = normalization(_imgs_hat, self.minmax, self.img_bounds)
        # H,W次元のみ正規化（B,S,C次元は保持）
        batch, seq, height, width, channels = _imgs_hat.shape
        for b in range(batch):
            for s in range(seq):
                for c in range(channels):
                    # H,W次元のみ正規化
                    _imgs_hat[b,s,:,:,c] = normalization(
                        _imgs_hat[b,s,:,:,c],  # (H, W) の2次元配列
                        self.minmax, 
                        self.img_bounds
                    )
        
        _imgs_hat = np.uint8(np.clip(_imgs_hat, 0, 255))
        pred_imgs = _imgs_hat[self.select_idxs]
        return pred_imgs
    
    # def deprocess_fovea_img(self, imgs, fix_xys, img_fovea_size, density_ratio):
    #     batch, seq, c, h, w = imgs.shape
    #     fix_dim = fix_xys.shape[-2]
        
    #     _imgs = imgs.flatten(0,1)
    #     _fix_xys = fix_xys.flatten(0,1)
        
    #     _imgs = _imgs.unsqueeze(1).repeat([1,fix_dim,1,1,1])
    #     _imgs = _imgs.flatten(0,1)
    #     _fix_xys = _fix_xys.flatten(0,1)
        
    #     grid_forward = self.fovea_processor.make_xy2ret_grid_r(_fix_xys, h, img_fovea_size[0], density_ratio)
    #     _fovea_img = torch.nn.functional.grid_sample(_imgs, grid_forward, align_corners=True, mode='bilinear')
        
    #     _fovea_img = _fovea_img.unflatten(0, (1,-1))
    #     _fovea_img = self.deprocess_img(_fovea_img)
        
    #     batch, _, _h, _w, c = _fovea_img.shape
    #     fovea_img = _fovea_img.reshape(batch, seq, -1, _h, _w, c)
    #     return fovea_img
    
    def deprocess_vec(self, vecs_hat):
        _vecs_hat = vecs_hat.detach().clone().cpu().numpy()
        _vecs_hat = normalization(_vecs_hat, self.minmax, self.vec_bounds)
        pred_vecs = _vecs_hat[self.select_idxs]
        return pred_vecs

    def deprocess_key(self, key, img_size):   # 3,309,5,2
        batch, seq, _ = key.shape
        _key = key.reshape(batch, seq, -1, 2)
        x_key = torch.clip(_key[:,:,:,0].unsqueeze(-1)*img_size[1], 0.0, img_size[1])
        y_key = torch.clip(_key[:,:,:,1].unsqueeze(-1)*img_size[0], 0.0, img_size[0])
        _key = torch.cat([x_key, y_key], dim=-1)
        key = np.uint8(_key.detach().clone().cpu().numpy())
        
        return key
    
    def deprocess_gm(self, gm, img_size):
        batch, seq, _ = gm.shape
        _gm = gm.reshape(batch, seq, -1, 6)
        x_mu = torch.clip(_gm[:,:,:,0].unsqueeze(-1)*img_size[1], 0.0, img_size[1])
        y_mu = torch.clip(_gm[:,:,:,1].unsqueeze(-1)*img_size[0], 0.0, img_size[0])
        _mu = torch.cat([x_mu, y_mu], dim=-1)
        mu = np.uint8(_mu.detach().clone().cpu().numpy())
        
        x_sigma = _gm[:,:,:,2].unsqueeze(-1).detach().clone().cpu().numpy()
        y_sigma = _gm[:,:,:,3].unsqueeze(-1).detach().clone().cpu().numpy()
        theta = _gm[:,:,:,4].unsqueeze(-1).detach().clone().cpu().numpy()
        scale = _gm[:,:,:,5].unsqueeze(-1).detach().clone().cpu().numpy()
        
        return mu, x_sigma, y_sigma, theta, scale

def resize_img(imgs, resize):
    imgs = torch.tensor(imgs)
    imgs = imgs.permute(0, 1, 4, 2, 3)  # (batch, seq, c, height, width)
    batch, seq, c, h, w = imgs.shape
    imgs = imgs.reshape(-1, c, h, w)
    
    _imgs = torchvision.transforms.Resize(resize)(imgs)
    _imgs = _imgs.reshape(batch, seq, c, resize[0], resize[1])
    _imgs = _imgs.permute(0, 1, 3, 4, 2)
    imgs = _imgs.numpy()
    return imgs