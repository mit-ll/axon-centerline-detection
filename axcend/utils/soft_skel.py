import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def soft_erode(img):
    p1 = -F.max_pool3d(-img,(3,1,1),(1,1,1),(1,0,0))
    p2 = -F.max_pool3d(-img,(1,3,1),(1,1,1),(0,1,0))
    p3 = -F.max_pool3d(-img,(1,1,3),(1,1,1),(0,0,1))
    return torch.min(torch.min(p1, p2), p3)


def soft_dilate(img):
    return F.max_pool3d(img,(3,3,3),(1,1,1),(1,1,1))


def soft_open(img):
    return soft_dilate(soft_erode(img))


def soft_skel(img, iter_):
    img1  =  soft_open(img)
    skel  =  F.relu(img-img1)
    for j in range(iter_):
        img  =  soft_erode(img)
        img1  =  soft_open(img)
        delta  =  F.relu(img-img1)
        skel  =  skel +  F.relu(delta-skel*delta)
    return skel

def soft_skel_numpy(img, iter_):
    img = torch.Tensor(img)
    if img.ndim < 4:
        img = soft_skel(img.unsqueeze(0).unsqueeze(0), iter_)
        return img.squeeze().numpy()
    return soft_skel(img, iter_).numpy()
