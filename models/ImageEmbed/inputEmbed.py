 # Copyright (c) Facebook, Inc. and its affiliates.
# Modified by HanLin from  https://github.com/VITA-Group/CADTransformer.
import os
import sys
import torch
import torch.nn as nn

from ..ImageEmbed.HRNet48 import get_seg_model
from ..ImageEmbed.mul_points_sampling import vert_align_optimized

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "../", "../"))
sys.path.insert(0, os.path.join(BASE_DIR, "../"))
# from transformer import TransformerBlock

from torch.nn import functional as F

_FEAT_DIMS = {
    "resnet18": (64, 128, 256, 512),
    "resnet34": (64, 128, 256, 512),
    "resnet50": (256, 512, 1024, 2048),
    "resnet101": (256, 512, 1024, 2048),
    "resnet152": (256, 512, 1024, 2048),
    "hrnet18": (18, 36, 72, 144),
    "hrnet48": (48, 96, 192, 384),
    "resnet101_fpn": (256, 256, 256, 256),
    "unet": (512, 256, 128, 64),
    "unet_down": (512, 256, 128, 64),
    "vgg16": (64, 128, 256, 512)
}


def vert_align_custom(feats, verts, interp_mode='bilinear', padding_mode='zeros', align_corners=True):

    if torch.is_tensor(verts):
        if verts.dim() != 3:
            raise ValueError("verts tensor should be 3 dimensional")
        grid = verts
    else:
        raise ValueError(
            "verts must be a tensor or have a "
            + "`points_padded' or`verts_padded` attribute."
        )
    grid = grid[:, None, :, :2]  
    if torch.is_tensor(feats):
        feats = [feats]
    for feat in feats:
        if feat.dim() != 4:
            raise ValueError("feats must have shape (N, C, H, W)")
        if grid.shape[0] != feat.shape[0]:
            raise ValueError("inconsistent batch dimension")

    feats_sampled = []

    for feat in feats:

        feat_sampled = F.grid_sample(
            feat,
            grid,
            mode=interp_mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )  # (N, C, 1, V)

        feat_sampled = feat_sampled.squeeze(dim=2).transpose(1, 2)  # (N, V, C)
        feats_sampled.append(feat_sampled)

    feats_sampled = torch.cat(feats_sampled, dim=2)  # (N, V, sum(C))
    return feats_sampled


from torch.utils.checkpoint import checkpoint


class InputEmbed(nn.Module):
    def __init__(self, cfg, outdim):
        super().__init__()
        print(f"> InputEmbed: {cfg.Embeding_backbone}")  
        if cfg.Embeding_backbone == "hrnet48":  
            self.EmbedBackbone = get_seg_model(cfg)
        else:
            raise NotImplementedError

        self.outdim = outdim

        self.EmbedDim = _FEAT_DIMS[cfg.Embeding_backbone]
 
   
        self.bottleneck = nn.Sequential(
                nn.Linear(sum(self.EmbedDim) * 2, cfg.embed_interdim),
                nn.LeakyReLU(),
                nn.Linear(cfg.embed_interdim, self.outdim),
            )

        self.vert_align_func = vert_align_optimized

    def forward(self, stage_list, is_mulpts_sampling=True, pool_method: str = 'max_avg_cat'):
        image = stage_list["img"]
        x = stage_list["ct"]
        mulpts_indiv_segs_tensor = stage_list["img_spts"]

        device, dtype = x.device, x.dtype        

        img_feats = self.EmbedBackbone(image)
        
        if not is_mulpts_sampling:

            factor = torch.tensor([1, 1], device=device, dtype=dtype).view(1, 1, 2)  
            xy_norm = x * factor  
            vert_align_feats = vert_align_custom(img_feats, xy_norm)
 
        else:
            vert_align_feats = self.vert_align_func(img_feats, mulpts_indiv_segs_tensor, pool_method=pool_method, chunk_size=256)  # (Total_N, 2*sum(C)) / (Total_N, sum(C))

        vert_align_feats = self.bottleneck(vert_align_feats)                   

        stage_list.update({
                "img_vert_align_feats": vert_align_feats,   
            })

        return stage_list


