"""
HiConSpot Model
Pointcept detached version

Author: Han Lin (hcl0jf26@gmail.com)
Please cite our work if the code is helpful to you.
"""
from collections import defaultdict

from torch.nn import functional as F
import torch
import torch.nn as nn
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np

from torch.utils.checkpoint import checkpoint


def _sample_and_pool_chunk(grid_chunk, img_feats_list, pool_method, align_corners, V_chunk, P):

    sampled_layers = []
    
    for feat_map in img_feats_list:

        s = F.grid_sample(feat_map.float(), grid_chunk.float(), 
                          mode='bilinear', padding_mode='zeros', 
                          align_corners=align_corners).to(feat_map.dtype)
        
        s = s.squeeze(2).transpose(1, 2).contiguous().view(V_chunk, P, -1)
        sampled_layers.append(s)


    sample_feat_chunk = torch.cat(sampled_layers, dim=-1)


    if pool_method == 'max_avg_cat':
        f_max = sample_feat_chunk.max(dim=1)[0]
        f_avg = sample_feat_chunk.mean(dim=1)
        res = torch.cat([f_max, f_avg], dim=-1)
    elif pool_method == 'max':
        res = sample_feat_chunk.max(dim=1)[0]
    else:  
        res = sample_feat_chunk.mean(dim=1)
        
    return res.contiguous()


def vert_align_optimized(
        img_feats: List[torch.Tensor],
        sample_points_list: List[torch.Tensor],
        pool_method: str = 'max_avg_cat',
        align_corners: bool = True,
        chunk_size: int = 512,   
        enable_checkpoint: bool = False):

    batch_size = img_feats[0].shape[0]
    device = img_feats[0].device
    all_sample_results = []


    for b in range(batch_size):
        pts_b = sample_points_list[b].to(device) 
        V_i, P, _ = pts_b.shape
        

        cur_img_feats = [f[b:b+1] for f in img_feats]
        
        sample_b_results = []

         
        for v_start in range(0, V_i, chunk_size):
            v_end = min(v_start + chunk_size, V_i)
            v_len = v_end - v_start
            
            chunk_pts = pts_b[v_start:v_end] 
            grid_chunk = chunk_pts.clone()
            grid_chunk[..., 0] = 2 * grid_chunk[..., 0] - 1
            grid_chunk[..., 1] = -1 * (2 * grid_chunk[..., 1] - 1)
            grid_chunk = grid_chunk.view(1, 1, -1, 2).contiguous()

 
            is_recomputing = enable_checkpoint and img_feats[0].requires_grad
            
            if is_recomputing:

                res_chunk = checkpoint(
                    _sample_and_pool_chunk,
                    grid_chunk, cur_img_feats, pool_method, align_corners, v_len, P,
                    use_reentrant=False
                )
            else:
                res_chunk = _sample_and_pool_chunk(
                    grid_chunk, cur_img_feats, pool_method, align_corners, v_len, P
                )
            
            sample_b_results.append(res_chunk)
            

        all_sample_results.append(torch.cat(sample_b_results, dim=0))


    return torch.cat(all_sample_results, dim=0)

