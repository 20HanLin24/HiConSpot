import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .....model.v2.PTv3.utils.point_class import *
from .....model.v2.HiConSpot.tools.pointtransformer_utils import *
from .....model.v2.HiConSpot.tools.pointops.functions import pointops

class NativeSubMConv(PointModule):
    """
    原生 PyTorch 实现的子流形稀疏卷积 (Submanifold Sparse Convolution)
    兼容 2D 和 3D，用于替代不稳定的 spconv。dimension=3为3d，2为2d。
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, dimension=3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dimension = dimension  

        self.num_spatial = kernel_size ** dimension
        self.weight = nn.Parameter(torch.randn(self.num_spatial, in_channels, out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, point):
         
        feat = point.feat   
        grid_coord = point.grid_coord[:, :self.dimension].long()   
        batch = point.batch.long()  
        
        N = feat.shape[0]
        device = feat.device
  
        if self.dimension == 2:
            coord_hash = (batch << 40) | (grid_coord[:, 0] << 20) | grid_coord[:, 1]
        else:
            coord_hash = (batch << 42) | (grid_coord[:, 0] << 28) | (grid_coord[:, 1] << 14) | grid_coord[:, 2]

        sorted_hash, sorted_idx = torch.sort(coord_hash)

        offset_range = torch.arange(-(self.kernel_size // 2), self.kernel_size // 2 + 1, device=device)
        offsets = torch.stack(torch.meshgrid([offset_range] * self.dimension, indexing='ij'), dim=-1).reshape(-1, self.dimension)

        out_feat = torch.zeros(N, self.out_channels, device=device)

         
        for i in range(self.num_spatial):
            rel_offset = offsets[i]  

            neighbor_coord = grid_coord + rel_offset
            if self.dimension == 2:
                neighbor_hash = (batch << 40) | (neighbor_coord[:, 0] << 20) | neighbor_coord[:, 1]
            else:
                neighbor_hash = (batch << 42) | (neighbor_coord[:, 0] << 28) | (neighbor_coord[:, 1] << 14) | neighbor_coord[:, 2]

            indices = torch.searchsorted(sorted_hash, neighbor_hash)
             
            indices = indices.clamp(max=N - 1)
            found_mask = (sorted_hash[indices] == neighbor_hash)
            
            if found_mask.any():
                neighbor_idx = sorted_idx[indices[found_mask]]
                 
                matched_feat = feat[neighbor_idx] @ self.weight[i]
                 
                out_feat[found_mask] += matched_feat

        out_feat += self.bias
        point.feat = out_feat
        return point
    

 
 
 
 
 
        
 
 
 
 
 
 
 
 
        
 
 
 
 
 

 
 
 
        
 
 
 
 
 
        
 
 
 
        
 
 
        
 
 
 
        
 
 
 


class PointFA(PointModule):
    """点特征聚合模块用于替代spconv模块, 可以选择不同的局部特征聚合方式"""
    def __init__(self, in_channels, channels, kernel_size=3, dimension=2, mode="maxavgpool", share_planes=8):
        """channels为输入输出维度, nsample为采样点数, 相当于spconv卷积核的大小"""
        super().__init__()
        nsample = kernel_size ** dimension   
        self.mode = mode
        self.nsample = nsample
        self.expansion = 1
        self.out_channels = channels

        if mode == "ptLayer":
             
             
            self.fc1 = nn.Sequential(
                nn.Linear(in_channels, channels, bias=False),   
                nn.BatchNorm1d(channels),
                nn.ReLU(inplace=True),
            )
            self.aggregator = nn.Sequential(
                PointTransformerLayer(
                    in_planes=channels, 
                    out_planes=channels, 
                    share_planes=share_planes, 
                    nsample=nsample
                ),
            )
            self.fc2 = nn.Sequential(
                nn.BatchNorm1d(channels),
                nn.ReLU(inplace=True),
                nn.Linear(channels, channels * self.expansion, bias=False),
                nn.BatchNorm1d(channels * self.expansion),
                nn.ReLU(inplace=True)
            )
        elif mode == "weight":
            if in_channels != channels:
                self.pre_proj = nn.Linear(in_channels, channels)
            else:
                self.pre_proj = nn.Identity()
            self.weight_generator = nn.Sequential(
                nn.Linear(3, channels // 4),
                nn.ReLU(inplace=True),
                nn.Linear(channels // 4, channels)
            )
            self.post_mlp = nn.Sequential(
                nn.BatchNorm1d(channels),
                nn.ReLU(inplace=True),
                nn.Linear(channels, channels)
            )
        else:  
            self.pre_proj = nn.Linear(in_channels, channels) if in_channels != channels else nn.Identity()
            self.mlp_pool = nn.Sequential(
                nn.Linear(2 * channels, channels),
                nn.BatchNorm1d(channels),
                nn.ReLU(inplace=True),
                nn.Linear(channels, channels)
            )

    def forward(self, point: Point):
        with torch.amp.autocast('cuda', enabled=False):   
            if self.mode == "ptLayer":
                 
                 
                p, x, o = point.coord.float(), point.feat.float(), point.offset.int()
                x = self.fc1(x)
                 
                pxo = [p, x, o, None]
                new_feat, knn_idx = self.aggregator(pxo)
                new_feat = self.fc2(new_feat)
                
                 
                point.feat = new_feat
            
            elif  self.mode == "weight":   
                coord, feat, offset = point.coord.float(), point.feat.float(), point.offset.int()
                 
                 
                idx, _ = pointops.knnquery(self.nsample, coord, coord, offset, offset)
                
                 
                 
                feat_projected = self.pre_proj(feat)
                grouped_feat = pointops.grouping(feat_projected, idx)
                 
                 
                 
                m, nsample = idx.shape
                grouped_xyz = coord[idx.view(-1).long(), :].view(m, nsample, 3)
                relative_xyz = grouped_xyz - coord.unsqueeze(1)  
                
                 
                 
                weights = F.softmax(self.weight_generator(relative_xyz), dim=1)

                 
                 
                local_feat = torch.sum(weights * grouped_feat, dim=1)
                point.feat = self.post_mlp(local_feat)

            else:
                coord, feat, offset = point.coord.float(), point.feat.float(), point.offset.int()
                 
                 
                idx, _ = pointops.knnquery(self.nsample, coord, coord, offset, offset)
                
                 
                 
                grouped_feat = pointops.grouping(feat, idx)
                 
                local_feat1 = grouped_feat.mean(dim=1) 
                local_feat2 = grouped_feat.max(dim=1)[0]

                local_feat = torch.cat([local_feat1, local_feat2], dim=1)
                local_feat = self.mlp_pool(local_feat)
                
                point.feat = local_feat
        
        return point


