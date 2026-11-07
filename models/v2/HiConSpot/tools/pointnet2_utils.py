"""
Author: Haoxi Ran
Date: 06/30/2022
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .....model.v2.HiConSpot.tools.pointops.functions import pointops


def sample_and_group(stride, nsample, xyz, points, offset, return_idx=False, num_sector=1):
     
    if stride > 1:
        new_offset, sample_idx = [offset[0].item() // stride], offset[0].item() // stride
        for i in range(1, offset.shape[0]):
            sample_idx += (offset[i].item() - offset[i - 1].item()) // stride
            new_offset.append(sample_idx)
        new_offset = torch.cuda.IntTensor(new_offset)
        if num_sector > 1:
            fps_idx = pointops.sectorized_fps(xyz, offset, new_offset, num_sector)   
        else:
            fps_idx = pointops.furthestsampling(xyz, offset, new_offset)   
        new_xyz = xyz[fps_idx.long(), :]   
    else:
        new_xyz = xyz
        new_offset = offset

     
    N, M = xyz.shape[0], new_xyz.shape[0]
    group_idx, _ = pointops.knnquery(nsample, xyz, new_xyz, offset, new_offset)   
    group_xyz = xyz[group_idx.view(-1).long(), :].view(M, nsample, 3)   
    group_xyz_norm = group_xyz - new_xyz.unsqueeze(1)

    if points is not None and not return_idx:
        C = points.shape[1]
        group_points = points[group_idx.view(-1).long(), :].view(M, nsample, C)
        new_points = torch.cat([group_xyz_norm, group_points], dim=-1)   
    else:
        new_points = group_xyz_norm

    if return_idx:
        return new_xyz, new_points, new_offset, group_idx
    else:
        return new_xyz, new_points, new_offset


class PointNetSetAbstraction(nn.Module):
    """
    PointNet2 SA Module

    """

    def __init__(self, stride, nsample, in_channel, mlp, num_sector=1):
        super(PointNetSetAbstraction, self).__init__()
        self.stride = stride
        self.nsample = nsample
        self.num_sector = num_sector
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, pos_feat_off):
        xyz, points, offset = pos_feat_off   

        new_xyz, new_points, new_offset = sample_and_group(self.stride, self.nsample, xyz, points, offset,
                                                           num_sector=self.num_sector)

         
         
        new_points = new_points.transpose(1, 2).contiguous()   
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        new_points = torch.max(new_points, 2)[0]

        return [new_xyz, new_points, new_offset]


class PointNetFeaturePropagation(nn.Module):
    """
    PointNet2 FP Module

    """

    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Linear(last_channel, out_channel))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, pos_feat_off1, pos_feat_off2):
        xyz1, points1, offset1 = pos_feat_off1   
        xyz2, points2, offset2 = pos_feat_off2   

        idx, dist = pointops.knnquery(3, xyz2, xyz1, offset2, offset1)   
        dist_recip = 1.0 / (dist + 1e-8)   
        norm = torch.sum(dist_recip, dim=1, keepdim=True)
        weight = dist_recip / norm   
        
        interpolated_points = torch.cuda.FloatTensor(xyz1.shape[0], points2.shape[1]).zero_()
        for i in range(3):
            interpolated_points += points2[idx[:, i].long(), :] * weight[:, i].unsqueeze(-1)


         
        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=1)   
        else:
            new_points = interpolated_points

         
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        return new_points
