"""
HiConSpot Model
Pointcept detached version

Author: Han Lin (hcl0jf26@gmail.com)
Please cite our work if the code is helpful to you.
"""

import torch
import torch.nn as nn
from ...PTv3.model import *
from torch_scatter import scatter_max, scatter_mean, scatter_softmax, scatter_sum 
from ...utils.dxfgnn_utils import ConnectivityFEV1
from ....ImageEmbed.inputEmbed import InputEmbed




class PTv3BackboneV2(nn.Module):

    def __init__(
            self, cfg, 
            cls_mode=False,         
            func=NativeSubMConv,    
            funcmode=None,  
            aggregation_mode="add",
            en_CFE=False,
            cfg_gnn=None,
            grid_size=None,
            en_checkpoint=False,
        ):

        super().__init__()
        self.in_planes = cfg.in_channels
        pz = cfg.patch_size  
        self.en_CFE = cfg.en_CFE if hasattr(cfg, 'en_CFE') else en_CFE
        self.grid_size = grid_size if grid_size is not None else 0.0005
        self.en_checkpoint = en_checkpoint

        self.model = PointTransformerV3_2d(
            in_channels=cfg.in_channels,
            order=("z", "z-trans", "hilbert", "hilbert-trans"),
            stride=cfg.stride,                    
            enc_depths=cfg.enc_depths,              
            enc_channels=cfg.enc_channels,          
            enc_num_head=(2, 4, 8, 16, 32),         
            enc_patch_size=(pz, pz, pz, pz, pz),  

            dec_depths=(2, 2, 2, 2),            
            dec_channels=(64, 64, 128, 256),    
            dec_num_head=(4, 4, 8, 16),         
            dec_patch_size=(pz, pz, pz, pz),  
            func=func,
            funcmode=funcmode,
            cls_mode = cls_mode,
            enable_rpe=True,  
            enable_flash=False,  
            upcast_attention=True,  
            upcast_softmax=True,  
        )

        self.planes = list(cfg.dec_channels) + [cfg.enc_channels[-1]]  
        self.cls_mode = cls_mode

        self.aggregation_mode = aggregation_mode
        if self.aggregation_mode == "cat":
            self.fc1 = nn.Sequential(
                nn.Linear(self.planes[0]*2, self.planes[0]),
                nn.LeakyReLU(),
            )

        self.pool_mode = cfg.pool_mode if hasattr(cfg, 'pool_mode') else None
        if self.pool_mode == "learnable":
            self.pooling_linear_mean = MLP(self.planes[0], 1, norm_fn=nn.LayerNorm, num_layers=3) 
            self.pooling_linear_max = MLP(self.planes[0], 1, norm_fn=nn.LayerNorm, num_layers=3)  
        
        if self.en_CFE:
            self.CFE = ConnectivityFEV1(in_channels=self.planes[0], hidden_dim=self.planes[3], cfg_gnn=cfg_gnn)
        else:
            self.CFE = None 

    def primitive_mixed_pooling(self, point_feat, point_coords, unique_point_ids, aggregation_mode="add"):

  
        num_primitives = unique_point_ids.max().item() + 1


        max_feat, _ = scatter_max(point_feat, unique_point_ids, dim=0, dim_size=num_primitives)


        mean_feat = scatter_mean(point_feat, unique_point_ids, dim=0, dim_size=num_primitives)


        if aggregation_mode == "add":
            primitive_feat = max_feat + mean_feat
        elif aggregation_mode == "cat":
            primitive_feat = torch.cat([max_feat, mean_feat], dim=-1) 
        else:
            primitive_feat = max_feat + mean_feat

        primitive_coords = scatter_mean(point_coords, unique_point_ids, dim=0, dim_size=num_primitives)

        return primitive_feat, primitive_coords
    
    def primitive_mixed_pooling_learnable(self, point_feat, point_coords, unique_point_ids):

        num_primitives = unique_point_ids.max().item() + 1
        
        bg_mean = scatter_mean(point_feat, unique_point_ids, dim=0, dim_size=num_primitives)

        rel_feat_mean = bg_mean[unique_point_ids] - point_feat

        weights_mean = scatter_softmax(self.pooling_linear_mean(rel_feat_mean), unique_point_ids, dim=0)

        feat_mean_weighted = scatter_sum(point_feat * weights_mean, unique_point_ids, dim=0, dim_size=num_primitives)


        bg_max, _ = scatter_max(point_feat, unique_point_ids, dim=0, dim_size=num_primitives)  
        rel_feat_max = bg_max[unique_point_ids] - point_feat  
        weights_max = scatter_softmax(self.pooling_linear_max(rel_feat_max), unique_point_ids, dim=0)  
        feat_max_weighted = scatter_sum(point_feat * weights_max, unique_point_ids, dim=0, dim_size=num_primitives)  

        if self.aggregation_mode == "add":
            primitive_feat = feat_mean_weighted + feat_max_weighted
        elif self.aggregation_mode == "cat":
            primitive_feat = torch.cat([feat_mean_weighted, feat_max_weighted], dim=-1) 
        else:
            primitive_feat = feat_mean_weighted + feat_max_weighted

        primitive_coords = scatter_mean(point_coords, unique_point_ids, dim=0, dim_size=num_primitives)
        
        return primitive_feat, primitive_coords

    def forward(self, stage_list):

        p0 = stage_list["inputs"]["p_out"]  # (n, 3)，coords
        x0 = stage_list["inputs"]["f_out"]  # (n, c),feats
        o0 = stage_list["inputs"]["offset"]   # (b)，offsets
        x0 = p0 if self.in_planes == 3 else torch.cat((p0, x0), 1)  # x0(max_num, 9)


        entity_offset = stage_list["entity_offset"]        
        unique_point_ids = stage_list["unique_point_ids"]  


        counts = torch.diff(o0, prepend=torch.tensor([0], device=o0.device))

        data_dict = {
            "coord": p0,
            "feat": x0,
            "offset": o0,
            "grid_size": self.grid_size  
        }


        point = Point(data_dict)
        point.serialization(order=self.model.order, shuffle_orders=self.model.shuffle_orders)

        point.sparsify_2d()

        point = self.model.embedding(point)


        up_list = []
        enc_outputs = []
        for i, layer in enumerate(self.model.enc):
            point = layer(point)
            if self.cls_mode:
                up_list.append({
                    'p_out': point.coord,
                    'f_out': point.feat,
                    'offset': point.offset.int()
                })

        if not self.cls_mode:
            for i, layer in enumerate(self.model.dec):

                up_list.append({
                    'p_out': point.coord,
                    'f_out': point.feat,
                    'offset': point.offset.int()
                })
                point = layer(point)


            up_list.append({
                'p_out': point.coord,
                'f_out': point.feat,
                'offset': point.offset.int()
            })

  
            if torch.allclose(p0, point.coord, atol=1e-5):

                final_feat = point.feat
                final_coord = point.coord
            else:
   
                inv_idx = point.serialized_inverse[0] 
                final_feat = point.feat[inv_idx]
                final_coord = point.coord[inv_idx]

            if self.pool_mode == "learnable":
                primitive_feat, primitive_coords = self.primitive_mixed_pooling_learnable(
                    final_feat, 
                    final_coord, 
                    unique_point_ids,
                )
            else:
                primitive_feat, primitive_coords = self.primitive_mixed_pooling(
                    final_feat, 
                    final_coord, 
                    unique_point_ids,
                    aggregation_mode=self.aggregation_mode,
                )

            if self.aggregation_mode == "cat":
                primitive_feat = self.fc1(primitive_feat)

            if self.en_CFE:
                primitive_feat = self.CFE(
                    primitive_feat, 
                    map_list=stage_list.get('maps_list'), 
                )

            up_list.reverse()


            stage_list.update({
                "primitive_features": primitive_feat,   
                "primitive_coords": primitive_coords,   
                "entity_offset": entity_offset,         
            })

        stage_list['up'] = up_list
        return stage_list


class AddModule(nn.Module):
    def forward(self, feats):
        res = sum(feats)
        return res
    
class FusionModule(nn.Module):
    def __init__(self, in_channels, FF_mode='cat', use_residual=False):
        super().__init__()
        self.FF_mode = FF_mode
        if self.FF_mode == "cat":
            self.fc_cat = nn.Sequential(
                nn.LayerNorm(in_channels * 2),
                nn.Linear(in_channels * 2, in_channels * 2),
                nn.Linear(in_channels * 2, in_channels * 4),
                nn.GELU(),
                nn.Linear(in_channels * 4, in_channels),
            )
        elif self.FF_mode == 'add':
            self.FF = AddModule()
        else:
            self.FF = AddModule()
    
    def forward(self, stage_list):
        primitive_features = stage_list["primitive_features"] 
        img_vert_align_feats = stage_list["img_vert_align_feats"] 

        if self.FF_mode == "cat":
            fused_feats = torch.cat([primitive_features, img_vert_align_feats], dim=-1)
            fused_feats = self.fc_cat(fused_feats)
        else:
            fused_feats = self.FF([primitive_features, img_vert_align_feats])
        
        stage_list.update({
                "primitive_features": fused_feats,                  
                "backbone_primitive_features": primitive_features,  
            })

        return stage_list



class dxfBackbone(nn.Module):
    def __init__(self, cfg, FF_mode='cat', use_residual=False, cls_mode=False, grid_size=None):
        super().__init__()
        self.backbone = PTv3BackboneV2(cfg, cls_mode=cls_mode, grid_size=grid_size)  
        self.img_backbone = InputEmbed(
            cfg, outdim=self.backbone.planes[0], max_points=None, enable_instant_pooling=True, pool_method='max_avg_cat', sampling_func="opm")

        self.in_channels = self.backbone.planes[0]
        self.FF = FusionModule(in_channels=self.in_channels, FF_mode=FF_mode, use_residual=use_residual)
        
    
    def forward(self, stage_list):
        stage_list = self.backbone(stage_list)  
        stage_list = self.img_backbone(stage_list)
        stage_list = self.FF(stage_list)

        return stage_list



