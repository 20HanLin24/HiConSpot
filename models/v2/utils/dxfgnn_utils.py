"""
HiConSpot Model
Pointcept detached version

Author: Han Lin (hcl0jf26@gmail.com)
Please cite our work if the code is helpful to you.
"""

 
import torch
import numpy as np
from torch_scatter import scatter_max, scatter_mean
from collections import defaultdict, deque

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def get_hierarchical_connectivity_groups_fastV2(num_nodes, res_with_edge, edge_features):

    flat_nns = np.concatenate(res_with_edge['thresh_nns'])               
    flat_dists = np.concatenate(edge_features['dist'])                   
    flat_rels = np.array(np.concatenate(edge_features['relation']))      
    
    counts = [len(x) for x in res_with_edge['thresh_nns']]
    source_idx = np.repeat(np.arange(num_nodes), counts)   

    is_touching = flat_dists < 1e-5  
    
    mask_l1 = is_touching

    match_l2 = np.all(flat_rels == [0, 0, 0, 1, 0, 0], axis=1)
    
    match_l3 = np.all(flat_rels == [0, 1, 0, 0, 0, 0], axis=1)
    
    match_l4 = np.all(flat_rels == [1, 0, 0, 0, 0, 0], axis=1)

    mask_l2 = is_touching & (match_l2 | match_l4)
    mask_l3 = is_touching & (match_l3 | match_l4)
    mask_l4 = is_touching & match_l4

    def get_labels(mask):
        u = source_idx[mask]
        v = flat_nns[mask]

        data = np.ones(len(u))
  
        adj_matrix = csr_matrix((data, (u, v)), shape=(num_nodes, num_nodes))

        _, labels = connected_components(csgraph=adj_matrix, directed=False, return_labels=True)
        return torch.from_numpy(labels)

    map_l1 = get_labels(mask_l1)
    map_l2 = get_labels(mask_l2)
    map_l3 = get_labels(mask_l3)
    map_l4 = get_labels(mask_l4)

    return map_l1, map_l2, map_l3, map_l4  


from torch_scatter import scatter_max, scatter_mean, scatter_softmax, scatter_sum
import torch.nn as nn
import torch.nn.functional as F
    

class MLP(nn.Sequential):
    def __init__(self, in_channels, out_channels, norm_fn=None, num_layers=2):
        modules = []
        for _ in range(num_layers - 1):
            modules.append(nn.Linear(in_channels, in_channels))
            if norm_fn:
                modules.append(norm_fn(in_channels))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(in_channels, out_channels))
        return super().__init__(*modules)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        nn.init.normal_(self[-1].weight, 0, 0.01)
        nn.init.constant_(self[-1].bias, 0)

    
class ConnectivityFEV1(nn.Module):
    def __init__(self, in_channels, hidden_dim):

        super().__init__()
        self.hidden_dim = hidden_dim

        self.attn_score = nn.Sequential(
            nn.Linear(in_channels, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        self.group_fc = nn.Sequential(
            nn.Linear(in_channels * 3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, in_channels)
        )
        
        self.level_weights = nn.Parameter(torch.tensor([0.2, 0.6, 0.5, 1.0]))
        
        self.final_fuse = nn.Linear(in_channels * 2, in_channels)

    def pool_group_features(self, x, group_map):

         
        m_max, _ = scatter_max(x, group_map, dim=0)  
        
         
        m_mean = scatter_mean(x, group_map, dim=0)  
        
         
        scores = self.attn_score(x)  
        weights = scatter_softmax(scores, group_map, dim=0)  
        m_attn = scatter_sum(x * weights, group_map, dim=0)  
        
        combined = torch.cat([m_max, m_mean, m_attn], dim=-1)
        group_feat = self.group_fc(combined)  
        
        return group_feat[group_map]

    def forward(self, x, map_list):
   
        hierarchical_enhancement = torch.zeros_like(x)
        for i, g_map in enumerate(map_list):
            g_map_cuda = g_map.to(x.device).long()

            level_context = self.pool_group_features(x, g_map_cuda)

            hierarchical_enhancement += self.level_weights[i] * level_context
            
        x_struct = x

        final_x = self.final_fuse(torch.cat([hierarchical_enhancement, x_struct], dim=-1))
        
        return final_x
    
