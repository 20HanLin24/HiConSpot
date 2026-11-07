import glob
import math
import os
import os.path as osp
import sys


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
 
sys.path.insert(0, project_root)

from glob import glob
import torch

import math
import os
import os.path as osp

import numpy as np
import torch

import argparse


import shutil
import time

import time



from ..v2.HiConSpot.utils.logger import get_root_logger

from ...configs.config import _C as config
from ...configs.config import update_config

import torchvision.transforms as T
from torch.utils.data import Dataset

import os.path as osp



from .svgnet_aug.aug_utils import *
from .mul_points_sampling import split_point_sets_to_individual_tensors, get_sampled_points_by_ndft


import torchvision.transforms as transforms
from PIL import Image

import random
import numpy as np

from ..v2.utils.dxfgnn_utils import  get_hierarchical_connectivity_groups_fastV2


LABEL_NUM = 35
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


 
SVG_CATEGORIES = [
     
    {"color": [224, 62, 155], "isthing": 1, "id": 0, "name": "single door"},
    {"color": [157, 34, 101], "isthing": 1, "id": 1, "name": "double door"},
    {"color": [232, 116, 91], "isthing": 1, "id": 2, "name": "sliding door"},
    {"color": [101, 54, 72], "isthing": 1, "id": 3, "name": "folding door"},
    {"color": [172, 107, 133], "isthing": 1, "id": 4, "name": "revolving door"},
    {"color": [142, 76, 101], "isthing": 1, "id": 5, "name": "rolling door"},
     
    {"color": [96, 78, 245], "isthing": 1, "id": 6, "name": "window"},
    {"color": [26, 2, 219], "isthing": 1, "id": 7, "name": "bay window"},
    {"color": [63, 140, 221], "isthing": 1, "id": 8, "name": "blind window"},
    {"color": [233, 59, 217], "isthing": 1, "id": 9, "name": "opening symbol"},
     
    {"color": [122, 181, 145], "isthing": 1, "id": 10, "name": "sofa"},
    {"color": [94, 150, 113], "isthing": 1, "id": 11, "name": "bed"},
    {"color": [66, 107, 81], "isthing": 1, "id": 12, "name": "chair"},
    {"color": [123, 181, 114], "isthing": 1, "id": 13, "name": "table"},
    {"color": [94, 150, 83], "isthing": 1, "id": 14, "name": "TV cabinet"},
    {"color": [66, 107, 59], "isthing": 1, "id": 15, "name": "Wardrobe"},
    {"color": [145, 182, 112], "isthing": 1, "id": 16, "name": "cabinet"},
    {"color": [152, 147, 200], "isthing": 1, "id": 17, "name": "gas stove"},
    {"color": [113, 151, 82], "isthing": 1, "id": 18, "name": "sink"},
    {"color": [112, 103, 178], "isthing": 1, "id": 19, "name": "refrigerator"},
    {"color": [81, 107, 58], "isthing": 1, "id": 20, "name": "airconditioner"},
    {"color": [172, 183, 113], "isthing": 1, "id": 21, "name": "bath"},
    {"color": [141, 152, 83], "isthing": 1, "id": 22, "name": "bath tub"},
    {"color": [80, 72, 147], "isthing": 1, "id": 23, "name": "washing machine"},
    {"color": [100, 108, 59], "isthing": 1, "id": 24, "name": "squat toilet"},
    {"color": [182, 170, 112], "isthing": 1, "id": 25, "name": "urinal"},
    {"color": [238, 124, 162], "isthing": 1, "id": 26, "name": "toilet"},
     
    {"color": [247, 206, 75], "isthing": 1, "id": 27, "name": "stairs"},
     
    {"color": [237, 112, 45], "isthing": 1, "id": 28, "name": "elevator"},
    {"color": [233, 59, 46], "isthing": 1, "id": 29, "name": "escalator"},

     
    {"color": [172, 107, 151], "isthing": 0, "id": 30, "name": "row chairs"},
    {"color": [102, 67, 62], "isthing": 0, "id": 31, "name": "parking spot"},
    {"color": [167, 92, 32], "isthing": 0, "id": 32, "name": "wall"},
    {"color": [121, 104, 178], "isthing": 0, "id": 33, "name": "curtain wall"},
    {"color": [64, 52, 105], "isthing": 0, "id": 34, "name": "railing"},
    {"color": [0, 0, 0], "isthing": 0, "id": 35, "name": "bg"},   
]


def set_seed(seed):
    np.random.seed(seed)    
    random.seed(seed)        
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  

    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False

def imagenet_preprocess():
    return T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def angles_with_horizontal(coords):
    coords = np.array(coords).reshape(-1, 4, 2)
    start_points, end_points = coords[:, 0, :], coords[:, -1, :]   
    lines = np.concatenate([start_points[:, None, :],
                            end_points[:, None, :]], axis=1)   
    x1, y1 = lines[:, 0, 0], lines[:, 0, 1]   
    x2, y2 = lines[:, 1, 0], lines[:, 1, 1]   
    slopes = (y2 - y1) / (x2 - x1 + 1e-8)   
    slopes[x2 - x1 == 0] = np.inf   
    theta_radians = np.arctan(slopes)   
    return theta_radians   

def transform_to_ptv3_input_vectorized(fixed_length_points, nd_ft_list, centers):

     
    pts = torch.as_tensor(fixed_length_points, dtype=torch.float32)
    nd_ft = torch.as_tensor(nd_ft_list, dtype=torch.float32)
    centers = torch.as_tensor(centers, dtype=torch.float32)
    V, M, _ = pts.shape     
     
    mask = (pts[:, :, 0] != -1.0) | (pts[:, :, 1] != -1.0)
     
    relative_offsets = pts - centers.unsqueeze(1)
    rel_off_final = relative_offsets[mask]  
     
    seq_indices = torch.arange(M, device=pts.device).float() / float(M)   
    seq_enc_final = seq_indices.unsqueeze(0).expand(V, -1)[mask].unsqueeze(1)  
     
    valid_pts = pts[mask]   
    coords = torch.cat([valid_pts, torch.zeros((valid_pts.shape[0], 1), device=valid_pts.device)], dim=1) 
     
    radians = torch.atan2(valid_pts[:, 1], valid_pts[:, 0] + 1e-8) / math.pi
    radians = radians.unsqueeze(1)  
     
    lengths = nd_ft[:, 0:1]  
    one_hots = nd_ft[:, 7:]  
     
    entity_indices = (torch.arange(V, device=pts.device))
    entity_indices = entity_indices.float().unsqueeze(1)
     
    def broadcast_to_points(feature_v):
         
        expanded = feature_v.unsqueeze(1).expand(-1, M, -1)
        return expanded[mask]

    point_lengths = broadcast_to_points(lengths)       
    point_one_hots = broadcast_to_points(one_hots)     
    point_entity_ids = broadcast_to_points(entity_indices)  

    feats = torch.cat([
        rel_off_final,   
        seq_enc_final,   
        radians,         
        point_lengths,   
        point_one_hots,  
    ], dim=1)

    return coords, feats, point_entity_ids




class HiConSpotDataset_svg(Dataset):
    """增加了预处理, 连通层提取处理"""
    CLASSES = tuple([x["name"] for x in SVG_CATEGORIES])

    def __init__(self, split='train', cfg=None, aug=None, max_prim=20000, root=None,
                 data_norm="min", logger=None, do_aug=None, repeat=1, do_filter=True, size=800, is_mulpts_sampling=False, 
                 process_dir_name=None, max_points=49, do_process=True):
        self.root = root             
        self.max_prim = max_prim     
        self.do_filter = do_filter   

        self.split = split   
        self.data_norm = data_norm   
        self.aug = aug
        self.do_aug = do_aug
        self.repeat = repeat 

        if cfg is not None:
            if hasattr(cfg, root):
                self.root = cfg.root   
            else:
                if split == 'train':
                    self.root = cfg.data.train.data_root
                    self.aug = cfg.data.train.aug
                    self.aug_ratio = self.aug.aug_prob
                else:
                    self.root = cfg.data.test.data_root
                    self.aug = cfg.data.test.aug
                    self.aug_ratio = 0

            self.filter_num = cfg.filter_num   
            self.do_aug = cfg.do_aug
            self.size = size
            self.set_random_seed(cfg.seed)   
        else:
            self.set_random_seed(123)   
            self.filter_num = 16
            self.aug = None
            self.aug_ratio = 0.5
            self.do_aug = False
            self.size = 800

        self.max_size = 0
         
        self.anno_path_list = glob(os.path.join(self.root, "npy", split, "*.npy"))    
        self.anno_path_list = sorted(self.anno_path_list)
         
        self.image_path_list = glob(os.path.join(self.root, "png", split, "*.png"))   
        self.image_path_list = sorted(self.image_path_list)
        self.img_dir = os.path.join(self.root, "png", split)

        if self.do_filter:   
            logger.info(f" > before filter_smallset: {len(self.anno_path_list)}")   
            npy_save_path = os.path.join(self.root,
                                     f"filtered_file_path_list_{split}_flt{self.filter_num}_max{max_prim}.npy")
            if not os.path.exists(npy_save_path):   
                self.anno_path_list = self.filter_smallset_npy()
                filtered_file_path_list_dict = {"npy_path_list": self.anno_path_list}
                np.save(npy_save_path, filtered_file_path_list_dict)
            else:
                temp_data = np.load(npy_save_path, allow_pickle=True).item()   
                self.anno_path_list = temp_data["npy_path_list"]

        logger.info(f" > after filter_smallset: {len(self.anno_path_list)}")
        if logger:
            logger.info(f"Load {split} dataset located in {os.path.join(self.root, 'npy', split)}: {len(self.anno_path_list)} npy")   

        self.length = len(self.anno_path_list)
        self.instance_queues = []
        self.data_idx = np.arange(self.length)

        self.is_mulpts_sampling = is_mulpts_sampling
         
        self.get_sampled_points = get_sampled_points_by_ndft(dist_sampling=True, step_distance=cfg.step_dist, max_points=max_points)
        self.get_sampled_points_img = get_sampled_points_by_ndft(dist_sampling=True, step_distance=0.005, max_points=max_points)

         
        if do_process:
            self.process_dir = os.path.join(self.root, split, process_dir_name)
            if not os.path.exists(self.process_dir):
                os.makedirs(self.process_dir)
            self.prefile_path_list = [os.path.join(self.process_dir, os.path.basename(npy_path)) for npy_path in
                                        self.anno_path_list]
             
            need_process = any(
                (not os.path.exists(p)) and (not os.path.exists(p.replace(".npy", ".npz"))) for p in self.prefile_path_list
            ) 
            if need_process:
                print("begin process data:")
                 
                self.pre_processV2()
            else:
                print("LOAD process data:")

    def filter_smallset_npy(self):
        anno_path_list_new = []   
        for idx, ann_path in enumerate(self.anno_path_list):
            adj_node_classes = np.load(ann_path, allow_pickle=True).item()   
            target = adj_node_classes["cat"]   
            if self.split == "train":   
                if self.max_prim >= len(target) >= self.filter_num:   
                    anno_path_list_new.append(self.anno_path_list[idx])   
                    if self.max_size < len(target):
                        self.max_size = len(target)
            else:   
                if len(target) >= self.filter_num:   
                    anno_path_list_new.append(self.anno_path_list[idx])   
                    if self.max_size < len(target):
                        self.max_size = len(target)
        return anno_path_list_new

    def __len__(self):
        return self.length * self.repeat
    
    def pre_processV2(self, min_points=3000):
        for idx in range(self.length):
            npy_path = self.anno_path_list[idx]
            basename = os.path.basename(npy_path)   
            prefile_path = os.path.join(self.process_dir, basename)
            if not os.path.exists(prefile_path) and not os.path.exists(prefile_path.replace(".npy", ".npz")):
                print(f"processing: {basename}")
                npydata = np.load(npy_path, allow_pickle=True).item()   

                nodes = np.array(npydata["nd"]) if npydata.get("nd") is not None else np.array(npydata["nd_ft"])
                center = npydata["ct_norm_z2o"]   
                center_norm = np.array(center, dtype=np.float32)

                num = center_norm.shape[0]
                max_num = max(num, min_points)
                 
                classes = npydata["cat"]
                seg = []
                for seg_id in classes:
                    if seg_id == 0:
                        seg.append(LABEL_NUM)
                    elif seg_id > 0:
                        seg.append(seg_id - 1)
                assert len(seg) == len(classes), 'processed semantic id num != orig semantic id num'
                 
                semanticIds = np.full((num, ), LABEL_NUM)
                semanticIds[:] = np.array(seg)
                semanticIds = semanticIds.astype(np.int64)

                instanceIds = np.full((num, ), -1)

                ins = np.array(npydata["inst"])
                
                valid_pos = ins != -1
                ins[valid_pos] += idx * max_num   
                instanceIds[:] = np.array(ins)
                instanceIds = instanceIds.astype(np.int64)

                label = np.concatenate([semanticIds[:, None], instanceIds[:, None]], axis=1)   

                lengths = np.zeros(num)   
                 
                lengths[:] = np.array(nodes[:, 0])   

                res_with_edge_nns_dict = npydata["res_with_edge_nns_dict"]
                edge_features = npydata["edge_features"]
                map_l1, map_l2, map_l3, map_l4 = get_hierarchical_connectivity_groups_fastV2(num, res_with_edge_nns_dict, edge_features)      

                data = {
                        "label": label.astype(np.int16), "lengths": lengths.astype(np.float16),
                        "map_pri2group": [m.numpy().astype(np.int32) for m in (map_l1, map_l2, map_l3, map_l4)],
                    }

                save_path = os.path.join(self.process_dir, basename.replace('.npy', '.npz'))
                np.savez_compressed(save_path, **data)
                print(f"generate npy,{save_path}")
    
    def _get_itemV2(self, index, min_points=3000):
        assert len(self.image_path_list) == len(self.anno_path_list)   
        prefile_path = self.prefile_path_list[index]
        if os.path.exists(prefile_path):
            prefile_data = np.load(prefile_path, allow_pickle=True).item()   
        elif os.path.exists(prefile_path.replace('.npy', '.npz')):
            prefile_data = np.load(prefile_path.replace('.npy', '.npz'), allow_pickle=True)   
        else:
            exit(f"there exist not {prefile_path}")
        basename = os.path.basename(prefile_path).split('.')[0]   

        npy_path = os.path.join(self.root, "npy", self.split, basename + '.npy')
        npydata = np.load(npy_path, allow_pickle=True).item()   
        nd_ft_list = np.array(npydata["nd_ft"], dtype=np.float32).tolist()
        center = npydata["ct_norm_z2o"]   
        center_norm = np.array(center, dtype=np.float32)
        mulpts_segs = self.get_sampled_points.get_fixed_length_pointsets(
                    nd_ft_list=nd_ft_list, max_points=None, arc_type="mid", size=(self.size, self.size))   
        mulpts_segs_img = self.get_sampled_points_img.get_fixed_length_pointsets(
                    nd_ft_list=nd_ft_list, max_points=None, arc_type="mid", size=(self.size, self.size))   
         
        mulpts_indiv_segs_tensor = split_point_sets_to_individual_tensors(mulpts_segs_img)   
         
        coord, feat, point_entity_ids = transform_to_ptv3_input_vectorized(mulpts_segs, nd_ft_list, center_norm)

        center = npydata["ct_norm_z2o"]   
        center = 2 * np.array(center, dtype=np.float32) - 1   
        ct = torch.from_numpy(center)

        label = prefile_data["label"]
        lengths = prefile_data["lengths"]

        map_l1, map_l2, map_l3, map_l4 = prefile_data["map_pri2group"]
         
        num = center_norm.shape[0]
        max_num = max(num, min_points)
         
        classes = npydata["cat"]
        seg = []
        for seg_id in classes:
            if seg_id == 0:
                seg.append(LABEL_NUM)
            elif seg_id > 0:
                seg.append(seg_id - 1)
        assert len(seg) == len(classes), 'processed semantic id num != orig semantic id num'
         
        semanticIds = np.full((num, ), LABEL_NUM)
        semanticIds[:] = np.array(seg)
        assert len(seg) == len(semanticIds), 'semanticIds num != orig semantic id num'
        semanticIds = semanticIds.astype(np.int64)
         
        instanceIds = np.full((num, ), -1)
         
        ins = np.array(npydata["inst"])
        valid_pos = ins != -1
        ins[valid_pos] += index * max_num   
        instanceIds[:] = ins
        assert len(instanceIds) == len(ins), 'instanceIds num != orig ins id num'
        instanceIds = instanceIds.astype(np.int64)
         
        label = np.concatenate([semanticIds[:, None], instanceIds[:, None]], axis=1)   


        img_path = os.path.join(self.img_dir, basename + ".png")
         
        image = Image.open(img_path).convert("RGB")
        if self.split != 'val':
            image = image.resize((self.size, self.size))
         
        trans_tensor = transforms.ToTensor()
        image = trans_tensor(image)

        return coord, feat, label, lengths, basename, point_entity_ids, map_l1, map_l2, map_l3, map_l4, image, ct, mulpts_indiv_segs_tensor

    def __getitem__(self, index):
        data_idx = self.data_idx[index % len(self.data_idx)]   
        xyz, feat, label, lengths, basename, ptids, map_l1, map_l2, map_l3, image, ct, mulpts_indiv_segs_tensor = self._get_itemV2(data_idx)
        if self.split == "train":
            if self.do_aug:
                return self.transform_train_point(
                    xyz, feat, label, lengths, basename, ptids, map_l1, map_l2, map_l3, image, ct, mulpts_indiv_segs_tensor)
            else:
                return self.transform_test(
                    xyz, feat, label, lengths, basename, ptids, map_l1, map_l2, map_l3, image, ct, mulpts_indiv_segs_tensor)
         
        else:
            return self.transform_test(
                xyz, feat, label, lengths, basename, ptids, map_l1, map_l2, map_l3, image, ct, mulpts_indiv_segs_tensor)
     

    def set_random_seed(self, seed, deterministic=False):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True   
            torch.backends.cudnn.benchmark = False   
    
    def transform_train_point(self, coord, feat, label, lengths, basename, point_entity_ids,
                              map_l1, map_l2, map_l3, map_l4, image, ct, mulpts_indiv_segs_tensor):

        if isinstance(coord, torch.Tensor):
            coord = coord.cpu().numpy()
        if isinstance(feat, torch.Tensor):
            feat = feat.cpu().numpy()
        if isinstance(point_entity_ids, torch.Tensor):
            point_entity_ids = point_entity_ids.cpu().numpy()

         
        if self.aug.hflip and np.random.rand() < self.aug_ratio:   
            args = RandomHorizonFilp(coord[:, :2], width=1)
            coord[:, :2] = args
            feat[:, 0] = -feat[:, 0]   

         
        if self.aug.vflip and np.random.rand() < self.aug_ratio:
            args = RandomVerticalFilp(coord[:, :2], Hight=1)
            coord[:, :2] = args
            feat[:, 1] = -feat[:, 1]   

         
        if self.aug.rotate.enable and np.random.rand() < self.aug_ratio:
            _min, _max = self.aug.rotate.angle   
            angle = random.uniform(_min, _max)   
            args = rotate_xy(coord[:, :2], width=1, height=1, angle=angle)
            coord[:, :2] = args

             
            angle_rad = angle * math.pi / 180.0
            c, s = math.cos(angle_rad), math.sin(angle_rad)
            dx, dy = feat[:, 0].copy(), feat[:, 1].copy()
            feat[:, 0] = dx * c - dy * s
            feat[:, 1] = dx * s + dy * c
             

        if self.aug.rotate2 and np.random.rand() < self.aug_ratio:
            args, anglerot2 = random_rotate_pt(coord[:, :2], width=1, height=1)
            coord[:, :2] = args

             
            angle_rad2 = anglerot2 * math.pi / 180.0
            c2, s2 = math.cos(angle_rad2), math.sin(angle_rad2)
            dx2, dy2 = feat[:, 0].copy(), feat[:, 1].copy()
            feat[:, 0] = dx2 * c2 - dy2 * s2
            feat[:, 1] = dx2 * s2 + dy2 * c2
             

         
        if self.aug.shift.enable and np.random.rand() < self.aug_ratio:
            _min, _max = self.aug.shift.scale
            scale = np.random.uniform(_min, _max, 3)
            scale[2] = 0
            coord += scale

         
        if self.aug.scale.enable and np.random.rand() < self.aug_ratio:
            _min, _max = self.aug.scale.ratio
            scale = np.random.uniform(_min, _max, 1)
            coord *= scale
             
            feat[:, :2] *= scale
            feat[:, 4] *= scale

        mix_coord, mix_feat, mix_label = [], [], []
        mix_coord.append(coord)   
        mix_feat.append(feat)     
        mix_label.append(label)   

        coord = np.concatenate(mix_coord, axis=0)   
        feat = np.concatenate(mix_feat, axis=0)     
         
        feat[:, 3] = np.arctan(coord[:, 1] / (coord[:, 0] + 1e-8)) / math.pi   


        label = np.concatenate(mix_label, axis=0)

        shuf_idx = np.arange(coord.shape[0])   
        np.random.shuffle(shuf_idx)            
        coord, feat = coord[shuf_idx], feat[shuf_idx]   

         
        point_entity_ids = point_entity_ids[shuf_idx]  
         
        if self.data_norm == 'mean':
            coord -= np.mean(coord, 0)   
        elif self.data_norm == 'min':
            coord -= np.min(coord, 0)    

         
        return (torch.FloatTensor(coord), torch.FloatTensor(feat), torch.LongTensor(label), None, basename, torch.LongTensor(point_entity_ids),
                map_l1, map_l2, map_l3, map_l4, image, ct, mulpts_indiv_segs_tensor)

    def transform_test(
            self, coord, feat, label, lengths, basename, point_entity_ids, map_l1, map_l2, map_l3, map_l4, image, ct, mulpts_indiv_segs_tensor):
        if isinstance(coord, torch.Tensor):
            coord = coord.cpu().numpy()
        if isinstance(feat, torch.Tensor):
            feat = feat.cpu().numpy()
        if isinstance(point_entity_ids, torch.Tensor):
            point_entity_ids = point_entity_ids.cpu().numpy()
         
        if self.data_norm == 'mean':
            coord -= np.mean(coord, 0)
        elif self.data_norm == 'min':
            coord -= np.min(coord, 0)
        return (torch.FloatTensor(coord), torch.FloatTensor(feat), torch.LongTensor(label), 
                torch.FloatTensor(lengths), basename, torch.LongTensor(point_entity_ids),
                map_l1, map_l2, map_l3, map_l4 , image, ct, mulpts_indiv_segs_tensor)

    def _process_img_spts(self, mulpts_indiv_segs_list):

        processed_list = []
        for sample_list in mulpts_indiv_segs_list:
             
             
            sample_tensor = torch.stack(sample_list, dim=1)  
            processed_list.append(sample_tensor)
        return processed_list  

     
    def collate_fn(self, batch):

        coord, feat, label, lengths, _, point_ids, maps_l1, maps_l2, maps_l3, maps_l4, images, cts, mulpts_indiv_segs_tensor = list(zip(*batch))

        images = torch.stack(images)
        ct = torch.cat(cts)
        img_spts = self._process_img_spts(mulpts_indiv_segs_tensor)
 
        global_maps_l1, global_maps_l2, global_maps_l3, global_maps_l4 = [], [], [], []
        
        curr_group_offset_l1 = 0
        curr_group_offset_l2 = 0
        curr_group_offset_l3 = 0
        curr_group_offset_l4 = 0

        unique_point_ids, entity_offset, entity_offset_count = [], [], 0

        offset, count = [], 0   

        for i in range(len(coord)):
            count += coord[i].shape[0]
            offset.append(count)

            p_id = point_ids[i]
             
            unique_point_ids.append(p_id.long() + entity_offset_count)  
             
            global_maps_l1.append(torch.IntTensor(maps_l1[i]) + curr_group_offset_l1)
            global_maps_l2.append(torch.IntTensor(maps_l2[i]) + curr_group_offset_l2)
            global_maps_l3.append(torch.IntTensor(maps_l3[i]) + curr_group_offset_l3)
            global_maps_l3.append(torch.IntTensor(maps_l4[i]) + curr_group_offset_l4)         
             
            curr_group_offset_l1 += (maps_l1[i].max() + 1)
            curr_group_offset_l2 += (maps_l2[i].max() + 1)
            curr_group_offset_l3 += (maps_l3[i].max() + 1)
            curr_group_offset_l3 += (maps_l4[i].max() + 1)
             
            entity_offset_count += (p_id.max().item() + 1)   
            entity_offset.append(entity_offset_count)
        
        lengths = torch.cat(lengths) if lengths[0] is not None else None

        basename_list = [item[4] for item in batch]

        return (
            torch.cat(coord),                         
            torch.cat(feat),                          
            torch.cat(label),                         
            torch.IntTensor(offset),                  
            lengths,                                  
            basename_list,                           
            torch.cat(unique_point_ids).view(-1),    
            torch.IntTensor(entity_offset),          
            [torch.cat(global_maps_l1), torch.cat(global_maps_l2), torch.cat(global_maps_l3), torch.cat(global_maps_l4)],   
            images,    
            ct,        
            img_spts,  
        )




if __name__ == "__main__":
    
    def parse_args():

        parser = argparse.ArgumentParser("svgnet")
        parser.add_argument("--config", type=str, default='configs/svg/svg_pointT.yaml', help="path to config file")   
        parser.add_argument("--dist", default=False, action="store_true", help="run with distributed parallel")   
        parser.add_argument("--sync_bn", action="store_true", help="run with sync_bn")   
        parser.add_argument("--resume", type=str, help="path to resume from")   
        parser.add_argument("--work_dir", type=str, help="working directory")   
        parser.add_argument("--skip_validate", action="store_true", help="skip validation")
        parser.add_argument("--local_rank", type=int, default=0)   
        parser.add_argument("--seed", type=int, default=2000)
        parser.add_argument("--exp_name", type=str, default="default")   

        parser.add_argument("--data_root", type=str, default="")
        parser.add_argument("--filter", type=int, default=0)
        parser.add_argument("--do_aug", type=bool, default=True)
        parser.add_argument("--fp16", type=bool, default=True)

        parser.add_argument("--lr_rate", type=int,
                            default=1e-4,
                            help='learning rate')   
        parser.add_argument("--input_dim", type=int, default=9,
                            help='the nodes features dimension to input to model')   
        parser.add_argument("--cls_num", type=int, default=35,
                            help='the number of the label to model, except bg')   
        parser.add_argument("--Q_num", type=int, default=500,
                            help='the number of the queries')   
        parser.add_argument("--enc_stride", type=int, default=[1, 4, 4, 4, 4],
                            help='the stride of the PT sampling')   
        parser.add_argument("--enc_nsample", type=int, default=[16, 16, 16, 16, 16],
                            help='the number of the PT sampling points')   
        parser.add_argument("--eos_coef", type=int, default=0.1,
                            help='the weight of the BG')   
        parser.add_argument("--epochs", type=int, default=50,
                            help='epoch for training')   
        parser.add_argument("--max_prim", type=int,
                            default=20000,
                            help='maximum primitive number for each batch')   
        parser.add_argument("--log_step", type=int,
                            default=100,
                            help='steps for logging')   
        parser.add_argument("--load_ckpt", type=str,
                            default='',
                            help='load checkpoint')   
        parser.add_argument("--matcher_cost_class", type=float,
                            default=0.5,
                            help='cost_class weight')  
        parser.add_argument("--matcher_cost_mask", type=float,
                            default=1,
                            help='cost_mask weight')  
        parser.add_argument("--matcher_cost_dice", type=float,
                            default=1,
                            help='cost_dice weight')  
        parser.add_argument("--train_batch_size", type=int,
                            default=4,
                            help='batch size') 
        
         
        parser.add_argument('--embed_backbone', type=str,
                            default="hrnet48")                                                        
        parser.add_argument('--backbone_cfg',
                            type=str,
                            default=None,  
                            help='Backbone configure file name')                                      
        parser.add_argument('--do_pretrained_backbone', type=str,
                            default=False)                                                            
        parser.add_argument('--pretrained_model', type=str,
                            default=None)     
        args = parser.parse_args()
        return args

    args = parse_args()

    args.work_dir = "/home/16t/linh/linhreproduce_main/main/datasets/test/2026floorplantol2"
    args.data_root = "/home/16t/linh/linhreproduce_main/main/datasets/Floor_Plan_CAD/202606_1500s"

    args.skip_validate = False
    args.local_rank = 0
    args.seed = 2000  

    args.filter = 8
    args.do_aug = False
    args.fp16 = True
    args.max_prim = 5000
    args.train_batch_size = 1


    device = torch.device('cuda:{}'.format(args.local_rank))
     
     
    cfg = update_config(config, args)   

    rank = 0

    set_seed(args.seed + rank)   
    cfg.defrost()
    cfg.dataloader.train.batch_size = 2    
    cfg.data.train.repeat = 2              
    cfg.dataloader.train.num_workers = 3   
    cfg.matcher.Disentangled_Matching = True
    cfg.matcher.DM_weight = 8.0
    cfg.optimizer.decoder_multiplier = 1
    cfg.model.stride = (2,2,4,4)
    cfg.model.patch_size = 256
    cfg.model.num_decoders = 6
    cfg.step_epoch = 43
    cfg.model.aggregation_mode = 'add'
    cfg.model.pool_mode = "defualt"    
    cfg.model.share_kernels = False
    cfg.model.spatial_cfg_enable = None
    cfg.model.spatial_cfg_radius = None
    cfg.model.isolation_mode = "half"
    cfg.model.isolation_attn = True
    cfg.step_dist = 0.001
    cfg.matcher.cost_class = 0.5
    cfg.matcher.cost_mask = 1
    cfg.matcher.cost_dice = 1
    cfg.matcher.alpha = 0.25
    cfg.matcher.gamma = 2
    cfg.model.en_CFE = True                                                                     
    cfg.model.cfg_gnn_en = False                                                                  
    cfg.model.cfg_gnn_dim = 8                                                           
    cfg.work_dir = args.work_dir
    imgsize=700
    os.makedirs(cfg.work_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())   
    log_file = osp.join(cfg.work_dir, f"{timestamp}.log")    
    logger = get_root_logger(log_file=log_file)              

    dataset = HiConSpotDataset_svg(split='train', cfg=cfg, aug=cfg.data.test.aug, max_prim=18000, root=cfg.data.test.data_root,
                                   logger=logger, do_aug=False, repeat=1, do_filter=False, is_mulpts_sampling=True, size=700, max_points=49, debug=True,
                                   do_process=True)
    
  

