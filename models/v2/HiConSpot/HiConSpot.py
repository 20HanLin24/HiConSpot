"""
HiConSpot Model

Author: Han Lin (hcl0jf26@gmail.com)
Please cite our work if the code is helpful to you.
"""


import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .utils.utils import cuda_cast
from .modules.ptv3_img_wrapper import dxfBackbone

from .modules.decoder import Decoder

import numpy as np

def batch_to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, (list, tuple)):
        return type(batch)(batch_to_device(item, device) for item in batch)
    elif isinstance(batch, dict):
        return {k: batch_to_device(v, device) for k, v in batch.items()}
    else:
        return batch




class HiConSpot(nn.Module):
    def __init__(self, cfg, criterion=None, FF_mode='add', grid_size=None, alpha=0.5, mask_threshold=0.5, overlap_threshold=0.8, inst_pred_thresh=0.5,):
        super().__init__()

        self.criterion = criterion
        self.mask_threshold = mask_threshold
        self.overlap_threshold = overlap_threshold
        self.inst_pred_thresh = inst_pred_thresh

        # NOTE backbone
        self.backbone = dxfBackbone(cfg, FF_mode=FF_mode, cls_mode=False, grid_size=grid_size)  

        self.decoder = Decoder(cfg, planes=self.backbone.backbone.planes, in_channels=64, alpha=alpha, 
                                            isolation_attn=True, isolation_mode="half")

        self.num_classes = cfg.semantic_classes  
        self.bg_id = cfg.semantic_classes
        self.test_object_score = 0.1

    def train(self, mode=True):
        super().train(mode)

    def forward(self, batch, return_loss=True):
        coords, feats, semantic_labels, offsets, lengths, basename_list, unique_point_ids, entity_offset, maps_list, imgs, cts, img_spts = batch
        return self._forward(
            coords, feats, offsets, semantic_labels, lengths, 
            return_loss=return_loss, 
            basename_list=basename_list, 
            unique_point_ids=unique_point_ids, 
            entity_offset=entity_offset,
            maps_list=maps_list,
            imgs=imgs,
            cts=cts,
            img_spts=img_spts,
        )
    
    def prepare_targetsV2(self, semantic_labels):
        
        instance_ids = semantic_labels[:, 1].cpu().numpy()  
        semantic_ids = semantic_labels[:, 0].cpu().numpy()  

        keys = []  
        for sem_id, ins_id in zip(semantic_ids,
                                  instance_ids):
            if (sem_id, ins_id) not in keys:
                keys.append((sem_id, ins_id))

        unique_sems = list(set(semantic_ids))
        svg_len = semantic_ids.shape[0]  

        # print("keys", keys)

        cls_targets, mask_targets = [], []

        for (sem_id, ins_id) in keys:
            if sem_id == self.num_classes and ins_id == -1:
                continue  

            tensor_mask = torch.zeros(svg_len)  
            
            ind1 = np.where(semantic_ids == sem_id)[0]
            ind2 = np.where(instance_ids == ins_id)[0]

            ind = list(set(ind1).intersection(ind2))
            tensor_mask[ind] = 1  
            cls_targets.append(sem_id)  


            mask_targets.append(tensor_mask.unsqueeze(1))  

        cls_targets = torch.tensor(cls_targets) if cls_targets else torch.tensor([self.num_classes])

        mask_targets = torch.cat(mask_targets, dim=1) if mask_targets else torch.zeros(svg_len, 1)


        inst_cls_targets, inst_mask_targets = [], []
   
        for (sem_id, ins_id) in keys:
            if sem_id == self.num_classes or ins_id == -1:
                continue  

            tensor_mask = torch.zeros(svg_len)  

            ind1 = np.where(semantic_ids == sem_id)[0]
            ind2 = np.where(instance_ids == ins_id)[0]

            ind = list(set(ind1).intersection(ind2))
            tensor_mask[ind] = 1  
            inst_cls_targets.append(sem_id)  

            inst_mask_targets.append(tensor_mask.unsqueeze(1))  
        cls_targets_inst = torch.tensor(inst_cls_targets) if inst_cls_targets else torch.tensor([self.num_classes])
        mask_targets_inst = torch.cat(inst_mask_targets, dim=1) if inst_mask_targets else torch.zeros(svg_len, 1)

        cls_targets_sem, mask_targets_sem = [], []
        for sem_id in range(self.num_classes):
            tensor_mask = torch.zeros(svg_len)  
            if sem_id == self.num_classes:
                continue
            if sem_id in unique_sems:
                ind = np.where(semantic_ids == sem_id)[0]
                tensor_mask[ind] = 1
            cls_targets_sem.append(sem_id)
            mask_targets_sem.append(tensor_mask.unsqueeze(1))  

        cls_targets_sem = torch.tensor(cls_targets_sem) if cls_targets_sem else torch.tensor([self.num_classes])

        mask_targets_sem = torch.cat(mask_targets_sem, dim=1) if mask_targets_sem else torch.zeros(svg_len, 1)

        total_cls_targets = torch.cat([cls_targets_inst, cls_targets_sem], dim=0)
        total_mask_targets = torch.cat([mask_targets_inst, mask_targets_sem], dim=1)

        return [{
            "labels": cls_targets.to(semantic_labels.device),  
            "masks": mask_targets.to(semantic_labels.device),  

            "sem_labels": cls_targets_sem.to(semantic_labels.device),  
            "sem_masks": mask_targets_sem.to(semantic_labels.device),  

            "inst_labels": cls_targets_inst.to(semantic_labels.device),  
            "inst_masks": mask_targets_inst.to(semantic_labels.device),  

            "total_labels": total_cls_targets.to(semantic_labels.device),  
            "total_masks": total_mask_targets.to(semantic_labels.device),  
        }]

    @cuda_cast  
    def _forward(self, coords, feats, offsets, semantic_labels, lengths, return_loss=True, basename_list='', 
                 unique_point_ids=None, entity_offset=None, maps_list=None, imgs=None, cts=None, img_spts=None):

        stage_list = {'inputs': {'p_out': coords, "f_out": feats, "offset": offsets, },
                      "semantic_labels": semantic_labels[:, 0], 
                      "instances_labels": semantic_labels[:, 1],  
                      "unique_point_ids": unique_point_ids,  
                      "entity_offset": entity_offset,  
                      "maps_list": maps_list,        
                      "img": imgs, 
                      "ct": cts,
                      "img_spts": img_spts, 
                }


        targets = self.prepare_targetsV2(semantic_labels)

        stage_list.update({"tgt": targets})  

        stage_list = self.backbone(stage_list)  

        outputs = self.decoder(stage_list)      


        model_outputs = {}

        if not self.training:
            sem_pred, sem_pred_with_bg = (
                self.semantic_inference(outputs["pred_logits"], outputs["sem_pred_masks"], self.mask_threshold))  
            model_outputs.update(dict(sem_pred=sem_pred,),)  
            model_outputs.update(dict(sem_pred_with_bg=sem_pred_with_bg, ), )  
            model_outputs.update(dict(semantic_labels=semantic_labels[:, 0],),)  
            instances = self.instance_inference(outputs, overlap_threshold=self.overlap_threshold , pred_thresh=self.inst_pred_thresh )  
            model_outputs.update(dict(instances=instances,),)  
            model_outputs.update(dict(targets=targets[0],),)  
            model_outputs.update(dict(lengths=lengths,),)  
            model_outputs.update(dict(basename_list=basename_list, ), )


        if not return_loss:  
            return model_outputs


        losses = self.criterion(outputs, targets)
        loss_value, loss_dicts = self.parse_losses(losses)


        return model_outputs, loss_value, loss_dicts  
    def instance_inference(self, outputs, overlap_threshold=0.8, pred_thresh=0.5):

        all_logits = outputs["pred_logits"].squeeze(0)        
        all_masks = outputs["pred_masks"]                     
        q_counts = outputs["q_counts"]                        

        results = []

        curr_q_start = 0

        for b in range(len(q_counts)):
            n_q = q_counts[b]

            sample_logits = all_logits[curr_q_start : curr_q_start + n_q] 

            sample_masks = all_masks[b]
            if sample_masks.dim() == 3: 
                sample_masks = sample_masks.squeeze(0)
            prob = F.softmax(sample_logits, dim=-1)
            scores, labels = prob.max(-1)  
            mask_pred = sample_masks.sigmoid()

            keep = (labels != self.num_classes) & (scores >= self.test_object_score)
            if not keep.any():
                curr_q_start += n_q
                print(f"Error! queries is skiped.labels != self.num_classes:{labels != self.num_classes},sample_logits: {sample_logits}, scores:{scores}\n")
                continue

            cur_scores = scores[keep]  
            cur_classes = labels[keep] 
            cur_masks = mask_pred[keep]  
            cur_prob_masks = cur_scores.unsqueeze(-1) * cur_masks 
            
            try:

                cur_mask_ids = cur_prob_masks.argmax(0) 
            except Exception as e:

                curr_q_start += n_q
                continue

            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                pred_score = cur_scores[k].item()
                
                mask = (cur_mask_ids == k) & (cur_masks[k] >= pred_thresh)
                
                mask_area = mask.sum().item()
                original_area = (cur_masks[k] >= pred_thresh).sum().item()

                if mask_area > 0 and original_area > 0:

                    if mask_area / original_area < overlap_threshold:
                        continue
                    
                    results.append({
                        "masks": mask.cpu().numpy(),
                        "labels": pred_class,
                        "scores": pred_score,
                        # "sample_idx": b 
                    })
                else:
                    pass

            curr_q_start += n_q

        print("num_instances:", len(results))
        return results
    
    def semantic_inference(self, mask_cls, mask_pred, mask_threshold=0.5):

        sem_mask_pred = mask_pred.sigmoid()                                             # [1, sem_Q, N]
        sem_pred = sem_mask_pred[0].transpose(0, 1)  
        sem_pred_with_bg = sem_pred.clone()
        max_prob, _ = sem_pred_with_bg.max(dim=-1)
        void_mask = max_prob < mask_threshold               
        bg_score = torch.zeros((sem_pred.shape[0], 1), device=sem_pred.device)
        bg_score[void_mask] = 0.5
        sem_pred_with_bg = torch.cat([sem_pred_with_bg, bg_score], dim=-1)

        return sem_pred, sem_pred_with_bg 

    def parse_losses(self, losses):
        
        loss = sum(v for v in losses.values())
        losses["loss"] = loss
        for loss_name, loss_value in losses.items():
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            losses[loss_name] = loss_value.item()  
        return loss, losses

    