# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by HanLin from https://github.com/facebookresearch/detr/blob/master/models/detr.py
 
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.cuda.amp import autocast
import numpy as np
from itertools import chain

def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):

    inputs = inputs.sigmoid()   
     
    inputs = inputs.flatten(1)   
     
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)   
     
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]   
    loss = 1 - (numerator + 1) / (denominator + 1)   
    return loss   


batch_dice_loss_jit = torch.jit.script(
    batch_dice_loss
)   


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):

    hw = inputs.shape[1]   
     
    pos = F.binary_cross_entropy_with_logits(inputs, torch.ones_like(inputs), reduction="none")   
    neg = F.binary_cross_entropy_with_logits(inputs, torch.zeros_like(inputs), reduction="none")   
     
    loss = torch.einsum("nc,mc->nm", pos, targets) + \
           torch.einsum("nc,mc->nm", neg, (1 - targets))   

    return loss / hw   


 
 
batch_sigmoid_ce_loss_jit = torch.jit.script(
    batch_sigmoid_ce_loss
)   


class HungarianMatcher_HiConSpot(nn.Module):
    def __init__(self, num_classes, cost_class: float = 1, cost_mask: float = 1, cost_dice: float = 1, num_points: int = 0, 
                 Disentangled_Matching=True, DM_weight=None, alpha=0.25, gamma=2, use_one2one_match=False):

        super().__init__()
        self.cost_class = cost_class   
        self.cost_mask = cost_mask   
        self.cost_dice = cost_dice   
        self.num_classes = num_classes
        self.Disentangled_Matching = Disentangled_Matching   
        self.DM_weight = DM_weight                           
        self.topk = 1
        self.alpha = alpha
        self.gamma = gamma
        self.use_one2one_match = use_one2one_match

        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"

        self.num_points = num_points   
    
    @torch.cuda.amp.autocast(enabled=False)
    def cal_C2res(self, raw_logits, out_mask, num_queries, tgt_ids, tgt_mask, is_valid_match=None, alpha=0.25, gamma=4):

        raw_logits = raw_logits.float()

        if torch.any(torch.isnan(raw_logits) | torch.isinf(raw_logits)) | torch.any(torch.isneginf(raw_logits)):
            print("raw_logits, nan/inf/neginf")
         
        out_prob = raw_logits.softmax(-1)   

        out_mask = out_mask.float()   
         
        tgt_mask = tgt_mask.float()
         
        cost_mask = batch_sigmoid_ce_loss_jit(out_mask, tgt_mask)
         
        cost_dice = batch_dice_loss_jit(out_mask, tgt_mask)

        if (~torch.isfinite(cost_dice)).any():
            cost_dice = torch.nan_to_num(cost_dice, nan=1.0, posinf=1.0, neginf=1.0)

        iou = (1 - cost_dice) / (1 + cost_dice + 1e-8).clamp(min=0, max=1.0)

        _out_prob = torch.nan_to_num(out_prob[:, tgt_ids], nan=0.0).clamp(min=0.0, max=1.0) * iou.clamp(min=0, max=1.0)

        neg_cost_class = (1 - alpha) * (_out_prob ** gamma) * (-(1 - _out_prob + 1e-5).log())
         
        pos_cost_class = alpha * ((1 - _out_prob) ** gamma) * (-(_out_prob + 1e-5).log())

        cost_class = pos_cost_class - neg_cost_class   
        try:
            C = self.cost_mask * cost_mask + self.cost_dice * cost_dice + self.cost_class * cost_class   
            C_cloned = C.clone()

            if is_valid_match is not None:
                C[~is_valid_match] = 1e8

            C = C.reshape(num_queries, -1).cpu()   
             
            values = torch.topk(C, self.topk + 1, dim=0, sorted=True, largest=False).values[-1:, :]
             
            ids = torch.argwhere(C < values)
             
            if ids.numel() == 0 or self.use_one2one_match:
                res = linear_sum_assignment(C_cloned.reshape(num_queries, -1).cpu().numpy())   
            else:
                res = (ids[:, 0], ids[:, 1])
        except:
            _out_prob = out_prob[:, tgt_ids].clamp(min=0.0, max=1.0)
            neg_cost_class = (1 - alpha) * (_out_prob ** gamma) * (-(1 - _out_prob + 1e-5).log())
            pos_cost_class = alpha * ((1 - _out_prob) ** gamma) * (-(_out_prob + 1e-5).log())
            cost_class = pos_cost_class - neg_cost_class
            try:
                C = self.cost_mask * cost_mask + self.cost_class * cost_class + self.cost_dice * cost_dice
                C_cloned = C.clone()
                if is_valid_match is not None:
                    C[~is_valid_match] = 1e8

                C = C.reshape(num_queries, -1).cpu()   
                 
                values = torch.topk(C, self.topk + 1, dim=0, sorted=True, largest=False).values[-1:, :]
                 
                ids = torch.argwhere(C < values)
                 
                if ids.numel() == 0:
                     
                    res = linear_sum_assignment(C_cloned.reshape(num_queries, -1).cpu().numpy())   
                else:
                     
                    res = (ids[:, 0], ids[:, 1])
            except:
                 
                cost_mask = torch.nan_to_num(cost_mask, nan=1.0, posinf=1.0, neginf=1.0)
                C = self.cost_mask * cost_mask + self.cost_class * cost_class + self.cost_dice * cost_dice
                C_cloned = C.clone()
                if is_valid_match is not None:
                    C[~is_valid_match] = 1e8
                C = C.reshape(num_queries, -1).cpu()
                 
                values = torch.topk(C, self.topk + 1, dim=0, sorted=True, largest=False).values[-1:, :]
                 
                ids = torch.argwhere(C < values)
                 
                if ids.numel() == 0:
                     
                    res = linear_sum_assignment(C_cloned.reshape(num_queries, -1).cpu().numpy())   
                else:
                     
                    res = (ids[:, 0], ids[:, 1])
        return res
    
    @torch.no_grad()   
    def memory_efficient_forward_oneformer(self, outputs, targets, id=None,):

        if id is not None:
            all_logits = outputs['aux_outputs'][id]["ins_pred_logits"].squeeze(0)     
            all_masks = outputs['aux_outputs'][id]["pred_masks"]           
        else:
            all_logits = outputs["ins_pred_logits"].squeeze(0)     
            all_masks = outputs["pred_masks"]                      
        q_counts = outputs["q_counts"]                     
        mem_counts = outputs["mem_counts"]                 

        query_init_indices = outputs["queries_entity_ids"]       

        entity_offset = outputs["stage_list"]["entity_offset"]   

        assert entity_offset[-1] == targets[0]["masks"].shape[0]

        tgt_ids = targets[0]["labels"]

        tgt_mask = targets[0]["masks"].float().transpose(0, 1)

        indices = []
        curr_q_start  = 0   
        curr_n_start  = 0   

        for b in range(len(q_counts)):

            n_inst_q = q_counts[b]    
            n_mem = mem_counts[b]     

            sample_logits = all_logits[curr_q_start : curr_q_start + n_inst_q]   
            sample_masks = all_masks[b].squeeze(0)
            cur_tgt_mask = tgt_mask[:, curr_n_start : curr_n_start + n_mem]      

            is_valid_match = None
   
            q_init = query_init_indices[b]

            if self.Disentangled_Matching:
                is_valid_match = tgt_mask[:, q_init].transpose(0, 1) > 0.5 

            if len(tgt_ids) == 0:
                indices.append((torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)))
                curr_q_start += n_inst_q
                curr_n_start += n_mem
                continue

            alpha = self.alpha
            gamma = self.gamma

            res = self.cal_C2res(sample_logits, sample_masks, n_inst_q, tgt_ids, cur_tgt_mask, is_valid_match=is_valid_match, alpha=alpha, gamma=gamma)
 
            row_idx, col_idx = res
            
            indices.append((
                torch.as_tensor(row_idx, dtype=torch.int64),
                torch.as_tensor(col_idx, dtype=torch.int64)
            ))

            curr_q_start += n_inst_q  
            curr_n_start += n_mem     

        return indices   

    @torch.no_grad()
    def forward(self, outputs, targets, id=None):
        """Performs the matching
        """
        return self.memory_efficient_forward_oneformer(outputs, targets, id)

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


 

 

 
 

 
 

 
 
 
 
 
 

 
 
 
 
 

 
 
 

 
 