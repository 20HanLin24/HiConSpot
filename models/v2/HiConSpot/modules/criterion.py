 
 # Copyright (c) Facebook, Inc. and its affiliates.
# Modified by HanLin from https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
MaskFormer criterion.
"""

import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist
from ..utils.heads import ContrastHead


 
def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def dice_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float,):

    inputs = inputs.sigmoid()   
     
    numerator = 2 * (inputs * targets).sum(-1)   
    denominator = inputs.sum(-1) + targets.sum(-1)   
    loss = 1 - (numerator + 1) / (denominator + 1)   
    return loss.sum() / num_masks   


dice_loss_jit = torch.jit.script(
    dice_loss
)   


def dice_loss_mean(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float,):
 
    inputs = inputs.sigmoid()   
     
    numerator = 2 * (inputs * targets).sum(-1)   
    denominator = inputs.sum(-1) + targets.sum(-1)   
    loss = 1 - (numerator + 1e-5) / (denominator + 1e-5)   
    return loss.mean() / num_masks   


dice_loss_mean_jit = torch.jit.script(
    dice_loss_mean
)   


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):

    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)   


def sigmoid_bce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):

    loss = F.binary_cross_entropy_with_logits(inputs, targets)

    return loss / num_masks


sigmoid_bce_loss_jit = torch.jit.script(
    sigmoid_bce_loss
)   


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):

    prob = inputs.sigmoid()   
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")   
     
    p_t = prob * targets + (1 - prob) * (1 - targets)
     
     
     
    loss = ce_loss * ((1 - p_t) ** gamma)   

     
    if alpha >= 0:
         
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss   
     
    return loss.mean(1).sum() / num_boxes   


def calculate_uncertainty(logits):

    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class SetCriterion_HiConSpot(nn.Module):
  
    def __init__(self, matcher, weight_dict, cfg, dn_no_lb=False, pure_mode=False, use_mean_batch_loss=False):
        super().__init__()
        self.num_classes = cfg.num_classes   
        self.matcher = matcher   
        self.weight_dict = weight_dict   
        self.class_weights = cfg.class_weights   
        self.eos_coef = cfg.eos_coef   
        self.losses = {"labels", "masks", 'sem_masks'}   
        self.dn_no_lb = dn_no_lb   
         
        self.bg_id = self.num_classes                     
        empty_weight = torch.ones(self.num_classes + 1)   
        empty_weight[self.bg_id] = self.eos_coef   
        if self.class_weights != -1:
            assert len(self.class_weights) == self.num_classes, "CLASS WEIGHTS DO NOT MATCH"
            tmp_mask = torch.ones_like(empty_weight, dtype=torch.bool)   
            tmp_mask[self.bg_id] = False                                 
            empty_weight[tmp_mask] = torch.tensor(self.class_weights)
            del tmp_mask

        self.use_mean_batch_loss = use_mean_batch_loss
        self.pure_mode = pure_mode

        self.register_buffer("empty_weight", empty_weight)   
         
        self.head = ContrastHead(cfg.contrast)
    
    @torch.cuda.amp.autocast(enabled=False)
    def loss_labels(self, outputs, targets, indices, num_masks, cts=None,):
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float().squeeze(0)   
        device = src_logits.device

        inst_q_counts = outputs["q_counts"]
        assert src_logits.shape[0] == sum(inst_q_counts) + self.num_classes
        q_counts = outputs["q_counts"]                     
        mem_counts = outputs["mem_counts"]                 
        curr_q_start  = 0   
        curr_n_start  = 0   
        loss_ce_batch = 0

        for b in range(len(q_counts)):
            n_inst_q = q_counts[b]    
            n_mem = mem_counts[b]     
            sample_logits = src_logits[curr_q_start : curr_q_start + n_inst_q]   

            src_intq_idx, tgt_intq_idx = indices[b]

            target_classes_o = targets[0]["labels"][tgt_intq_idx]

             
            target_classes = torch.full((sample_logits.shape[0],), self.bg_id, dtype=torch.long, device=device)  
            target_classes[src_intq_idx] = target_classes_o   

             
            curr_q_start += n_inst_q  
            curr_n_start += n_mem     
        
             
            loss_ce = F.cross_entropy(sample_logits, target_classes, 
                                    self.empty_weight.float() if self.empty_weight is not None else None)
            
            loss_ce_batch += loss_ce

        sem_src_logit = src_logits[-self.num_classes:]
        target_classes = torch.full((sem_src_logit.shape[0],), self.bg_id, dtype=torch.long, device=device) 

        raw_sem_labels = outputs["stage_list"]["semantic_labels"]
        present_classes = torch.unique(raw_sem_labels)
         
        for cls_id in present_classes:
            if cls_id != self.bg_id:  
                target_classes[cls_id] = cls_id 
        
        loss_ce = F.cross_entropy(sem_src_logit, target_classes, 
                                  self.empty_weight.float() if self.empty_weight is not None else None)
        
        loss_ce_batch += loss_ce
        
        losses = {"loss_ce": loss_ce_batch * self.weight_dict["loss_ce"]}   
        return losses
    
    @torch.cuda.amp.autocast(enabled=False)
    def loss_masks(self, outputs, targets, indices, num_masks, cts=None,):
         
        assert "pred_masks" in outputs
        
         
        src_masks = outputs["pred_masks"]                                    
        global_tgt_masks = targets[0]["masks"].float().transpose(0, 1)  
        device = src_masks[0].device

        q_counts = outputs["q_counts"]                     
        mem_counts = outputs["mem_counts"]                 
        curr_q_start  = 0   
        curr_n_start  = 0   

        if not self.pure_mode:
            batch_loss_mask = 0
            batch_loss_dice = 0
        else:
            batch_loss_mask = []
            batch_loss_dice = []

        for b in range(len(src_masks)):
            n_inst_q = q_counts[b]    
            n_mem = mem_counts[b]     

            src_mask = src_masks[b].float().squeeze(0)
            cur_tgt_mask = global_tgt_masks[:, curr_n_start : curr_n_start + n_mem]
             
             
            src_intq_idx, tgt_intq_idx = indices[b]

             
             
             
            matched_src_inst = src_mask[src_intq_idx]         
            matched_tgt_inst = cur_tgt_mask[tgt_intq_idx]     

            if not self.pure_mode:
                loss_mask = sigmoid_ce_loss_jit(matched_src_inst, matched_tgt_inst, 1)
                loss_dice = dice_loss_jit(matched_src_inst, matched_tgt_inst, 1)
                batch_loss_mask += loss_mask
                batch_loss_dice += loss_dice
            else:
                loss_mask = sigmoid_bce_loss_jit(matched_src_inst, matched_tgt_inst, 1)
                loss_dice = dice_loss_mean_jit(matched_src_inst, matched_tgt_inst, 1)
                batch_loss_mask.append(loss_mask)
                batch_loss_dice.append(loss_dice)

             
            curr_q_start += n_inst_q  
            curr_n_start += n_mem     
        if not self.pure_mode:
            batch_loss_mask = batch_loss_mask / num_masks
            batch_loss_dice = batch_loss_dice / num_masks
        else:
            batch_loss_mask = torch.stack(batch_loss_mask).mean() if self.use_mean_batch_loss else torch.stack(batch_loss_mask).sum()
            batch_loss_dice = torch.stack(batch_loss_dice).mean() if self.use_mean_batch_loss else torch.stack(batch_loss_dice).sum()
         
         
         
        losses = {
            "loss_mask": batch_loss_mask*self.weight_dict["loss_mask"],   
            "loss_dice": batch_loss_dice*self.weight_dict["loss_dice"],   
        }
        del matched_src_inst
        del matched_tgt_inst
        return losses
    
    @torch.cuda.amp.autocast(enabled=False)
    def loss_sem_masks(self, outputs, targets, indices, num_masks, cts=None,):
         
        assert "pred_masks" in outputs
        
         
        src_masks = outputs["sem_pred_masks"].float().squeeze(0)    
        raw_sem_labels = outputs["stage_list"]["semantic_labels"]   
        tgt_sem = targets[0]["sem_masks"].float().transpose(0, 1)   
        device = src_masks.device

        sem_logits_map = src_masks.transpose(0, 1)   
         
         
         
        bg_logit = torch.zeros((sem_logits_map.shape[0], 1), device=device)
        final_logits = torch.cat([sem_logits_map, bg_logit], dim=1)  

        ce_loss_jit = torch.nn.CrossEntropyLoss(weight=self.empty_weight.float() if self.empty_weight is not None else None).to(device)
        loss_ce = ce_loss_jit(final_logits, raw_sem_labels)

         
         
        loss_dice = dice_loss_jit(src_masks, tgt_sem, num_masks=1)

         
        losses = {
            "loss_ce_sem": loss_ce*self.weight_dict["loss_dice"],  
             
            "loss_dice_sem": loss_dice*self.weight_dict["loss_dice"],   
        }
        del src_masks
        del tgt_sem
        return losses

    def get_loss(self, loss, outputs, targets, indices, num_masks, cts=None, ):

        loss_map = {
            'labels': self.loss_labels,   
            'masks': self.loss_masks,
            'sem_masks': self.loss_sem_masks,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks, cts)

    def forward(self, outputs, targets):
         
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs" and
                               k != "dn_out"}   
         
         
        indices = self.matcher(outputs_without_aux, targets)
  
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor([num_masks], dtype=torch.float, device=next(iter(outputs.values())).device)   

        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)

         
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()   

        losses = {}

        for loss in self.losses:
 
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        if outputs.get("objectness_scores") is not None:
            objectness_scores = outputs["objectness_scores"]

            raw_sem_labels = outputs["stage_list"]["semantic_labels"] 

            target_obj = (raw_sem_labels != self.bg_id).float().unsqueeze(-1)

            loss_confidence = F.binary_cross_entropy_with_logits(objectness_scores, target_obj)
            losses.update({"loss_pt_confidence": loss_confidence})

         
        if "aux_outputs" in outputs:
             
            for i in range(len(outputs['aux_outputs'])):
                 
                indices = self.matcher(outputs, targets, id=i)
                for loss in self.losses:
                    aux_outputs = outputs["aux_outputs"][i]
                    aux_outputs["q_counts"] = outputs["q_counts"]
                    aux_outputs["mem_counts"] = outputs["mem_counts"]
                    aux_outputs["stage_list"] = outputs["stage_list"]
                    aux_outputs["queries_entity_ids"] = outputs["queries_entity_ids"]
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}   
                    losses.update(l_dict)

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


