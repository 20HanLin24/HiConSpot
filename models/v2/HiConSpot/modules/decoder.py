import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import autocast
from .position_embedding import PositionEmbeddingCoordsSine


class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):

        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None):

        q = k = self.with_pos_embed(tgt, query_pos)

        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None):

        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask, tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):

        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        """与相对位置编码相加"""
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory, memory_mask=None, memory_key_padding_mask=None, pos=None, query_pos=None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt, memory, memory_mask=None, memory_key_padding_mask=None, pos=None, query_pos=None):

        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory, memory_mask=None, memory_key_padding_mask=None, pos=None, query_pos=None):

        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):

        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class Decoder(nn.Module):

    def __init__(self, cfg, planes, in_channels=64, alpha=0.25, isolation_attn=True, isolation_mode="all"):
        """
        {}spatial_cfg: dict, 包含 {'enable': bool, 'radius': float}
        """
        super().__init__()
        self.hidden_dim = cfg.hidden_dim
        self.mask_dim = cfg.hidden_dim
        self.dim_feedforward = cfg.dim_feedforward
        self.planes = planes
        self.num_classes = cfg.semantic_classes
        self.num_layers = cfg.num_decoders
        self.nhead = cfg.num_heads
        self.dropout = cfg.dropout
        self.pre_norm = False

        self.shared_decoder = cfg.shared_decoder

        self.alpha = alpha

        self.in_channels = in_channels

        self.isolation_attn = isolation_attn
        self.isolation_mode = isolation_mode

        self.input_proj = nn.Linear(in_channels, self.hidden_dim)

        self.mask_feat_proj1 = nn.Sequential(
            nn.Linear(in_channels, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.inst_temperature = nn.Parameter(torch.zeros(1))
        self.mask_feat_proj = nn.Sequential(
            nn.Linear(in_channels, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.pos_enc = PositionEmbeddingCoordsSine(pos_type="fourier",
                                                   d_pos=self.hidden_dim,
                                                   gauss_scale=cfg.gauss_scale,
                                                   normalize=cfg.normalize_pos_enc)

        self.semantic_query = nn.Embedding(self.num_classes, self.hidden_dim)
        self.sem_query_pos = nn.Embedding(self.num_classes, self.hidden_dim)

        self.instance_query_proj = nn.Sequential(
            nn.Linear(in_channels, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        self.instance_query_pos_proj = nn.Sequential(
            nn.Linear(3, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        self.transformerlayers = nn.ModuleList()
        for _ in range(self.num_layers):
            tmp_transformerlayer = nn.ModuleDict()

            tmp_transformerlayer.update(
                {
                    "self_attn": VarlenSelfAttentionLayerV2(
                        d_model=self.hidden_dim,
                        nhead=self.nhead,
                        dropout=self.dropout,
                        normalize_before=self.pre_norm,
                        activation="gelu",
                    )
                }
            )

            tmp_transformerlayer.update(
                {
                    "cross_attn": VarlenCrossAttentionLayerV2(
                        d_model=self.hidden_dim,
                        nhead=self.nhead,
                        dropout=self.dropout,
                        normalize_before=self.pre_norm,
                        activation="gelu",
                    )
                }
            )

            tmp_transformerlayer.update(
                {
                    "ffn": FFNLayer(
                        d_model=self.hidden_dim,
                        dim_feedforward=self.dim_feedforward,
                        dropout=self.dropout,
                        normalize_before=self.pre_norm,
                        activation="gelu",
                    )
                }
            )
            self.transformerlayers.append(tmp_transformerlayer)

        self.class_embed = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.num_classes + 1),
            nn.Linear(self.num_classes + 1, (self.num_classes + 1) * 4),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear((self.num_classes + 1) * 4, self.num_classes + 1),
        )

        self.mask_sem_kernel_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        )

        self.mask_inst_kernel_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        )

    def get_pos_encs(self, coords):
        """
        input:coords[p5(m4, 3), p4(m3, 3), p3(m2, 3), p2(m1, 3)]，中心点坐标矩阵组成的列表
        N为输入批次的总节点数，当B=1、不增强时，N=max_num
        mi==new_offsetsi[-1]==N/stride**(i)，c==planes[i]=[64,128,256,512,1024]
        """
        pos_encodings_pcd = []

        for i in range(len(coords)):
            coords_batch = coords[i]
            scene_min = coords_batch.min(dim=0)[0][None, ...]
            scene_max = coords_batch.max(dim=0)[0][None, ...]
            with autocast(enabled=False):
                tmp = self.pos_enc(coords_batch[None, ...].float(),
                                   input_range=[scene_min, scene_max])

                pos_encodings_pcd.append(tmp.permute(2, 0, 1))
        return pos_encodings_pcd

    def get_refined_masked_attn_mask_from_list(self, list_pred_masks, q_offsets, e_offsets, threshold=0.5,
                                               spatial_cfg=None, primitive_coords=None, sem_pred_mask=None, ):
        """生成组合掩码：动态掩码 (Masked Attention)"""
        device = list_pred_masks[0].device
        total_q = q_offsets[-1]
        total_n = e_offsets[-1]

        list_inst_attn_masks = []

        for b in range(len(list_pred_masks)):

            qs, qe = q_offsets[b], q_offsets[b + 1]
            es, ee = e_offsets[b], e_offsets[b + 1]

            sample_logits = list_pred_masks[b]
            sample_prob = sample_logits.sigmoid()

            sample_mask = (sample_prob < threshold)

            if spatial_cfg is not None and spatial_cfg.get('enable', False):
                radius = spatial_cfg.get('radius', 0.1)
                sample_coords = primitive_coords[es:ee]

                weights = sample_prob + 1e-8

                centroids = torch.matmul(weights, sample_coords) / weights.sum(dim=-1, keepdim=True)

                dists = torch.cdist(centroids, sample_coords, p=2)
                sample_mask = sample_mask | (dists > radius)

            all_masked_in_sample = sample_mask.all(dim=-1)
            if all_masked_in_sample.any():
                sample_mask[all_masked_in_sample] = False

            sample_mask = sample_mask.repeat(self.nhead, 1, 1)

            list_inst_attn_masks.append(sample_mask)

        sem_attn_mask = None
        if sem_pred_mask is not None:
            sem_prob = sem_pred_mask.sigmoid()
            sem_attn_mask = (sem_prob < threshold)

            all_masked_sem = sem_attn_mask.all(dim=-1)
            sem_attn_mask[all_masked_sem] = False

            sem_attn_mask = sem_attn_mask.unsqueeze(0).repeat(self.nhead, 1, 1)

        return list_inst_attn_masks, sem_attn_mask

    def _set_aux_loss(self, outputs_class, outputs_seg_masks, sem_predictions_mask, inst_predictions_class):
        """这个函数用于生成模型的辅助损失（auxiliary losses），即中间层损失"""

        return [
            {"pred_logits": a, "pred_masks": b, "sem_pred_masks": c, "ins_pred_logits": d}
            for a, b, c, d in
            zip(outputs_class[:-1], outputs_seg_masks[:-1], sem_predictions_mask[:-1], inst_predictions_class[:-1])
        ]

    def _forward_single_decoder_layer(self, layer, queries, kv, query_pos, pos_embed, attn_mask, inst_q_offsets,
                                      entity_offset_from0):

        if self.isolation_attn and self.isolation_mode == "all":
            q = layer["self_attn"](
                queries,
                tgt_mask=None,
                query_pos=query_pos,
                isolation_attn=True,
                q_offset=inst_q_offsets,
            )
        else:
            q = layer["self_attn"](
                queries,
                tgt_mask=None,
                query_pos=query_pos,
                isolation_attn=False,
                q_offset=inst_q_offsets,
            )

        q = layer["cross_attn"](
            tgt=q,
            memory=kv,
            memory_mask=attn_mask,
            pos=pos_embed,
            query_pos=query_pos,
            isolation_attn=self.isolation_attn,
            q_offset=inst_q_offsets,
            e_offset=entity_offset_from0,
        )

        q = layer["ffn"](q)

        return q

    def forward(self, stage_list, primitive_features=None, primitive_coords=None, entity_offset=None):

        primitive_features = stage_list["primitive_features"]
        primitive_coords = stage_list["primitive_coords"]
        entity_offset = stage_list["entity_offset"]

        device = primitive_features.device
        B = len(entity_offset)
        mem_counts = torch.diff(entity_offset, prepend=torch.tensor([0], device=device)).tolist()

        mem_for_attn = self.input_proj(primitive_features)

        mem_for_mask1 = self.mask_feat_proj1(primitive_features)
        mem_for_mask = self.mask_feat_proj(primitive_features)

        pos_embed = self.get_pos_encs([primitive_coords])[0]

        ins_queries_list = []
        q_counts = []
        query_pos_list = []
        queries_entity_ids = []
        start_idx = 0

        ins_q = self.instance_query_proj(primitive_features)
        ins_queries_pos = self.instance_query_pos_proj(primitive_coords)

        for i, count in enumerate(mem_counts):
            end_idx = entity_offset[i]

            if self.training:

                n_keep = max(int(count * self.alpha), 1)

                relative_keep_idx = torch.randperm(count, device=device)[:n_keep]

                keep_idx = relative_keep_idx + start_idx

            else:
                n_keep = count
                keep_idx = torch.arange(start_idx, start_idx + n_keep, device=device)

            ins_keep_q = ins_q[keep_idx]
            ins_keep_q_pos = ins_queries_pos[keep_idx]

            ins_queries_list.append(ins_keep_q)
            query_pos_list.append(ins_keep_q_pos)

            q_counts.append(n_keep)
            queries_entity_ids.append(keep_idx)
            start_idx = end_idx

        inst_q_offsets = torch.cumsum(torch.tensor([0] + q_counts), dim=0).tolist()
        entity_offset_from0 = torch.cumsum(torch.tensor([0] + mem_counts), dim=0).tolist()

        ins_queries = torch.cat(ins_queries_list, dim=0)
        ins_query_pos = torch.cat(query_pos_list, dim=0)

        sem_q = self.semantic_query.weight
        cur_query_pos = self.sem_query_pos.weight

        queries = torch.cat([ins_queries, sem_q], dim=0)
        query_pos = torch.cat([ins_query_pos, cur_query_pos], dim=0)

        list_pred_inst_masks = []
        list_pred_sem_mask = []
        if self.share_kernels_head:
            for i in range(len(inst_q_offsets) - 1):
                qs, qe = inst_q_offsets[i], inst_q_offsets[i + 1]

                es, ee = entity_offset_from0[i], entity_offset_from0[i + 1]
                kernels = self.mask_kernel_head(queries.squeeze(1))
                pred_masks = torch.einsum("qc,nc->qn", kernels[qs:qe], mem_for_mask[es:ee])
                list_pred_inst_masks.append(pred_masks.unsqueeze(0))

                sem_pred_mask = torch.einsum("qc,nc->qn", kernels[-self.num_classes:], mem_for_mask[es:ee])
        else:
            for i in range(len(inst_q_offsets) - 1):

                qs, qe = inst_q_offsets[i], inst_q_offsets[i + 1]

                es, ee = entity_offset_from0[i], entity_offset_from0[i + 1]
                inst_kernel = self.mask_inst_kernel_head(queries.squeeze(1)[qs:qe])
                if not self.nshare_mlp:
                    pred_masks = torch.einsum("qc,nc->qn", inst_kernel, mem_for_mask[es:ee])
                else:
                    pos_embed_flat = pos_embed.squeeze(1)

                    pred_masks = torch.einsum("qc,nc->qn", inst_kernel, mem_for_mask1[es:ee])
                list_pred_inst_masks.append(pred_masks.unsqueeze(0))

                sem_kernel = self.mask_sem_kernel_head(queries.squeeze(1)[-self.num_classes:])
                sem_pred_mask = torch.einsum("qc,nc->qn", sem_kernel, mem_for_mask[es:ee])
                list_pred_sem_mask.append(sem_pred_mask)
        sem_masks = torch.cat(list_pred_sem_mask, dim=1)

        queries = queries.unsqueeze(1)
        kv = mem_for_attn.unsqueeze(1)
        query_pos = query_pos.unsqueeze(1)

        predictions_class = []
        inst_predictions_class = []
        predictions_mask = []
        sem_preds_mask = []

        for layer in self.transformerlayers:
            attn_mask, sem_attn_mask = self.get_refined_masked_attn_mask_from_list(
                list_pred_inst_masks,
                q_offsets=inst_q_offsets,
                e_offsets=entity_offset_from0,
                threshold=0.5,
                spatial_cfg=self.spatial_cfg,
                primitive_coords=primitive_coords,
                sem_pred_mask=sem_masks,
            )
            attn_mask.append(sem_attn_mask)

            q = self._forward_single_decoder_layer(layer, queries, kv, query_pos, pos_embed, attn_mask, inst_q_offsets,
                                                   entity_offset_from0)

            queries = q

            list_pred_inst_masks = []
            list_pred_sem_mask = []

            for i in range(len(inst_q_offsets) - 1):
                qs, qe = inst_q_offsets[i], inst_q_offsets[i + 1]

                es, ee = entity_offset_from0[i], entity_offset_from0[i + 1]
                inst_kernel = self.mask_inst_kernel_head(queries.squeeze(1)[qs:qe])

                pred_masks = torch.einsum("qc,nc->qn", inst_kernel, mem_for_mask1[es:ee])

                list_pred_inst_masks.append(pred_masks.unsqueeze(0))

                sem_kernel = self.mask_sem_kernel_head(queries.squeeze(1)[-self.num_classes:])
                sem_pred_mask = torch.einsum("qc,nc->qn", sem_kernel, mem_for_mask[es:ee])
                list_pred_sem_mask.append(sem_pred_mask)

            sem_masks = torch.cat(list_pred_sem_mask, dim=1)
            pred_logits = self.class_embed(queries.squeeze(1))

            inst_predictions_class.append(pred_logits[:-self.num_classes].unsqueeze(0))
            predictions_class.append(pred_logits.unsqueeze(0))
            sem_preds_mask.append(sem_masks.unsqueeze(0))
            predictions_mask.append(list_pred_inst_masks)

        return {
            "pred_logits": pred_logits.unsqueeze(0),
            "ins_pred_logits": pred_logits[:-self.num_classes].unsqueeze(0),
            "pred_masks": list_pred_inst_masks,
            "sem_pred_masks": sem_masks.unsqueeze(0),
            "q_counts": q_counts,
            "mem_counts": mem_counts,
            "queries_entity_ids": queries_entity_ids,
            "stage_list": stage_list,
            'aux_outputs': self._set_aux_loss(
                predictions_class, predictions_mask, sem_preds_mask, inst_predictions_class
            ),
        }


class VarlenSelfAttentionLayerV2(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False, en_semQ=True):
        super().__init__()
        self.en_semQ = en_semQ
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None, isolation_attn=None,
                     q_offset=None):
        if isolation_attn:
            output = torch.zeros_like(tgt)
            num_samples = len(q_offset) - 1

            for i in range(num_samples):
                s, e = q_offset[i], q_offset[i + 1]

                tgt_slice = tgt[s:e]
                pos_slice = query_pos[s:e] if query_pos is not None else None

                m_slice = tgt_mask[i] if (tgt_mask is not None and isinstance(tgt_mask, list)) else None
                kp_slice = None
                q = k = self.with_pos_embed(tgt_slice, pos_slice)
                res_inst = self.self_attn(q, k, value=tgt_slice, attn_mask=m_slice,
                                          key_padding_mask=kp_slice)[0]

                output[s:e] = self.norm(tgt_slice + self.dropout(res_inst))
            if self.en_semQ:
                s_sem = q_offset[-1]
                q_sem = tgt[s_sem:]
                p_sem = query_pos[s_sem:] if query_pos is not None else None

                m_slice = None
                kp_slice = None

                q_inp_sem = self.with_pos_embed(q_sem, p_sem)
                kv_inp_all = self.with_pos_embed(tgt, query_pos)

                res_sem = self.self_attn(q_inp_sem, kv_inp_all, value=tgt, attn_mask=m_slice,
                                         key_padding_mask=kp_slice)[0]
                output[s_sem:] = self.norm(q_sem + self.dropout(res_sem))
            return output
        else:

            q = k = self.with_pos_embed(tgt, query_pos)

            tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)[0]
            tgt = tgt + self.dropout(tgt2)
            tgt = self.norm(tgt)
            return tgt

    def forward_pre(self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None, isolation_attn=None,
                    q_offset=None):
        tgt2 = self.norm(tgt)
        if isolation_attn:
            output = torch.zeros_like(tgt)
            num_samples = len(q_offset) - 1

            for i in range(num_samples):
                s, e = q_offset[i], q_offset[i + 1]

                tgt_slice = tgt2[s:e]
                pos_slice = query_pos[s:e] if query_pos is not None else None

                m_slice = tgt_mask[i] if (tgt_mask is not None and isinstance(tgt_mask, list)) else None
                kp_slice = None
                q = k = self.with_pos_embed(tgt_slice, pos_slice)

                res_inst = self.self_attn(q, k, value=tgt_slice, attn_mask=m_slice,
                                          key_padding_mask=kp_slice)[0]

                output[s:e] = tgt_slice + self.dropout(res_inst)
            if self.en_semQ:
                s_sem = q_offset[-1]
                q_sem = tgt2[s_sem:]
                p_sem = query_pos[s_sem:] if query_pos is not None else None

                m_slice = None
                kp_slice = None

                q_inp_sem = self.with_pos_embed(q_sem, p_sem)
                kv_inp_all = self.with_pos_embed(tgt2, query_pos)

                res_sem = self.self_attn(q_inp_sem, kv_inp_all, value=tgt, attn_mask=m_slice,
                                         key_padding_mask=kp_slice)[0]
                output[s_sem:] = q_sem + self.dropout(res_sem)
            return output
        else:
            q = k = self.with_pos_embed(tgt2, query_pos)
            tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)[0]
            tgt = tgt + self.dropout(tgt2)

            return tgt

    def forward(self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None, isolation_attn=None,
                q_offset=None):

        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask, tgt_key_padding_mask, query_pos, isolation_attn, q_offset=q_offset)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos, isolation_attn, q_offset=q_offset)


class VarlenCrossAttentionLayerV2(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False, en_semQ=True):

        super().__init__()
        self.en_semQ = en_semQ
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):

        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory, memory_mask=None, memory_key_padding_mask=None, pos=None, query_pos=None,
                     isolation_attn=None, q_offset=None, e_offset=None):

        if isolation_attn:
            output = torch.zeros_like(tgt)
            num_samples = len(q_offset) - 1

            for i in range(num_samples):
                qs, qe = q_offset[i], q_offset[i + 1]

                es, ee = e_offset[i], e_offset[i + 1]

                tgt_slice = tgt[qs:qe]
                mem_slice = memory[es:ee]
                q_pos_slice = query_pos[qs:qe] if query_pos is not None else None
                k_pos_slice = pos[es:ee] if pos is not None else None

                m_slice = memory_mask[i] if (memory_mask is not None and isinstance(memory_mask, list)) else None
                kp_slice = None

                tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt_slice, q_pos_slice),
                                           key=self.with_pos_embed(mem_slice, k_pos_slice),
                                           value=mem_slice,
                                           attn_mask=m_slice,
                                           key_padding_mask=kp_slice)[0]

                output[qs:qe] = self.norm(tgt_slice + self.dropout(tgt2))
            if self.en_semQ:
                s_sem = q_offset[-1]
                q_sem = tgt[s_sem:]
                qp_sem = query_pos[s_sem:] if query_pos is not None else None

                m_slice = memory_mask[-1] if (memory_mask is not None and isinstance(memory_mask, list)) else None
                kp_slice = None

                res_sem = self.multihead_attn(
                    query=self.with_pos_embed(q_sem, qp_sem),
                    key=self.with_pos_embed(memory, pos),
                    value=memory,
                )[0]
                output[s_sem:] = self.norm(q_sem + self.dropout(res_sem))

            return output

        else:

            tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                       key=self.with_pos_embed(memory, pos),
                                       value=memory,
                                       attn_mask=memory_mask,
                                       key_padding_mask=memory_key_padding_mask)[0]
            tgt = tgt + self.dropout(tgt2)
            tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt, memory, memory_mask=None, memory_key_padding_mask=None, pos=None, query_pos=None,
                    isolation_attn=None, q_offset=None, e_offset=None):
        """q:tgt, k:memory, v:memory"""
        tgt2 = self.norm(tgt)
        if isolation_attn:
            output = torch.zeros_like(tgt)
            num_samples = len(q_offset) - 1

            for i in range(num_samples):
                qs, qe = q_offset[i], q_offset[i + 1]

                es, ee = e_offset[i], e_offset[i + 1]

                tgt_slice = tgt2[qs:qe]
                mem_slice = memory[es:ee]
                q_pos_slice = query_pos[qs:qe] if query_pos is not None else None
                k_pos_slice = pos[es:ee] if pos is not None else None

                m_slice = memory_mask[i] if (memory_mask is not None and isinstance(memory_mask, list)) else None
                kp_slice = None

                res = self.multihead_attn(query=self.with_pos_embed(tgt_slice, q_pos_slice),
                                          key=self.with_pos_embed(mem_slice, k_pos_slice),
                                          value=mem_slice,
                                          attn_mask=m_slice,
                                          key_padding_mask=kp_slice)[0]

                output[qs:qe] = tgt_slice + self.dropout(res)
            if self.en_semQ:
                s_sem = q_offset[-1]
                q_sem = tgt2[s_sem:]
                qp_sem = query_pos[s_sem:] if query_pos is not None else None

                m_slice = memory_mask[-1] if (memory_mask is not None and isinstance(memory_mask, list)) else None
                kp_slice = None

                res_sem = self.multihead_attn(
                    query=self.with_pos_embed(q_sem, qp_sem),
                    key=self.with_pos_embed(memory, pos),
                    value=memory,
                )[0]
                output[s_sem:] = q_sem + self.dropout(res_sem)

            return output
        else:
            tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                       key=self.with_pos_embed(memory, pos),
                                       value=memory, attn_mask=memory_mask,
                                       key_padding_mask=memory_key_padding_mask)[0]
            tgt = tgt + self.dropout(tgt2)

            return tgt

    def forward(self, tgt, memory, memory_mask=None, memory_key_padding_mask=None, pos=None, query_pos=None,
                isolation_attn=None, q_offset=None, e_offset=None):

        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos, isolation_attn=isolation_attn,
                                    q_offset=q_offset, e_offset=e_offset)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos, isolation_attn=isolation_attn,
                                 q_offset=q_offset, e_offset=e_offset)
