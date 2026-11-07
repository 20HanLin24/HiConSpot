import torch
import torch.nn as nn
import torch.nn.functional as F
from ..tools.pointops.functions import pointops

_inf = 1e9
_eps = 1e-12


def get_subscene_label(stage_n, stage_i, stage_list, target, nstride, num_classes, **kwargs):
    # o - num of points in each batch - b batch together - using BxN, used for finding the points in each example
    # calc the reduced size of points for the batch - n = [BxN], with each b clouds, i-th cloud containing N = b[i] points

    x = F.one_hot(target, num_classes)  # (n, ncls)
    return get_subscene_features(stage_n, stage_i, stage_list, x, nstride, **kwargs)


def get_subscene_features(stage_n, stage_i, stage_list, x, nstride, kr=None, extend=False, return_neighbor=False):
    """
    从指定阶段提取子场景特征并进行插值
    参数:
        stage_n: 阶段类型，通常为 'up' 或 'down'
        stage_i: 阶段索引，指示从哪个阶段提取特征
        stage_list: 包含各阶段点云信息的字典
        x: 当前特征，形状为 (N, C)，N为点数量，C为特征维度
        nstride: 每个阶段的下采样步长列表
        kr: KNN查询的邻居数，若为None则根据nstride自动计算
        extend: 是否扩展查询范围
        return_neighbor: 是否返回邻居信息
    返回:
        new_feat: 插值后的特征，形状为 (M, C)，M为目标点数量，由P_to确定
    decoder.py输入：
    stage_n="up", stage_i=step=4,3,2,1, stage_list, x=attn_mask(n,500), nstride=torch.tensor([4, 4, 4, 4])
    stage_list["up"]=up_list[
        {'p_out': p1, 'f_out': x1, 'offset': o1},  # n_0 = n, fdims = 32，coords(n, 3), newfeats(n, planes[0]), offsets1(b,)
        {'p_out': p2, 'f_out': x2, 'offset': o2},  # n_1，newcoords(m1, 3), newfeats(m1, planes[1]), new_offsets1(b,)
        {'p_out': p3, 'f_out': x3, 'offset': o3},  # n_2，newcoords(m2, 3), newfeats(m2, planes[2]), new_offsets2(b,)
        {'p_out': p4, 'f_out': x4, 'offset': o4},  # n_3，newcoords(m3, 3), newfeats(m3, planes[3]), new_offsets3(b,)
        {'p_out': p5, 'f_out': x5, 'offset': o5},  # n_4，newcoords(m4, 3), newfeats(m4, planes[4]), new_offsets4(b,)-fdims=512(extracted through dec5 = mlps)]
    不特指时，N为输入批次的总节点数，不特指时，n为输入节点特征张量节点数，当B=1、不增强时，N=n=max_num
    mi==new_offsetsi[-1]==N/stride**(i)，c==planes[i]=[64,128,256,512,1024]
    """
    # 特殊情况处理：若为第0阶段且不扩展，则直接返回输入特征
    if stage_i == 0 and not extend:
        return x.float()
    # 获取源点云（低分辨率，用于提供特征），这里实际是高分辨率
    stage_from = stage_list['up'][0]  # support
    # p_from: 点坐标，形状 (N, 3)，o_from: 批次分割索引，形状 (B,)
    p_from, o_from = stage_from['p_out'], stage_from['offset']  # p_from=coords(n, 3), o_from=offsets1(b,)
    # 获取目标点云（高分辨率，需要插值特征）
    stage_to = stage_list[stage_n][stage_i]  # query：{p5~2,x5~2,o5~2}
    # p_to: 目标点坐标，形状 (mi, 3)，o_to: 目标批次分割索引，形状 (B,)
    p_to, o_to = stage_to['p_out'], stage_to['offset']  # p_to=newcoords(mi, 3), o_to=new_offsetsi(b,)
    # 自动计算KNN邻居数（如果未指定）
    if kr is None:  # infer from sub-sampling (nstride) as default
        i = 1 if stage_i == 0 and extend else stage_i
        kr = torch.prod(nstride[:i])  # 【注】第一次kr这里是256
    # decoder.py输入：p_from(n, 3),p_to(mi, 3),x(n,500),o_from(b,),o_to(b,)->new_feat(mi, n)
    new_feat = pointops.interpolation(p_from.contiguous(), p_to.contiguous(), x.contiguous(), o_from, o_to, k=kr)
    # print(new_feat.shape)
    return new_feat  # (mi, 500)


'''
def get_subscene_features2(stage_n, stage_i, stage_list, x, nstride, kr=None, extend=False, return_neighbor=False):
    if stage_i == 0 and not extend:
        return x.float()

    if kr is None:  # infer from sub-sampling (nstride) as default
        i = 1 if stage_i == 0 and extend else stage_i
        kr = torch.prod(nstride[:i])

    stage_from = stage_list['up'][0]  # support
    p_from, o_from = stage_from['p_out'], stage_from['offset']

    stage_to = stage_list[stage_n][stage_i]  # query
    p_to, o_to = stage_to['p_out'], stage_to['offset']

    neighbor_idx, _ = pointops.knnquery(kr, p_from, p_to, o_from, o_to)  # (m, kr) - may have invalid neighbor

    # print('kr - x', kr, x.shape)
    # print(p_from.shape, p_to.shape)
    # print(o_from, o_to)
    # print('neighbor_idx.shape = ', neighbor_idx.shape)
    # print(neighbor_idx.min(), neighbor_idx.max(), (neighbor_idx == neighbor_idx.max()).int().sum())

    # neighbor_idx[neighbor_idx > p.shape[0]] = p.shape[0]  # collect all 0s if invalid neighbor
    # x = torch.cat([x, torch.zeros([1, self.config.num_classes])])

    neighbor_idx = neighbor_idx.view(-1).long()
    x = x[neighbor_idx, :].view(p_to.shape[0], kr, x.shape[1]) # (m, kr, ncls)
    #x = x.float().mean(-2)  # (m, ncls)
    #pool = nn.MaxPool1d(kr.item())
    pool = nn.AvgPool1d(kr.item())
    x = pool(x.transpose(1,2)).squeeze(-1)

    # x = x.float().sum(-2)  # (m, ncls)
    # cnt = (neighbor_idx < p.shape[0]).float().sum(-1, keepdim=True)  # (m, 1)
    # x /= cnt
    if return_neighbor:
        return x, neighbor_idx, kr
    return x


def batch_stack(features, offset, return_detail=False):
    shape_tail = features[0].shape[1:]
    offset = torch.cat([offset.new_zeros([1]), offset])
    batches_len = offset[1:] - offset[:-1]
    batches_pad = batches_len.max() - batches_len

    batches_feat = []
    for i, pad_n in enumerate(batches_pad):
        zeros = features.new_zeros([pad_n, *shape])
        batches_feat.append(torch.cat([features[offset[i]: offset[i+1]], zeros], 0))
    batches_feat = torch.stack(batches_feat)  # [B, Nmax, d]

    if return_detail:
        batches_feat, offset, batches_pad, batches_len
    return batches_feat


def get_boundary_mask(labels, neighbor_label=None, neighbor_idx=None, valid_mask=None, get_plain=False, get_cnt=False):
    """ assume all label valid indicated by valid_mask """
    labels_shape = labels.shape
    if neighbor_label is None:
        shape = [*neighbor_idx.shape, *labels.shape[1:]]  # [BxN, kr, ncls]
        neighbor_label = labels[neighbor_idx.view(-1).long(), ...].view(shape)
        print(shape, neighbor_idx.shape, labels.shape, neighbor_label.shape, flush=True)

    valid_neighbor = neighbor_label >= 0
    labels = labels.unsqueeze(-1)

    neq = labels != neighbor_label
    neq = torch.logical_and(neq, valid_neighbor)
    if get_cnt:
        bound = torch.sum(neq, dim=-1)
        bound = bound * valid_mask if valid_mask is not None else bound
    else:
        bound = torch.any(neq, dim=-1)
        bound = torch.logical_and(bound, valid_mask) if valid_mask is not None else bound  # mask out row of invalid center
    # assert len(bound.shape) == len(labels_shape), f'invalid shape - bound {bound.shape}, label {labels_shape}, neighbor label {neighbor_label.shape}, with valid_mask = {valid_mask}'

    if get_plain:
        # assert not get_cnt, 'no need to get plain if having cnt of boundary (together with valid neighbor)'
        eq = labels == neighbor_label  # same - T, diff - F, invalid center - F, invalid neighbor - F
        eq = torch.logical_or(eq, torch.logical_not(valid_neighbor))  # valid -> all eq => plain if neighbor all invalid
        plain = torch.all(eq, dim=-1)
        plain = torch.logical_and(plain, valid_mask) if valid_mask is not None else plain
        return bound, plain
    return bound  # [BxN]'''
