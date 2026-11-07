from typing import Tuple

import torch
from torch.autograd import Function
import torch.nn as nn
import os

try:
    import pointops_cuda
except ImportError:
    import warnings
    import os
    from torch.utils.cpp_extension import load

    # os.environ["CC"] =  r"\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.29.30133\bin\HostX64\x64\cl.exe"
    # os.environ["CXX"] = r"\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.29.30133\bin\HostX64\x64\cl.exe"
    # os.environ["CUDA_HOME"] = "ANACONDA/envs/cad_spotting"
    # os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9;7.5;8.6"  # 根据你的 GPU 调整（例如 RTX 4070 是 8.9）
    print(os.environ.get('CUDA_HOME'))

    warnings.warn("Unable to load pointops_cuda cpp extension.")
    pointops_cuda_src = os.path.join(os.path.dirname(__file__), "../src")  # modules/pointops/src
    pointops_cuda = load('pointops_cuda', [
        pointops_cuda_src + '/pointops_api.cpp',
        pointops_cuda_src + '/knnquery/knnquery_cuda.cpp',
        pointops_cuda_src + '/knnquery/knnquery_cuda_kernel.cu',
        pointops_cuda_src + '/interpolation/interpolation_cuda.cpp',
        pointops_cuda_src + '/interpolation/interpolation_cuda_kernel.cu',
        pointops_cuda_src + '/sampling/sampling_cuda.cpp',
        pointops_cuda_src + '/sampling/sampling_cuda_kernel.cu',
        pointops_cuda_src + '/subtraction/subtraction_cuda.cpp',
        pointops_cuda_src + '/subtraction/subtraction_cuda_kernel.cu',
        pointops_cuda_src + '/aggregation/aggregation_cuda.cpp',
        pointops_cuda_src + '/aggregation/aggregation_cuda_kernel.cu',
    ], build_directory=pointops_cuda_src, verbose=False)


class FurthestSampling(Function):
    @staticmethod
    # 原为(ctx, xyz, offset, new_offset)
    def forward(ctx, xyz, offset, new_offset):
        """
        input:
            xyz: (n, 3)：中心点坐标(max_num, 3)
            offset: (b)，一批中每个样本结束索引
            new_offset: (b)，记录offset采样后的结果，采样结果是节点在一批次中的位置索引(绝对位置)
        output: idx: (m)，m=new_offset[b - 1]
        【注】以数据集batch_size=1，stride=4的情况推导
        """
        assert xyz.is_contiguous()
        n, b, n_max = xyz.shape[0], offset.shape[0], offset[0]  # 一批次的总节点数量，批次数，批次中第一个特征矩阵的结束位置索引
        for i in range(1, b):
            # offset[i] - offset[i - 1]批次中的第i个特征矩阵的结束位置索引(相对该特征矩阵而言的位置)，也即每个矩阵的节点数
            n_max = max(offset[i] - offset[i - 1], n_max)  # 取批次中最大样本的节点数
        # torch.cuda.IntTensor(new_offset[b - 1].item()) 创建了一个形状为 (new_offset[b-1]) 的整型张量，存储在 GPU 上。此时，
        # 张量的初始值是未初始化的随机值（内存中的任意内容）。
        # .zero_() 方法将张量的所有元素原地修改为 0。原地操作会直接修改当前张量，而不是返回新的张量。
        # new_offset[b-1].item() 返回批次中最后一个样本的结束位置(相对样本本身)，即所有批次下采样后的总点数 m。
        idx = torch.cuda.IntTensor(new_offset[b - 1].item()).zero_()  # (max_num/4, )
        tmp = torch.cuda.FloatTensor(n).fill_(1e10)  # 创建一个形状为(n)的整型张量，用1e10填充，(max_num, )
        pointops_cuda.furthestsampling_cuda(b, n_max, xyz, offset, new_offset, tmp, idx)  # (max_num/4, )
        del tmp
        return idx  # (max_num/4, )


furthestsampling = FurthestSampling.apply


class SectorizedFurthestSampling(Function):
    @staticmethod
    def forward(ctx, xyz, offset, new_offset, num_sectors, min_points=10000):
        """
        input: xyz: (n, 3), offset: (b), new_offset: (b)
        output: idx: (m)
        """
        assert xyz.is_contiguous()

        # cut into batches
        last_offset = 0
        sizes = []
        new_sizes = []
        indices = []
        for i in range(offset.shape[0]):
            size = offset[i] - last_offset
            if size < min_points:
                tmp_num_sectors = 1
            else:
                tmp_num_sectors = num_sectors
            batch_xyz = xyz[last_offset:last_offset + size]
            angle = torch.atan2(batch_xyz[:, 0], batch_xyz[:, 1])  # [0, 2*pi]
            sector_range = torch.linspace(angle.min(), angle.max() + 1e-4, tmp_num_sectors + 1)
            for s in range(tmp_num_sectors):
                indices.append(
                    torch.where((angle >= sector_range[s]) & (angle < sector_range[s + 1]))[0] + last_offset
                )
                sizes.append(indices[-1].shape[0])
            if i > 0:
                new_size = (new_offset[i] - new_offset[i - 1]).item()
            else:
                new_size = new_offset[i].item()
            new_sizes_this_batch = [new_size // tmp_num_sectors for i in range(tmp_num_sectors)]
            new_sizes_this_batch[-1] += new_size % tmp_num_sectors
            new_sizes += new_sizes_this_batch
            last_offset = offset[i]

        sizes = torch.tensor(sizes, dtype=torch.long).to(offset)
        sector_offset = sizes.cumsum(dim=0)
        new_sizes = torch.tensor(new_sizes, dtype=torch.long).to(offset)
        new_sector_offset = new_sizes.cumsum(dim=0)
        indices = torch.cat(indices).long().to(offset.device)
        sector_xyz = xyz[indices].contiguous()

        # transform to sectors
        new_xyz = []
        n, b, n_max = sector_xyz.shape[0], sector_offset.shape[0], sector_offset[0]
        for i in range(1, b):
            n_max = max(sector_offset[i] - sector_offset[i - 1], n_max)
        idx = torch.cuda.IntTensor(new_sector_offset[b - 1].item()).zero_()
        tmp = torch.cuda.FloatTensor(n).fill_(1e10)
        pointops_cuda.furthestsampling_cuda(b, n_max, sector_xyz, sector_offset.int(), new_sector_offset.int(), tmp,
                                            idx)
        idx = indices[idx.long()]
        del tmp
        del sector_xyz
        return idx


sectorized_fps = SectorizedFurthestSampling.apply


class KNNQuery(Function):
    @staticmethod
    def forward(ctx, nsample, xyz, new_xyz, offset, new_offset):
        """
        对采样点集new_xyz进行knn查询，获取每个点的nsample个最邻近点id和它们间的距离
        ctx: 上下文对象，用于存储反向传播所需信息
        nsample: 每个查询点要查找的邻居数(K)
        input: xyz: (n, 3), new_xyz: (m, 3), offset: (b), new_offset: (b)
        output: idx: (m, nsample), dist2: (m, nsample)
        m=new_offset[b - 1]，如果stride=4，且b=1，则 m=n/4，n=max_num
        """
        if new_xyz is None:
            new_xyz = xyz
        assert xyz.is_contiguous() and new_xyz.is_contiguous()
        m = new_xyz.shape[0]
        # idx = torch.cuda.IntTensor(m, nsample).zero_() 旧方法弃用
        # dist2 = torch.cuda.FloatTensor(m, nsample).zero_()  # 创建一个(m, nsample)矩阵，用0填充
        idx = torch.zeros(m, nsample, dtype=torch.int32, device='cuda')      # 创建一个(m, nsample)矩阵，用0填充
        dist2 = torch.zeros(m, nsample, dtype=torch.float, device='cuda')    # 创建一个(m, nsample)矩阵，用0填充
        pointops_cuda.knnquery_cuda(m, nsample, xyz, new_xyz, offset, new_offset, idx, dist2)
        return idx, torch.sqrt(dist2)


knnquery = KNNQuery.apply


class Grouping(Function):
    @staticmethod
    def forward(ctx, input, idx):
        """
        input: input: (n, c), idx : (m, nsample)
        output: (m, nsample, c)
        """
        assert input.is_contiguous() and idx.is_contiguous()
        m, nsample, n, c = idx.shape[0], idx.shape[1], input.shape[0], input.shape[1]
        output = torch.cuda.FloatTensor(m, nsample, c)
        pointops_cuda.grouping_forward_cuda(m, nsample, c, input, idx, output)
        ctx.n = n
        ctx.save_for_backward(idx)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: grad_out: (m, c, nsample)
        output: (n, c), None
        """
        n = ctx.n
        idx, = ctx.saved_tensors
        m, nsample, c = grad_output.shape
        grad_input = torch.cuda.FloatTensor(n, c).zero_()
        pointops_cuda.grouping_backward_cuda(m, nsample, c, grad_output, idx, grad_input)
        return grad_input, None


grouping = Grouping.apply


def queryandgroup(nsample, xyz, new_xyz, feat, idx, offset, new_offset, use_xyz=True):
    """
    实现了点云处理中的局部邻域特征聚合操作，是 PointNet++ 等点云深度学习模型的核心组件。
    它通过 KNN 或自定义索引查找每个点的邻居，并将邻居的特征与相对坐标结合，形成局部特征组
    【new_xyz是最远点采样得到的点集时】：
    input: xyz: (n, 3), new_xyz: (m, 3), feat: (n, c), idx: (m, nsample)orNone, offset: (b), new_offset: (b)，
    output: new_feat: (m, nsample, 3+c), cat(采样点坐标和它的最邻近点xyz坐标的差值张量，采样点的最邻近点的节点特征张量)
            grouped_idx: (m, nsample)，采样点集中每个点的nsample个最邻近点索引id
    m=new_offset[b - 1]，如果stride=4，且b=1，则 m=n/4，n=max_num，第一次输入时c=9
    【new_xyz==xyz时】：m==n
    input: xyz: (n, 3), new_xyz: (n, 3), feat: (n, c), idx: None, offset: (b), new_offset: (b)，offset==new_offset
    output: new_feat: (n, nsample, 3+c), cat(输入点坐标和它的最邻近点xyz坐标的差值张量，输入点的最邻近点的节点特征张量)
            grouped_idx: (n, nsample)，输入new_xyz点集中每个点的nsample个最邻近点索引id
    """
    assert xyz.is_contiguous() and new_xyz.is_contiguous() and feat.is_contiguous()
    if new_xyz is None:
        new_xyz = xyz
    if idx is None:
        # knnquery返回: idx: (m, nsample), dist2: (m, nsample)，即对采样点集new_xyz在xyz中进行knn查询采样，获取每个点的nsample个最邻近点索引id和它们间的距离
        # knnquery返回: idx: (n, nsample), dist2: (n, nsample)，即对输入点集new_xyz在xyz中进行knn查询采样，获取每个点的nsample个最邻近点索引id和它们间的距离
        idx, _ = knnquery(nsample, xyz, new_xyz, offset, new_offset)  # (m, nsample) 或 (n, nsample)

    n, m, c = xyz.shape[0], new_xyz.shape[0], feat.shape[1]  # 一批次中总节点数，(最远点/输入点)采样数，当前节点特征张量的特征数
    # idx.view(-1)(m, nsample)->展平->(m*nsample)
    # idx.view(-1)(n, nsample)->展平->(n*nsample)
    grouped_xyz = xyz[idx.view(-1).long(), :].view(m, nsample, 3)  # (m, nsample, 3)，获取new_xyz点集的最邻近点的xyz坐标张量
    # grouped_xyz = grouping(xyz, idx) # (m, nsample, 3)
    # new_xyz.unsqueeze(1)->(m, 1, 3)
    # new_xyz.unsqueeze(1)->(n, 1, 3)
    grouped_xyz -= new_xyz.unsqueeze(1)  # (m, nsample, 3)，获取new_xyz点集的最邻近点xyz坐标和它自己的点坐标的差值张量
    grouped_feat = feat[idx.view(-1).long(), :].view(m, nsample, c)  # (m, nsample, c)，获取new_xyz点集的最邻近点的节点特征张量
    # grouped_feat = grouping(feat, idx) # (m, nsample, c)

    if use_xyz:
        # 把grouped_xyz, grouped_feat按最后一维度方向拼接
        # (m, nsample, 3+c),(m, nsample) 或 (n, nsample, 3+c),(n, nsample)
        return torch.cat((grouped_xyz, grouped_feat), -1), idx.long()
    else:
        return grouped_feat


class Subtraction(Function):
    @staticmethod
    def forward(ctx, input1, input2, idx):
        """
        input: input1: (n, c), input2: (n, c), idx: (n, nsample)
        output:  (n, nsample, c)
        """
        assert input1.is_contiguous() and input2.is_contiguous()
        n, c = input1.shape;
        nsample = idx.shape[-1]
        output = torch.cuda.FloatTensor(n, nsample, c).zero_()
        pointops_cuda.subtraction_forward_cuda(n, nsample, c, input1, input2, idx, output)
        ctx.save_for_backward(idx)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: grad_out: (n, nsample, c)
        output: grad_input1: (n, c), grad_input2: (n, c)
        """
        idx, = ctx.saved_tensors
        n, nsample, c = grad_output.shape
        grad_input1 = torch.cuda.FloatTensor(n, c).zero_()
        grad_input2 = torch.cuda.FloatTensor(n, c).zero_()
        pointops_cuda.subtraction_backward_cuda(n, nsample, c, idx, grad_output, grad_input1, grad_input2)
        return grad_input1, grad_input2, None


subtraction = Subtraction.apply


class Aggregation(Function):
    @staticmethod
    def forward(ctx, input, position, weight, idx):
        """
        input: input: (n, c), position: (n, nsample, c), weight : (n, nsample, c'), idx: (n, nsample)
        output: (n, c)
        """
        assert input.is_contiguous() and position.is_contiguous() and weight.is_contiguous()
        n, nsample, c = position.shape;
        w_c = weight.shape[-1]
        output = torch.cuda.FloatTensor(n, c).zero_()
        pointops_cuda.aggregation_forward_cuda(n, nsample, c, w_c, input, position, weight, idx, output)
        ctx.save_for_backward(input, position, weight, idx)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: grad_out: (n, c)
        output: grad_input: (n, c), grad_position: (n, nsample, c), grad_weight : (n, nsample, c')
        """
        input, position, weight, idx = ctx.saved_tensors
        n, nsample, c = position.shape;
        w_c = weight.shape[-1]
        grad_input = torch.cuda.FloatTensor(n, c).zero_()
        grad_position = torch.cuda.FloatTensor(n, nsample, c).zero_()
        grad_weight = torch.cuda.FloatTensor(n, nsample, w_c).zero_()
        pointops_cuda.aggregation_backward_cuda(n, nsample, c, w_c, input, position, weight, idx, grad_output,
                                                grad_input, grad_position, grad_weight)
        return grad_input, grad_position, grad_weight, None


aggregation = Aggregation.apply


def interpolation(xyz, new_xyz, feat, offset, new_offset, k=3):
    """
    功能：将低分辨率点云(xyz)的特征(feat)插值到高分辨率点云(new_xyz)上
    input: xyz: (m, 3), new_xyz: (n, 3), feat: (m, c), offset: (b), new_offset: (b)
    output: (n, c)
    参数:
        xyz: 低分辨率点云坐标，形状 (m, 3)
        new_xyz: 高分辨率点云坐标，形状 (n, 3)
        feat: 低分辨率点云的特征，形状 (m, c)，但c是高分辨率点云的特征维度，也就是说低分辨点云要先经过全连接层处理，把特征和高分辨率点云对齐
        offset: 批次分割索引，标记每个批次在xyz中的结束位置
        new_offset: 批次分割索引，标记每个批次在new_xyz中的结束位置
        k: 每个点的邻居数，默认为3
    返回:
        new_feat: 插值后的高分辨率点云特征，形状 (n, c)

    该函数用于以下模块，此时输入如下：
    deci[0]，i=[1.2.3.4]:
        xyz(mi, 3), new_xyz(mi-1, 3), feat(mi, c1), offset(b,), new_offset-1(b,),
        mi-1==new_offsetsi-1[-1]==max_num/stride**(i-1)，c1==planes[i-1]=[64,128,256,512]
        mi==new_offsetsi[-1]==max_num/stride**(i)，c2==planes[i]=[128,256,512,1024]
        m0==max_num
        这里n=mi-1，m=mi，c=c1
    """
    assert xyz.is_contiguous() and new_xyz.is_contiguous() and feat.is_contiguous()
    # idx: (n, k), dist2: (n, k)，对每个高分辨率点new_xyz，在低分辨率点xyz中查找k个最近邻点，获取每个点的k个最邻近点索引id和它们间的距离
    idx, dist = knnquery(k, xyz, new_xyz, offset, new_offset)  # (n, k), (n, k)
    # 计算反距离权重
    dist_recip = 1.0 / (dist + 1e-8)  # (n, k)
    norm = torch.sum(dist_recip, dim=1, keepdim=True)  # 横向累加，(n, k)->(n, 1)，如果keepdim=False，则(n, )
    weight = dist_recip / norm  # 反距离权重标准化，(n, k)
    # 加权特征聚合
    new_feat = torch.cuda.FloatTensor(new_xyz.shape[0], feat.shape[1]).zero_()  # 创建一个0浮点矩阵，形为(n, c)，c是高分辨率点云的特征维度
    for i in range(k):
        # 检查idx是否越界
        assert (idx >= 0).all(), "Negative index in idx"
        assert (idx < feat.size(0)).all(), f"Index {idx.max()} out of bounds for feat size {feat.size(0)}"
        # 检查weight是否为非负数（若权重应为概率）
        assert (weight >= 0).all(), "Negative weight detected"
        # 因为是idx的元素来自低分辨率点xyz的索引，因此可以通过该索引从低分辨率点云的特征feat中获取特征行向量，提取得到的行向量个数为n个
        # feat[idx[:, i].long(), :]->(n, c)，weight[:, i]->(n, 1)
        # 每次提取n个特征向量形成向量组与对位的反距离权重逐向量相乘，再逐次累加新提取的特征矩阵最终获得新的特征矩阵new_feat(n, c)
        new_feat += feat[idx[:, i].long(), :] * weight[:, i].unsqueeze(-1)
    return new_feat  # (n, c)


class Interpolation(Function):
    @staticmethod
    def forward(ctx, xyz, new_xyz, input, offset, new_offset, k=3):
        """
        input: xyz: (m, 3), new_xyz: (n, 3), input: (m, c), offset: (b), new_offset: (b)
        output: (n, c)
        """
        assert xyz.is_contiguous() and new_xyz.is_contiguous() and input.is_contiguous()
        idx, dist = knnquery(k, xyz, new_xyz, offset, new_offset)  # (n, k), (n, k)
        dist_recip = 1.0 / (dist + 1e-8)  # (n, k)
        norm = torch.sum(dist_recip, dim=1, keepdim=True)
        weight = dist_recip / norm  # (n, k)

        n, c, m = new_xyz.shape[0], input.shape[1], input.shape[0]
        output = torch.cuda.FloatTensor(n, c).zero_()
        pointops_cuda.interpolation_forward_cuda(n, c, k, input, idx, weight, output)
        ctx.m, ctx.k = m, k
        ctx.save_for_backward(idx, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: xyz: (m, 3), new_xyz: (n, 3), input: (m, c), offset: (b), new_offset: (b)
        output: (n, c)
        """
        m, k = ctx.m, ctx.k
        idx, weight = ctx.saved_tensors
        n, c = grad_output.shape
        grad_input = torch.cuda.FloatTensor(m, c).zero_()
        pointops_cuda.interpolation_backward_cuda(n, c, k, grad_output, idx, weight, grad_input)
        return None, None, grad_input, None, None, None


interpolation2 = Interpolation.apply
