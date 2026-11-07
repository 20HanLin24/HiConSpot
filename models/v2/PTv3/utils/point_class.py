import sys
from addict import Dict  # 增强版字典，支持属性式访问（如dict.key）
import torch
import torch.nn as nn
import spconv.pytorch as spconv  # 稀疏卷积库（处理点云稀疏性）

from collections import OrderedDict

# FlashAttention加速注意力计算（可选）
try:
    import flash_attn
except ImportError:
    flash_attn = None

from ..serialization import encode



# --------------------------- 工具函数（点云批次处理） ---------------------------
@torch.inference_mode()  # 推理模式：禁用梯度计算，提升效率
def offset2bincount(offset):
    """
    从offset计算每个批次的点数量（bincount）
    offset: 批次分割索引，如[100, 200, 300]表示batch0有100个点，batch1有100个点...
    返回：每个批次的点数量，如[100, 100, 100]
    """
    return torch.diff(
        offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long)
    )


@torch.inference_mode()
def offset2batch(offset):
    """
    从offset生成batch_id（每个点对应的批次索引）
    示例：offset=[100,200] → batch_id=[0]*100 + [1]*100
    """
    bincount = offset2bincount(offset)
    return torch.arange(
        len(bincount), device=offset.device, dtype=torch.long
    ).repeat_interleave(bincount)


@torch.inference_mode()
def batch2offset(batch):
    """
    从batch_id生成offset（批次分割索引）
    示例：batch_id=[0]*100 + [1]*100 → offset=[100, 200]
    """
    return torch.cumsum(batch.bincount(), dim=0).long()


class Point(Dict):
    """
    Point Structure of Pointcept

    A Point (point cloud) in Pointcept is a dictionary that contains various properties of
    a batched point cloud. The property with the following names have a specific definition
    as follows:

    - "coord": original coordinate of point cloud;
    - "grid_coord": grid coordinate for specific grid size (related to GridSampling);
    Point also support the following optional attributes:
    - "offset": if not exist, initialized as batch size is 1;
    - "batch": if not exist, initialized as batch size is 1;
    - "feat": feature of point cloud, default input of model;
    - "grid_size": Grid size of point cloud (related to GridSampling);
    (related to Serialization)
    - "serialized_depth": depth of serialization, 2 ** depth * grid_size describe the maximum of point cloud range;
    - "serialized_code": a list of serialization codes;
    - "serialized_order": a list of serialization order determined by code;
    - "serialized_inverse": a list of inverse mapping determined by code;
    (related to Sparsify: SpConv)
    - "sparse_shape": Sparse shape for Sparse Conv Tensor;
    - "sparse_conv_feat": SparseConvTensor init with information provide by Point;
        Pointcept框架的点云核心数据结构，封装批次点云的所有属性
    核心属性（固定含义）：
    - coord: 原始坐标 (N, 3)
    - grid_coord: 栅格化后的坐标（对应grid_size）(N, 3)
    可选属性：
    - offset/batch: 批次分割索引/每个点的批次ID（二选一，自动推导）
    - feat: 点云特征 (N, C)
    - grid_size: 栅格化步长（如0.01m）
    序列化相关：
    - serialized_depth: 序列化深度（编码立方体边长=2^depth), 
        在 Point Transformer V3 (PTv3) 中，深度 (depth) 是一个至关重要的超参数，它直接决定了空间离散化的精度和序列化编码的长度。
        depth 是二进制位深度。它定义了一个边长为2^depth的虚拟立方体网格。如果 depth = 10, 意味着在 X、Y、Z 每个轴上，空间被划分为 2^10=1024个等分。
        与grid_size关系: 最大网格索引 = 最大coord坐标/grid_size, depth 必须足以容纳(大于)这个最大索引。我们可以把 depth 类比为数字图像的分辨率。
    - serialized_code: 序列化编码（如Z-order编码）(K, N) K为编码顺序数
    - serialized_order: 编码排序后的索引 (K, N)
    - serialized_inverse: 排序的逆映射 (K, N)
    稀疏卷积相关：
    - sparse_shape: 稀疏卷积的空间形状
    - sparse_conv_feat: spconv的SparseConvTensor对象
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # If one of "offset" or "batch" do not exist, generate by the existing one
        # 自动补全offset/batch：只要有一个，就推导另一个
        if "batch" not in self.keys() and "offset" in self.keys():
            self["batch"] = offset2batch(self.offset)
        elif "offset" not in self.keys() and "batch" in self.keys():
            self["offset"] = batch2offset(self.batch)

    def serialization(self, order="z", depth=None, shuffle_orders=False):
        """
        Point Cloud Serialization
        relay on ["grid_coord" or "coord" + "grid_size", "batch", "feat"]
        点云序列化：将3D点坐标编码为一维序列（如Z-order/Hilbert）
        依赖：batch + (grid_coord 或 coord+grid_size) + feat
        Args:
            order: 编码顺序（支持多顺序，如["z", "hilbert"]）
            depth: 序列化深度（立方体边长=2^depth），自动推导时取grid_coord最大值的比特长度
            shuffle_orders: 是否打乱编码顺序（增强鲁棒性）
        """
        assert "batch" in self.keys()
        # 自动生成grid_coord（如果未提供）
        if "grid_coord" not in self.keys():
            # if you don't want to operate GridSampling in data augmentation,
            # please add the following augmentation into your pipline:
            # dict(type="Copy", keys_dict={"grid_size": 0.01}),
            # (adjust `grid_size` to what your want)
            assert {"grid_size", "coord"}.issubset(self.keys())
            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc"
            ).int()  # 等价于[(self.coord - self.coord.min(0)[0])/self.grid_size],[]表示向下取整
        # 自动推导序列化深度
        if depth is None:
            # Adaptive measure the depth of serialization cube (length = 2 ^ depth)
            depth = int(self.grid_coord.max()).bit_length()
        self["serialized_depth"] = depth

        # 编码长度校验：防止溢出（int64最多63位）
        # Maximum bit length for serialization code is 63 (int64)
        assert depth * 3 + len(self.offset).bit_length() <= 63
        # Here we follow OCNN and set the depth limitation to 16 (48bit) for the point position.
        # Although depth is limited to less than 16, we can encode a 655.36^3 (2^16 * 0.01) meter^3
        # cube with a grid size of 0.01 meter. We consider it is enough for the current stage.
        # We can unlock the limitation by optimizing the z-order encoding function if necessary.
        assert depth <= 16  # 限制深度（0.01m栅格下可覆盖655.36^3 m³）

        # The serialization codes are arranged as following structures:
        # [Order1 ([n]),
        #  Order2 ([n]),
        #   ...
        #  OrderN ([n])] (k, n)
        # 生成多顺序的序列化编码
        code = [
            encode(self.grid_coord, self.batch, depth, order=order_) for order_ in order
        ]
        code = torch.stack(code)  # (K, N) K=len(order)，n为编码后序列的长度。
        # 对编码排序，得到顺序和逆映射
        order = torch.argsort(code)  # (K, N) 每个顺序下的排序索引
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )  # 逆映射：排序后→原始索引

        # 打乱编码顺序（数据增强）
        if shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        # 保存序列化结果
        self["serialized_code"] = code
        self["serialized_order"] = order
        self["serialized_inverse"] = inverse

    def sparsify(self, pad=96):
        """
        稀疏化, sparsify 方法的作用是把“一堆点”转化成“一个稀疏的网格体”。
        个稀疏张量由三部分组成，这个方法就是在准备这三部分：
        特征 (Features)：点本身的属性（如颜色、法向量或隐藏特征）。
        索引 (Indices)：点在网格里的坐标。
        形状 (Spatial Shape)：整个网格盒子的总大小。
        Point Cloud Serialization
        Point cloud is sparse, here we use "sparsify" to specifically refer to
        preparing "spconv.SparseConvTensor" for SpConv.
        relay on ["grid_coord" or "coord" + "grid_size", "batch", "feat"]
        pad: padding sparse for sparse shape.
        点云稀疏化：转换为spconv的SparseConvTensor（适配稀疏卷积）
        依赖：feat + batch + (grid_coord 或 coord+grid_size)
        Args:
            pad: 稀疏形状的填充值（防止越界）
        """
        assert {"feat", "batch"}.issubset(self.keys())
        # 自动生成grid_coord（如果未提供）
        if "grid_coord" not in self.keys():
            # if you don't want to operate GridSampling in data augmentation,
            # please add the following augmentation into your pipline:
            # dict(type="Copy", keys_dict={"grid_size": 0.01}),
            # (adjust `grid_size` to what your want)
            assert {"grid_size", "coord"}.issubset(self.keys())
            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc"
            ).int()

        # 计算稀疏卷积所需的空间形状，(进行pad单位个扩充后)
        if "sparse_shape" in self.keys():
            sparse_shape = self.sparse_shape
        else:
            sparse_shape = torch.add(
                torch.max(self.grid_coord, dim=0).values, pad
            ).tolist()  # 等价于torch.max(self.grid_coord, dim=0).values + pad
        # 构建SparseConvTensor（spconv核心数据结构）
        sparse_conv_feat = spconv.SparseConvTensor(
            features=self.feat,  # 点特征
            indices=torch.cat(
                [self.batch.unsqueeze(-1).int(), self.grid_coord.int()], dim=1
            ).contiguous(),  # 稀疏索引：[batch_id, x, y, z] (N, 4)
            spatial_shape=sparse_shape,  # 空间形状 (D, H, W)
            batch_size=self.batch[-1].tolist() + 1,  # 批次大小
        )
        self["sparse_shape"] = sparse_shape
        self["sparse_conv_feat"] = sparse_conv_feat

    def sparsify_2d(self, pad=96):
        """
        稀疏化, sparsify 方法的作用是把“一堆点”转化成“一个稀疏的网格体”。
        个稀疏张量由三部分组成，这个方法就是在准备这三部分：
        特征 (Features)：点本身的属性（如颜色、法向量或隐藏特征）。
        索引 (Indices)：点在网格里的坐标。
        形状 (Spatial Shape)：整个网格盒子的总大小。
        Point Cloud Serialization
        Point cloud is sparse, here we use "sparsify" to specifically refer to
        preparing "spconv.SparseConvTensor" for SpConv.
        relay on ["grid_coord" or "coord" + "grid_size", "batch", "feat"]
        pad: padding sparse for sparse shape.
        点云稀疏化：转换为spconv的SparseConvTensor（适配稀疏卷积）
        依赖：feat + batch + (grid_coord 或 coord+grid_size)
        Args:
            pad: 稀疏形状的填充值（防止越界）
        """
        assert {"feat", "batch"}.issubset(self.keys())
        # 自动生成grid_coord（如果未提供）
        if "grid_coord" not in self.keys():
            # if you don't want to operate GridSampling in data augmentation,
            # please add the following augmentation into your pipline:
            # dict(type="Copy", keys_dict={"grid_size": 0.01}),
            # (adjust `grid_size` to what your want)
            assert {"grid_size", "coord"}.issubset(self.keys())
            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc"
            ).int()

        # 计算稀疏卷积所需的空间形状，(进行pad单位个扩充后)
        if "sparse_shape" in self.keys():
            sparse_shape_2d = self.sparse_shape
        else:
            sparse_shape_2d = torch.add(
                torch.max(self.grid_coord[:, :2], dim=0).values, pad
            ).tolist()  # 等价于torch.max(self.grid_coord, dim=0).values + pad
        # 构建SparseConvTensor（spconv核心数据结构）
        sparse_conv_feat = spconv.SparseConvTensor(
            features=self.feat,  # 点特征
            indices=torch.cat(
                [self.batch.unsqueeze(-1).int(), self.grid_coord[:, :2].int()], dim=1
            ).contiguous(),  # 稀疏索引：[batch_id, x, y, z] (N, 4)
            spatial_shape=sparse_shape_2d,  # 空间形状 (D, H, W)
            batch_size=self.batch[-1].tolist() + 1,  # 批次大小
        )
        self["sparse_shape"] = sparse_shape_2d
        self["sparse_conv_feat"] = sparse_conv_feat


class PointModule(nn.Module):
    r"""PointModule
    placeholder, all module subclass from this will take Point in PointSequential.
    PointModule基类
    所有自定义点云模块都继承此类，用于PointSequential的识别
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PointSequential(PointModule):
    r"""A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.
    点云版nn.Sequential，支持多类型模块的顺序执行
    支持的模块类型：
    1. PointModule：直接传入Point对象执行
    2. spconv模块：处理Point中的sparse_conv_feat
    3. 普通PyTorch模块：处理Point中的feat
    kernel_size=5,
    padding=1,
    bias=False,
    indice_key="stem",
    """

    def __init__(self, *args, **kwargs):
        '''*args是任意参数组成的元组, **kwargs是任意键值对组成的字典'''
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        for name, module in kwargs.items():
            if sys.version_info < (3, 6):
                raise ValueError("kwargs only supported in py36+")
            if name in self._modules:
                raise ValueError("name exists.")
            self.add_module(name, module)

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError("index {} is out of range".format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    def add(self, module, name=None):
        if name is None:
            name = str(len(self._modules))
            if name in self._modules:
                raise KeyError("name exists")
        self.add_module(name, module)

    def forward(self, input):
        # 【新增：修复 PyTorch Checkpoint 导致的 Point 类型退化问题】
        if type(input) is dict:
            input = Point(input)  # 重新将其包装为 Point 对象

        for k, module in self._modules.items():
            # Point module
            if isinstance(module, PointModule):
                input = module(input)
            # Spconv module
            elif spconv.modules.is_spconv_module(module):
                if isinstance(input, Point):
                    input.sparse_conv_feat = module(input.sparse_conv_feat)
                    input.feat = input.sparse_conv_feat.features
                else:
                    input = module(input)
            # PyTorch module
            else:
                if isinstance(input, Point):
                    input.feat = module(input.feat)
                    if "sparse_conv_feat" in input.keys():
                        input.sparse_conv_feat = input.sparse_conv_feat.replace_feature(
                            input.feat
                        )
                elif isinstance(input, spconv.SparseConvTensor):
                    if input.indices.shape[0] != 0:
                        input = input.replace_feature(module(input.features))
                else:
                    input = module(input)
        return input

