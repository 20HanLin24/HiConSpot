import torch.optim
from typing import Any, Dict, List, Set
import copy
import itertools


def build_optimizer(model, optim_cfg):
    assert "type" in optim_cfg
    _optim_cfg = optim_cfg.copy()
    optim_type = _optim_cfg.pop("type")
    optim = getattr(torch.optim, optim_type)

    return optim(filter(lambda p: p.requires_grad, model.parameters()), **_optim_cfg)


def build_new_optimizer(model, args):
    """
    args默认配置如下：
    optimizer:
      type: 'AdamW'
      lr: 0.0001
      weight_decay: 0.0001
      weight_decay_embed: 0.0
      decoder_multiplier: 1.0
      clip_gradients_enabled: True
      clip_gradients_type: "full_model"
      clip_gradients_norm_type: 2.0
      clip_gradients_value: 0.01
    """
    weight_decay_norm = args.weight_decay
    weight_decay_embed = args.weight_decay_embed
    defaults = {}
    defaults["lr"] = args.lr
    defaults["weight_decay"] = args.weight_decay
    # 可能用到的标准化类型
    norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )
    # 嵌套字典的列表，存储参数组的列表，每个元素是一个字典，包含参数和对应的超参数
    params: List[Dict[str, Any]] = []
    # 元素为torch.nn.parameter.Parameter的集合，使用集合记录已处理的参数，避免重复添加
    memo: Set[torch.nn.parameter.Parameter] = set()
    # 遍历模型的所有模块（递归）
    for module_name, module in model.named_modules():
        # 获取当前模块的所有参数（不递归子模块，避免重复）
        for module_param_name, value in module.named_parameters(recurse=False):
            # 跳过不需要梯度的参数（如冻结的层）
            if not value.requires_grad:
                continue
            # 跳过已处理的参数（避免嵌套模块重复添加）
            if value in memo:
                continue
            # 标记参数为已处理
            memo.add(value)
            # 复制默认超参数配置
            hyperparams = copy.copy(defaults)
            # 策略1：对解码器模块的参数应用学习率倍增
            if "decoder" in module_name:
                hyperparams["lr"] = hyperparams["lr"] * args.decoder_multiplier
            # # 策略2：对位置编码参数禁用权重衰减（常见于Transformer模型）
            if "relative_position_bias_table" in module_param_name or "absolute_pos_embed" in module_param_name:
                print(module_param_name)
                hyperparams["weight_decay"] = 0.0
            # 策略3：对归一化层使用专门的权重衰减值
            if isinstance(module, norm_module_types):
                hyperparams["weight_decay"] = weight_decay_norm
            # 策略4：对嵌入层使用专门的权重衰减值
            if isinstance(module, torch.nn.Embedding):
                hyperparams["weight_decay"] = weight_decay_embed
            # 将参数及其定制的超参数添加到参数组列表中
            params.append({"params": [value], **hyperparams})
            
    def maybe_add_full_model_gradient_clipping(args, optim):
        """
        为优化器添加全模型梯度裁剪功能（当配置启用时）
        参数:
            args: 配置参数，包含梯度裁剪相关设置
            optim: 原始优化器类（如torch.optim.AdamW）

        返回:
            如果启用了全模型梯度裁剪，返回一个包装后的优化器类；否则返回原始优化器类
        """
        # detectron2 doesn't have full model gradient clipping now
        clip_norm_val = args.clip_gradients_value  # 0.01
        enable = (
            args.clip_gradients_enabled  # 梯度剪裁使能
            and args.clip_gradients_type == "full_model"  # 执行全模型梯度剪裁
            and clip_norm_val > 0.0  # 梯度剪裁标准值大于0
        )

        # 定义一个包装类，继承自原始优化器类，重写step方法以添加梯度裁剪逻辑
        class FullModelGradientClippingOptimizer(optim):
            def step(self, closure=None):
                """重写优化器的step方法，在参数更新前执行全模型梯度裁剪"""
                # 获取所有需要优化的参数（将所有参数组中的参数展平为一个迭代器）
                all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                # 对所有参数的梯度执行L2范数裁剪（torch.nn.utils.clip_grad_norm_）
                # 确保梯度的总L2范数不超过clip_norm_val
                torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                # 调用原始优化器的step方法完成参数更新
                super().step(closure=closure)
        return FullModelGradientClippingOptimizer if enable else optim

    # 应用梯度剪裁
    optimizer_type = args.type
    if optimizer_type == "SGD":
        optimizer = maybe_add_full_model_gradient_clipping(args, torch.optim.SGD)(
            params, args.lr, momentum=args.momentum)
    elif optimizer_type == "AdamW":
        optimizer = maybe_add_full_model_gradient_clipping(args, torch.optim.AdamW)(
            params, args.lr)
    else:
        raise NotImplementedError(f"no optimizer type {optimizer_type}")
    return optimizer

