# HiConSpot

**HiConSpot: Hierarchical Connectivity‑Aware Multimodal Parsing of Civil and Structural Primitives in Substation CAD Drawings**

Qianyun Song<sup>a,c</sup>, Han Lin<sup>a,b</sup>, Yulong Zhang<sup>a,b,*</sup>, Song Zheng<sup>a,b</sup>, Yuhang Zhu<sup>a,b</sup>, Zhezhuang Xu<sup>a,b</sup>

<sup>a</sup> College of Electrical Engineering and Automation, Fuzhou University, Fuzhou 350108, Fujian, China<sup>b</sup> Key Laboratory of Industrial Automation Control Technology and Information Processing, Education Department of Fujian Province, Fuzhou 350000, Fujian, China<sup>c</sup> State Grid Fujian Electric Power Co., Ltd. Construction Company, Fuzhou 350000, Fujian, China

<sup>*</sup> Corresponding author: zhangyl@fzu.edu.cn

Code repository: [https://github.com/20HanLin24/HiConSpot](https://github.com/20HanLin24/HiConSpot)

## Overview

HiConSpot 是面向变电站 CAD 图纸的双流查询式框架，实现图元级语义分割与全景符号检测。模型不依赖图层、颜色等设计属性，依靠几何连通关系与光栅形态完成解析。

> 
> 本仓库仅开源**模型、预处理、损失代码**。国网自有变电站数据集属于涉密工业数据，**不对外开源**；训练权重因数据保密不予提供。读者可在公共数据集上复现整套算法流程。

## 1. Environment

### Hardware

Single NVIDIA RTX PRO 6000，训练使用 FP16 混合精度。

### Main Dependencies

- Python: 3.10
- CUDA: 12.9
- PyTorch: 2.x（匹配本机 CUDA 版本）
- Other：`numpy, scipy, pillow, yacs, addict, timm, torch‑scatter, tensorboard`
- Optional：`spconv, flash‑attn, svgpathtools`

### Quick install

```
conda create -n hiconspot python=3.10 -y
conda activate hiconspot

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
pip install numpy scipy pillow yacs addict timm tensorboard
# torch‑scatter 请根据你的torch/cuda版本从pyg官网下载对应wheel安装
```

## 2. Datasets

### FloorPlanCAD（公开基准数据集）

Project page: [https://floorplancad.github.io/](https://floorplancad.github.io/)

> 
> 论文使用子集划分：1500 train / 100 val / 100 test patches，遵循原数据集协议。

### 自有变电站数据集

> 
> ⚠️ Confidential industrial data from State Grid Fujian. **Not publicly available**.
> 原始 DXF 图纸、标注样本不会上传到此仓库。

## 3. Reference Code & Acknowledgements

本项目部分模块基于下列开源项目修改，原文件内保留原始版权声明：

1. **SymPoint**：评估器、数据增强、预处理逻辑
[https://github.com/nicehuster/SymPoint](https://github.com/nicehuster/SymPoint)
2. **CADTransformer**：HRNet‑W48 光栅主干网络实现
[https://github.com/VITA](https://github.com/VITA)‑Group/CADTransformer
3. **DETR**：匈牙利匹配、Set‑based 损失实现
[https://github.com/facebookresearch/detr](https://github.com/facebookresearch/detr)
4. **numpy‑hilbert‑curve**：Hilbert 曲线序列化工具
5. **MMCV**：半精度、日志、分布式工具参考

> 
> 架构思路参考：Point Transformer V3、OneFormer3D、SymPointV2、CADSpotting，未直接复制代码。

## 4. Citation

```
@article{song2026hiconspot,
  title   = {HiConSpot: Hierarchical Connectivity‑Aware Multimodal Parsing of Civil and Structural Primitives in Substation CAD Drawings},
  author  = {Song, Qianyun and Lin, Han and Zhang, Yulong and Zheng, Song and Zhu, Yuhang and Xu, Zhezhuang},
  journal = {TODO: journal name after acceptance},
  year    = {2026},
  note    = {Code: https://github.com/20HanLin24/HiConSpot}
}
```

## 5. License

- 本项目原创代码采用 **Apache‑2.0 License**，详见 `LICENSE` 文件。
- 改写的第三方模块保留其原始开源协议，请查看对应文件头部版权注释。

## Contact

- Corresponding author: zhangyl@fzu.edu.cn
- Code: Han Lin
