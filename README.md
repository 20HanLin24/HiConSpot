# HiConSpot

**HiConSpot: Hierarchical Connectivity-Aware Multimodal Parsing of Civil and Structural Primitives in Substation CAD Drawings**

Qianyun Song<sup>a,c</sup>, Han Lin<sup>a,b</sup>, Yulong Zhang<sup>a,b,\*</sup>, Song Zheng<sup>a,b</sup>, Yuhang Zhu<sup>a,b</sup>, Zhezhuang Xu<sup>a,b</sup>

<sup>a</sup> College of Electrical Engineering and Automation, Fuzhou University, Fuzhou 350108, Fujian, China
<sup>b</sup> Key Laboratory of Industrial Automation Control Technology and Information Processing, Education Department of Fujian Province, Fuzhou 350000, Fujian, China
<sup>c</sup> State Grid Fujian Electric Power Co., Ltd. Construction Company, Fuzhou 350000, Fujian, China

<sup>\*</sup> Corresponding author: zhangyl@fzu.edu.cn

Code repository: https://github.com/20HanLin24/HiConSpot

---

## Overview

HiConSpot is a dual-stream, query-based framework for substation CAD drawings that performs primitive-level semantic segmentation and panoptic symbol spotting. The model does not rely on designer-dependent attributes such as drawing layers or line colors; instead, it parses civil and structural primitives from geometric connectivity together with raster morphology.

This repository releases the **model, preprocessing, loss, and training/evaluation code** only. The proprietary State Grid substation dataset is confidential industrial data and is **not released**; trained weights are withheld for the same reason. The complete algorithmic pipeline can be reproduced on public or self-collected CAD datasets.

---

## 1. Environment

### Hardware

A single NVIDIA RTX PRO 6000 GPU; all training runs use FP16 mixed precision.

### Main dependencies

- Python: 3.10
- CUDA: 12.9
- PyTorch: 2.x (build matched to the local CUDA version)
- Other: `numpy, scipy, pillow, yacs, addict, timm, torch-scatter, tensorboard`
- Optional: `spconv, flash-attn, svgpathtools`

### Quick install

```bash
conda create -n hiconspot python=3.10 -y
conda activate hiconspot

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
pip install numpy scipy pillow yacs addict timm tensorboard
# For torch-scatter, install the wheel matching your torch/CUDA version from the PyG wheel index
```

---

## 2. Datasets

### FloorPlanCAD (public benchmark)

Project page: https://floorplancad.github.io/

The paper uses a reduced-budget subset: 1,500 training / 100 validation / 100 test patches, following the original dataset protocol.

### Proprietary substation dataset

> Confidential industrial data from State Grid Fujian. **Not publicly available.**

Raw DXF drawings and annotated samples are not uploaded to this repository.

---

## 3. Reference Code and Acknowledgements

Parts of this project are adapted from the open-source projects listed below; original copyright headers are preserved in the corresponding files.

1. **SymPoint** — PTv3 point-stream implementation, evaluator, data augmentation, preprocessing logic
   https://github.com/nicehuster/SymPoint
2. **CADTransformer** — HRNet-W48 raster backbone implementation
   https://github.com/VITA-Group/CADTransformer
3. **DETR** — Hungarian matching and set-based loss implementations
   https://github.com/facebookresearch/detr
4. **numpy-hilbert-curve** — Hilbert curve serialization utilities
5. **MMCV** — reference for mixed-precision, logging, and distributed utilities

> Architectural inspiration: Point Transformer V3, OneFormer3D, SymPointV2, and CADSpotting. No code was directly copied from these works.

---

## 4. Citation

```bibtex
@article{song2026hiconspot,
  title   = {HiConSpot: Hierarchical Connectivity-Aware Multimodal Parsing of
             Civil and Structural Primitives in Substation CAD Drawings},
  author  = {Qianyun Song, Han Lin, Yulong Zhang,
             Song Zheng, Yuhang Zhu, Zhezhuang Xu},
  journal = {TODO: journal name after acceptance},
  year    = {2026},
  note    = {Code: https://github.com/20HanLin24/HiConSpot}
}
```

---

## 5. License

- Original code specific to this project is released under the **Apache-2.0 License**; see the `LICENSE` file.
- Adapted third-party modules retain their upstream licenses. Refer to the copyright header at the top of each file.

---

## Contact

- Corresponding author: zhangyl@fzu.edu.cn
- Code implementation: Han Lin
