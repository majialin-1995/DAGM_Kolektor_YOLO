# DAGM & Kolektor 缺陷检测基准 — 基于 YOLOv5 / YOLOv8n 的 2+2 创新改进

本项目是一个可直接运行的工业缺陷检测基准，用于在 **DAGM2007** 和 **KolektorSDD/SDD2** 数据集上评估 YOLO 系列模型（YOLOv5n 与 YOLOv8n）。  
其中包含 **基线模型** 与 **四个创新点（2+2 分组）**，并配有完整的数据准备、训练与消融实验脚本。

---

## 🧩 项目功能概述

1. 将 DAGM / Kolektor 原始数据集转换为 **YOLO 检测格式**（images/、labels/、data.yaml）。  
2. 基于 **Ultralytics YOLOv8n** 运行基线与改进版训练。  
3. 可选支持 **YOLOv5n** 作为对照实验。  
4. 自动执行 **2+2 创新点的消融实验**，并保存结果到 `runs_*` 目录中。

---

## 🚀 四个创新点（2+2 分组）

### **A组：结构创新（Architecture）**
1. **A1 — P2 小目标检测头**：  
   在原 YOLOv8n 的 P3–P5 检测头之外新增 **P2 (stride=4)**，显著提升微小缺陷检测能力。  
2. **A2 — C2f‑SElite 颈部注意力模块**：  
   将部分 C2f 模块替换为轻量化 **SELite 通道注意力**，增强特征表达，参数量几乎不增加。  

### **B组：数据与优化（Data / Optimization）**
3. **B1 — SRTS（频域残差纹理抑制）**：  
   一种基于频谱残差显著性图的训练时数据增强方法，自动抑制周期性背景纹理，突出潜在缺陷区域。  
4. **B2 — NPR（负样本补放，Negative Patch Replay）**：  
   从背景中挖掘“难负样本”小块并随机混贴回原图，降低误检率、提升模型鲁棒性。  

> 消融路径：**B0 (基线)** → **+A1** → **+A1+A2** → **+A1+A2+B1** → **+A1+A2+B1+B2**

---

## 🧠 环境安装

```bash
python -m venv .venv
source .venv/bin/activate          # Windows 下为 .venv\Scripts\activate
pip install -U pip
pip install -U -r requirements.txt
```

依赖包括：`ultralytics`、`opencv-python`、`numpy`、`scipy`、`scikit-image`、`matplotlib`

---

## 📊 数据集准备

### 自动下载原始数据

```bash
python datasets/download_datasets.py --dataset all --output data/raw
```

- `data/raw/dagm2007/` 会包含 `Class1`…`Class10` 等 DAGM 原始结构。
- `data/raw/kolektorsdd/` 与 `data/raw/kolektorsdd2/` 会统一整理出 `images/` 与 `gt/` 目录。

如需单独下载，可通过 `--dataset dagm2007`、`--dataset kolektorsdd` 或 `--dataset kolektorsdd2` 指定。

### DAGM2007
```bash
python datasets/prepare_dagm.py --root data/raw/dagm2007 --out yolo_dagm
```
- 自动搜索类别文件夹（Class1..ClassN）及其中的 Train/Test/Images 等子目录。  
- 若存在掩码文件（如 `*_mask.png`），会自动转为 YOLO 格式的边界框。  

### KolektorSDD/SDD2
```bash
python datasets/prepare_kolektor.py --root data/raw/kolektorsdd --out yolo_kolektor
```
- 若需使用 KolektorSDD2，可将 `--root` 指向 `data/raw/kolektorsdd2`。
- 需包含 `images/` 与 `gt/`（或 `masks/`）文件夹。  
- 掩码将自动转换为检测框并写入标签文件。  

输出结果形如：
```
yolo_dagm/
 ├─ images/train/xxx.png
 ├─ labels/train/xxx.txt
 └─ data.yaml
```

---

## ⚙️ 快速开始（YOLOv8n 训练与消融）

**在 DAGM 上训练：**
```bash
python train_many.py --data yolo_dagm/data.yaml --project runs_dagm --epochs 150 --imgsz 640
```

**在 Kolektor 上训练：**
```bash
python train_many.py --data yolo_kolektor/data.yaml --project runs_kolektor --epochs 150 --imgsz 640
```

运行流程：
1. **B0**：YOLOv8n 官方基线  
2. **A1**：加入 P2 检测头  
3. **A1+A2**：加入 P2 + C2f-SElite  
4. **A1+A2+B1**：加入 SRTS 增强  
5. **A1+A2+B1+B2**：加入 SRTS + NPR  

结果保存在 `runs_*` 文件夹中，可比较 mAP@0.5 / mAP@0.5:0.95 / Recall / Params / FPS 等指标。

---

## 🧩 可选：YOLOv5n 基线

```bash
git clone https://github.com/ultralytics/yolov5
python tools/run_yolov5.py --data yolo_dagm/data.yaml --project runs_dagm
```

---

## 💡 提示与建议

- **A1** 适合检测微小目标；**B1+B2** 可在背景复杂、伪纹理多的情况下稳定提升效果。  
- **A2** 模块极轻量，适合在嵌入式环境部署。  
- 若目录结构与脚本假设不同，可微调 `datasets/prepare_*.py`。  
- 在 `train_many.py` 中可通过 `--device` 指定 GPU/CPU（默认 `0`）。  

---

## 📚 致谢与引用

- YOLOv5 / YOLOv8 由 Ultralytics 提供。  
- SRTS 灵感来源于 Hou & Zhang (CVPR 2007) 的 Spectral Residual Saliency 方法。  
- 数据集：DAGM2007、KolektorSDD/SDD2。  

---

## 📄 许可说明

该项目仅供科研与教学使用。禁止商业用途。
