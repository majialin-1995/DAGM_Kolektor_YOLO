# DAGM 缺陷检测基准 — 基于 YOLOv5 / YOLOv8n 的 2+2 创新改进

本项目是一个可直接运行的工业缺陷检测基准，用于在 **DAGM2007** 数据集上评估 YOLO 系列模型（YOLOv5n 与 YOLOv8n）。
其中包含 **基线模型** 与 **四个创新点（2+2 分组）**，并配有完整的数据准备、训练与消融实验脚本。

---

## 🧩 项目功能概述

1. 将 DAGM 原始数据集转换为 **YOLO 检测格式**（images/、labels/、data.yaml）。
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

### 1. 从 Kaggle 获取原始数据

- 访问 <https://www.kaggle.com/datasets/bassam165/dagm-2007-industrial-defect-detection-dataset> 下载 `DAGM2007.zip`。
- 将压缩包解压到任意目录（例如 `data_raw/dagm`）。解压后应包含 `Class1`~`Class10` 等子文件夹，内部含有 Train/Test 图像与对应掩码。

> 提示：如需命令行下载，可使用 `kaggle datasets download bassam165/dagm-2007-industrial-defect-detection-dataset -p data_raw`，随后自行解压。

### 2. 转换为 YOLO 数据格式

```bash
# DAGM2007 → YOLO
python datasets/prepare_dagm.py --root data_raw/DAGM_KaggleUpload --out yolo_dagm
```

- 将自动搜索类别文件夹（Class1..ClassN）及其中的 Train/Test/Images 等子目录。
- 若存在掩码文件（如 `*_mask.png`），会自动转为 YOLO 格式的边界框。

输出结果形如：
```bash
yolo_dagm/
 ├─ images/train/xxx.png
 ├─ labels/train/xxx.txt
 └─ data.yaml
```

---

## ⚙️ 快速开始

### 单次基线训练

```bash
yolo detect train \
  model=yolov8n.pt \
  data=yolo_dagm/data.yaml \
  project=runs_baseline \
  name=B0_yolov8n \
  epochs=150 imgsz=640 batch=4 device=0
```

上面的命令将基于官方 `yolov8n.pt` 权重启动一次 **B0 基线实验**，输出结果存放在 `runs_baseline/B0_yolov8n/`。

---

## 🧪 实验运行指南（2 组 × 4 个子实验）

以下脚本均在仓库根目录执行，可直接复制粘贴到终端。可按需修改 `epochs`、`imgsz`、`batch`、`device` 等参数。

### A 组 · 结构创新（Baseline + P2 + SELite + P2&SELite）

```bash
python - <<'PY'
from train_many import train_once, THIS

common = dict(data="yolo_dagm/data.yaml", project="runs_groupA", imgsz=640, epochs=150, batch=16, device=0)

# 1. 基线（B0）
train_once("yolov8n.pt", name="A0_Baseline", **common)

# 2. 仅加 P2 检测头（A1）
train_once(str(THIS / "models" / "yolov8n_p2_only.yaml"), name="A1_P2only", **common)

# 3. 仅加 C2f-SElite 注意力（A2）
train_once(str(THIS / "models" / "yolov8n_se_only.yaml"), name="A2_SELite", **common)

# 4. 同时加 P2 + C2f-SElite（A1+A2）
train_once(str(THIS / "models" / "yolov8n_p2_se.yaml"), name="A12_P2_SELite", **common)
PY
```

运行结束后，可在 `runs_groupA/` 中找到四个子实验的日志与权重，用以对比结构改进带来的增益。

### B 组 · 数据与优化（A12 基线 + SRTS + NPR + SRTS&NPR）

```bash
python - <<'PY'
from train_many import train_once, THIS
from callbacks.preproc_srts import UltralyticsSRTSCallback
from callbacks.npr_miner import UltralyticsNPRCallback

common = dict(data="yolo_dagm/data.yaml", project="runs_groupB", imgsz=640, epochs=150, batch=16, device=0)
base_model = str(THIS / "models" / "yolov8n_p2_se.yaml")

# 1. A12 结构作为组内基线
train_once(base_model, name="B0_A12_Baseline", **common)

# 2. 仅加 SRTS 频域抑制（B1）
train_once(base_model, name="B1_SRTS", callbacks=[UltralyticsSRTSCallback(prob=0.5, sigma=2.5, alpha=0.65)], **common)

# 3. 仅加 NPR 难负样本回放（B2）
train_once(base_model, name="B2_NPR", callbacks=[UltralyticsNPRCallback(prob=0.5, patch_size=128)], **common)

# 4. 同时加 SRTS + NPR（B1+B2）
train_once(base_model, name="B12_SRTS_NPR", callbacks=[
    UltralyticsSRTSCallback(prob=0.5, sigma=2.5, alpha=0.65),
    UltralyticsNPRCallback(prob=0.5, patch_size=128)
], **common)
PY
```

四个结果会保存在 `runs_groupB/` 中，建议重点比较 mAP、Precision、Recall 与收敛速度。

> 如果希望一次性跑完所有组合，也可以继续使用 `python train_many.py --data yolo_dagm/data.yaml --project runs_dagm --epochs 150 --imgsz 640`，脚本会按照 B0→A1→A1+A2→A1+A2+B1→A1+A2+B1+B2 的顺序自动执行。

---

## 🧩 可选：YOLOv5n 基线

```bash
git clone https://github.com/ultralytics/yolov5
python tools/run_yolov5.py --data yolo_dagm/data.yaml --project runs_dagm
```

---

## 💡 示与建议

- **A1** 适合检测微小目标；**B1+B2** 可在背景复杂、伪纹理多的情况下稳定提升效果。
- **A2** 模块极轻量，适合在嵌入式环境部署。
- 若目录结构与脚本假设不同，可微调 `datasets/prepare_dagm.py`。
- 在 `train_many.py` 中可通过 `--device` 指定 GPU/CPU（默认 `0`）。

---

## 📚 致谢与引用

- YOLOv5 / YOLOv8 由 Ultralytics 提供。
- SRTS 灵感来源于 Hou & Zhang (CVPR 2007) 的 Spectral Residual Saliency 方法。
- 数据集：DAGM2007。

---

## 📄 许可说明

该项目仅供科研与教学使用。禁止商业用途。
