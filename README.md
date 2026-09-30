# PViT — Petrographic Vision Transformer

**A Modified Vision Transformer Architecture with Scratch-Learning Capabilities for Plutonic Rock Classification**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.7+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/Dataset-10.5281%2Fzenodo.18613714-blue)](https://doi.org/10.5281/zenodo.18613714)

**Petrographic Vision Transformer (PViT)** is a family of Vision Transformer models that
automatically classify **15 plutonic rock types** from petrographic thin-section images,
trained entirely **from scratch** — without ImageNet pre-training.

<img width="697" alt="PViT overall architecture" src="https://github.com/user-attachments/assets/df158847-d911-40ae-93a2-3833557eb26e" />

## Highlights

- **98.34 % accuracy** on 15-class plutonic rock classification, trained from scratch.
- **Converges in ~12 epochs**, against ~20 epochs for the ViT, DaViT and EfficientNetV2
  baselines — roughly 40 % fewer epochs to reach peak performance.
- **PViT-Base beats ViT-Base by 0.62 pp** and matches DaViT-Large and EfficientNetV2-L
  while being the smallest model in the family.
- **No pre-training required**, which matters for petrography where large labelled
  thin-section corpora do not exist.

## Architectural modifications

Two changes are introduced over the standard ViT encoder:

- **new GELU activation** — a sigmoid-based approximation of the Gaussian Error Linear Unit
  used in the position-wise feed-forward network (FFN). It is cheaper to evaluate than the
  exact erf formulation while preserving the smooth gradient flow needed for the subtle
  optical textures in thin sections.
- **Fused QKV projection** — the Query, Key, and Value projections in multi-head
  self-attention are merged into a single joint linear transformation, cutting memory
  traffic and improving throughput per step.

The repository provides code for **training**, **evaluation**, and **attention
visualization** on plane-polarized light (PPL) and cross-polarized light (XPL)
thin-section images.

---

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.7.0 or higher
- CUDA 10.2 or higher (for GPU training)

### Tested configuration

| Component | Version |
|---|---|
| OS | Ubuntu 18.04.6 LTS |
| Python | 3.8.11 |
| PyTorch | 1.8.2 |
| CUDA | 10.2 |
| GPU | NVIDIA RTX series |

### Setup

```bash
git clone https://github.com/Sittiporn-GT/PViT-Petrographic-Vision-Transformer.git
cd PViT-Petrographic-Vision-Transformer

conda create -n pvit python=3.8 -y
conda activate pvit
```

Install PyTorch and CUDA first, following the
[official PyTorch instructions](https://pytorch.org/get-started/locally/), then install the
remaining dependencies:

```bash
pip install -r requirements.txt
```

---

## Dataset

**2,886 thin-section images** covering **15 plutonic rock types**, captured under both
plane-polarized light (PPL) and cross-polarized light (XPL).

| Split | Proportion | Images |
|---|---|---|
| Train | 70 % | 2,020 |
| Validation | 15 % | 433 |
| Test | 15 % | 433 |

### Rock classes

Anorthosite · Diorite · Dunite · Foidolite · Gabbro · Granite · Granodiorite · Greisen ·
Harzburgite · Hornblendite · Lherzolite · Monzonite · Norite · Orthopyroxenite · Syenite

### Directory structure

Each sample consists of a paired PPL and XPL image captured from the same field of view:

```
dataset/
├── train/
│   ├── Granite/
│   │   ├── 01_granite_PPL.jpg
│   │   ├── 01_granite_XPL.jpg
│   │   ├── 02_granite_PPL.jpg
│   │   ├── 02_granite_XPL.jpg
│   │   └── ...
│   ├── Gabbro/
│   ├── Diorite/
│   ├── Dunite/
│   └── ...
├── val/
│   ├── Granite/
│   ├── Gabbro/
│   └── ...
└── test/
    ├── Granite/
    ├── Gabbro/
    └── ...
```

**Naming rule:** a PPL image and its XPL counterpart must share the same numeric prefix and
differ only by the `_PPL` / `_XPL` suffix, so the loader can pair them automatically.

### Download

Due to GitHub size limits, the dataset is hosted on Zenodo:

**Demo / test set:** [10.5281/zenodo.18613714](https://doi.org/10.5281/zenodo.18613714)

---

## Training

Train PViT from scratch:

```bash
python train.py \
    --exp-name pvit_base \
    --data-dir data/dataset \
    --model pvit_base \
    --img-size 512 \
    --patch-size 16 \
    --batch-size 16 \
    --epochs 20 \
    --optimizer adamw \
    --weight-decay 0.001 \
    --seed 42
```

Logs, loss curves, and validation accuracy are written to `checkpoints/<exp-name>/`.
The best-performing checkpoint is saved as `model_best.pt`.

PViT typically reaches its peak validation accuracy at around **epoch 12**; training beyond
epoch 20 gives no further improvement.

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--exp-name` | — | Experiment name, used as the output folder |
| `--data-dir` | `data/dataset` | Path to the dataset root |
| `--model` | `pvit_base` | One of `pvit_base`, `pvit_large`, `pvit_huge` |
| `--img-size` | 512 | Input resolution |
| `--patch-size` | 16 | Patch size |
| `--batch-size` | 16 | Training batch size |
| `--epochs` | 20 | Number of training epochs |
| `--optimizer` | `adamw` | Optimizer |
| `--weight-decay` | 0.001 | Weight decay |
| `--seed` | 42 | Random seed for reproducibility |

### Training configuration

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Weight decay | 0.001 |
| Input resolution | 512 × 512 |
| Patch size | 16 × 16 |
| Initialization | From scratch (no pre-training) |
| Convergence | ~12 epochs |

---

## Evaluation

Evaluate a trained checkpoint on the test set:

```bash
python evaluate.py \
    --exp-name pvit_base \
    --checkpoint checkpoints/pvit_base/model_best.pt \
    --data-dir data/dataset/test
```

The script reports **accuracy**, **precision**, **recall** and **F1-score**, together with a
per-class confusion matrix over the 15 rock types.

---

## Attention visualization

Visualize self-attention maps and Grad-CAM interpretations:

```bash
python visualize_attention.py \
    --exp-name pvit_base \
    --checkpoint checkpoints/pvit_base/model_best.pt \
    --data-dir data/dataset/test \
    --output attention_results.png
```

This overlays attention heatmaps on the thin-section images, highlighting the petrographic
features that drive the model's decision — **twinning**, **extinction patterns**, and
**grain boundaries**.

---

## Results

All models below were trained **from scratch** on the same plutonic rock dataset and
evaluated on the held-out test split.

### Petrographic Vision Transformer (PViT)

| Model | Patch size | Training | Accuracy | Weights |
|---|---|---|---|---|
| **PViT-Base** | 16 × 16 | Scratch | **98.34 %** | [download](https://doi.org/10.5281/zenodo.23047502) |
| PViT-Base | 32 × 32 | Scratch | 95.03 % | [download](https://doi.org/10.5281/zenodo.23047595) |
| PViT-Large | 16 × 16 | Scratch | 97.52 % | [download](https://doi.org/10.5281/zenodo.23047658) |
| PViT-Large | 32 × 32 | Scratch | 91.71 % | [download](https://doi.org/10.5281/zenodo.23047719) |
| PViT-Huge | 16 × 16 | Scratch |  % | [download]() |
| PViT-Huge | 32 × 32 | Scratch | 93.17 % | [download](https://doi.org/10.5281/zenodo.23047784) |

Place the downloaded `.pt` files in `checkpoints/` before running evaluation.

### Comparison with baselines

| Model | Accuracy |
|---|---|
| **PViT-Base (ours)** | **98.34 %** |
| PViT-Large (ours) | 97.52 % |
| PViT-Huge (ours) | 96.07 % |
| ViT-Base | 97.72 % |
| ViT-Large | 96.38 % |
| ViT-Huge | 96.07 % |
| DaViT-Base | 96.27 % |
| DaViT-Large | 98.34 % |
| EfficientNetV2-M | 98.14 % |
| EfficientNetV2-L | 98.34 % |

PViT-Base matches the best-performing baselines (DaViT-Large, EfficientNetV2-L) while being
substantially smaller, and converges in roughly **12 epochs** compared with **~20 epochs**
for the baselines.

Additional architectures evaluated in the manuscript include Swin Transformer, DeiT, T2T,
TNT, Trans-SedNet, Rock-ViT, MobileNetV2, ResNet50, U-Net, DeepLabV3+ and CoCa-Base.

---

## Reproducibility

- All experiments reported in the manuscript were run with the configuration files provided
  in `configs/`.
- Random seeds are fixed via `--seed` to ensure deterministic behaviour.
- The demo dataset on Zenodo allows verification of code execution and output consistency
  without downloading the full corpus.

---

## Citation

If you find this repository useful, please cite:

```bibtex
@article{kongsukho2025pvit,
  title   = {A modified vision transformer architecture with scratch learning
             capabilities for plutonic rock classification},
  author  = {Kongsukho, Sittiporn and Maneerat, Warunee and Owada, Narihiro and
             Adachi, Tsuyoshi and Vateekul, Peerapon and Sutthirat, Chakkaphan},
  journal = {},
  year    = {2025},
  doi     = {}
}
```

Dataset:

```bibtex
@dataset{kongsukho2025pvit_data,
  title     = {Plutonic Rock Thin-Section Dataset (Demo)},
  author    = {Kongsukho, Sittiporn},
  year      = {2025},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.18613714}
}
```

---

## Related work

- [OTA — Optical Thin-Section Analysis](https://github.com/Sittiporn-GT/OTA-Optical-Thin-Section-Analysis)
  — dual-modal segmentation of individual mineral grains, a complementary task to the
  rock-type classification performed here.

---

## Acknowledgements

This work was supported by the **Development and Promotion of Science and Technology
Talented Project (DPST)**, the **Institute for the Promotion of Teaching Science and
Technology (IPST)**, and the **90th Anniversary of Chulalongkorn University Scholarship**
under the Ratchadapisek Somphot Endowment Fund.

**Affiliations:** Department of Geology, Faculty of Science, Chulalongkorn University ·
Department of Mineral Resources, Thailand · Faculty of International Resource Sciences,
Akita University, Japan · Department of Computer Engineering, Faculty of Engineering,
Chulalongkorn University.

---

## License

Released under the MIT License. See [LICENSE](LICENSE) for details.

## Contact

Sittiporn Kongsukho — please open an
[issue](https://github.com/Sittiporn-GT/PViT-Petrographic-Vision-Transformer/issues)
for questions or bug reports.
