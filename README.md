# CV Learning

A focused study repository for **Transfer Learning in Computer Vision** using HuggingFace Transformers as the primary framework.

## Goal

Learn all industry-standard transfer learning techniques in CV with working code examples, from feature extraction to zero-shot inference. Doubles as interview preparation material.

## Project Structure

```
cv-learning/
├── src/
│   ├── transfer_learning/
│   │   ├── README.md                        # Technique comparison + decision guide
│   │   ├── feature_extraction/              # Type 1: Frozen backbone
│   │   │   ├── mobilenetv3_small_100_lamb_in1k.py  ← HF Trainer example
│   │   │   └── README.md
│   │   ├── fine_tuning/                     # Type 2: Full / partial unfreeze
│   │   │   ├── RTDETR_V2_with_transformer.py       ← HF Trainer (object detection)
│   │   │   ├── RTDETR_V2_pure_pytorch.py           ← Pure PyTorch reference
│   │   │   └── README.md
│   │   ├── lora/                            # Type 3: Parameter-efficient (PEFT)
│   │   │   ├── vit_lora_image_classification.py    ← ViT + LoRA via PEFT
│   │   │   └── README.md
│   │   ├── knowledge_distillation/          # Type 4: Teacher → Student compression
│   │   │   ├── mobilenet_kd_from_vit.py            ← Custom KD Trainer
│   │   │   └── README.md
│   │   └── zero_shot/                       # Type 5: CLIP, no labels needed
│   │       ├── clip_zero_shot_classification.py    ← CLIP zero-shot eval
│   │       └── README.md
│   ├── utils/
│   │   ├── utils.py          # Device selection, dataset loaders
│   │   ├── config.py         # Config management
│   │   └── data_loader.py
│   └── configs/
│       ├── config.yaml
│       └── dataset.yaml
├── docs/                                    # Interview prep & learning resources
│   ├── README.md                            # Index + must-know topics
│   ├── 01_transfer_learning.md              # All 5 techniques with Q&A
│   ├── 02_huggingface_ecosystem.md          # HF Trainer, datasets, PEFT, evaluate
│   ├── 03_object_detection.md              # OD paradigms, DETR, losses, metrics
│   ├── 04_interview_prep.md                 # 50+ interview Q&A
│   └── 05_datasets_and_benchmarks.md       # Key datasets and leaderboards
├── data/                                    # Datasets (gitignored)
├── models/                                  # Downloaded pretrained models (gitignored)
├── outputs/                                 # Training outputs (gitignored)
├── requirements.txt
└── pyproject.toml
```

## Transfer Learning Techniques Covered

| # | Technique | Task | Model | Script |
|---|---|---|---|---|
| 1 | Feature Extraction | Image Classification | MobileNetV3-Small | `feature_extraction/` |
| 2 | Full Fine-Tuning | Object Detection | RT-DETRv2 | `fine_tuning/` |
| 3 | LoRA / PEFT | Image Classification | ViT-Base + LoRA | `lora/` |
| 4 | Knowledge Distillation | Image Classification | ViT → MobileNetV3 | `knowledge_distillation/` |
| 5 | Zero-Shot (CLIP) | Image Classification | CLIP ViT-B/32 | `zero_shot/` |

## Tech Stack

- **Primary**: HuggingFace `transformers` + `peft` + `datasets` + `evaluate`
- **Secondary**: PyTorch, torchvision
- **Dataset source**: Roboflow (auto-download via API)
- **Models**: Downloaded locally to `models/` from HuggingFace Hub

## Setup

```bash
uv sync
# or
pip install -r requirements.txt
```

### Download Models

```bash
# MobileNetV3 (feature extraction)
huggingface-cli download timm/mobilenetv3_small_100.lamb_in1k --local-dir ./models/mobilenetv3_small_100_lamb_in1k

# ViT-Base (LoRA + KD teacher)
huggingface-cli download google/vit-base-patch16-224 --local-dir ./models/vit_base_patch16_224

# RT-DETRv2 (fine-tuning)
huggingface-cli download PekingU/rtdetr_r50vd --local-dir ./models/rtdetr_r50vd

# CLIP (zero-shot)
huggingface-cli download openai/clip-vit-base-patch32 --local-dir ./models/clip_vit_base_patch32
```

### Run Examples

```bash
# Feature extraction (classification)
uv run python -m src.transfer_learning.feature_extraction.mobilenetv3_small_100_lamb_in1k

# Fine-tuning (object detection)
uv run python -m src.transfer_learning.fine_tuning.RTDETR_V2_with_transformer

# LoRA (PEFT)
uv run python -m src.transfer_learning.lora.vit_lora_image_classification

# Knowledge Distillation
uv run python -m src.transfer_learning.knowledge_distillation.mobilenet_kd_from_vit

# Zero-shot (CLIP)
uv run python -m src.transfer_learning.zero_shot.clip_zero_shot_classification
```

## Interview Prep Docs

See `docs/` folder:
- `docs/04_interview_prep.md` — 50+ Q&A covering TL, OD, PyTorch, augmentation, deployment
- `docs/01_transfer_learning.md` — Deep dive into all 5 techniques
- `docs/03_object_detection.md` — DETR family, mAP, losses, NMS

## License

MIT License