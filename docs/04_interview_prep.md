# ML / CV Interview Preparation

Interview questions organized by topic for ML Engineering / Computer Vision roles.

---

## Transfer Learning

**Q: What is the difference between feature extraction and fine-tuning?**
- Feature extraction: backbone frozen, only head trains. Fast, less data needed.
- Fine-tuning: backbone (partially or fully) unfrozen, trains with small LR.
- Rule of thumb: small dataset + similar domain → feature extraction. Large dataset / different domain → fine-tuning.

**Q: Why use warmup in LR scheduling?**
- At the start, weights are mismatched (especially the new head). A large LR would cause instability.
- Warmup slowly increases LR from near-zero to target LR over first N steps.
- Common: linear warmup for 5-10% of total steps, then cosine or linear decay.

**Q: What is catastrophic forgetting?**
- When a neural network trained on a new task completely forgets how to do the old task.
- Caused by overwriting pretrained weights with large gradient updates.
- Solutions: smaller LR, feature extraction, EWC (Elastic Weight Consolidation), LoRA.

**Q: When would you use LoRA over full fine-tuning?**
- GPU memory is limited (LoRA only trains ~1% of params, uses 10x less memory)
- Need to maintain multiple fine-tuned versions (store only adapters, not full models)
- Large model + small dataset (LoRA's regularization helps prevent overfitting)

**Q: Explain Knowledge Distillation loss formula.**
- `Loss = α * CE(student, hard_labels) + (1-α) * KL(softmax(s/T), softmax(t/T)) * T²`
- α = balance factor (0.5 typical), T = temperature (4 typical)
- T² term: compensates for gradient scaling — KL gradients are divided by T² due to softmax
- Temperature T: higher → softer distribution → more information from teacher

---

## Object Detection

**Q: What is NMS and when is it NOT needed?**
- NMS: removes duplicate detections for the same object by suppressing boxes with IoU > threshold.
- NOT needed for DETR-family models (RT-DETR, DINO) — they use bipartite matching, each query → one object.

**Q: What is the difference between mAP@50 and mAP@50:95?**
- mAP@50: IoU threshold of 0.5 (PASCAL VOC). A box is TP if it overlaps GT by 50%.
- mAP@50:95: Average of mAP at IoU thresholds 0.50, 0.55, 0.60 ... 0.95 (COCO). Much stricter.
- mAP@50:95 is the standard for modern papers.

**Q: Why does Focal Loss help in object detection?**
- In detection, most candidate boxes are easy negatives (background).
- Standard CE loss → model focuses on easy negatives, ignores hard positives.
- Focal Loss: `FL = -(1-pt)^γ * log(pt)` — down-weights easy examples (pt close to 1).
- γ=2 means a well-classified example contributes 1000x less loss than a misclassified one.

**Q: Explain Hungarian matching in DETR.**
- DETR predicts N=300 boxes but image may have M=5 objects.
- We need to match 5 predictions to 5 GT boxes (1-to-1, no duplicates).
- Hungarian algorithm finds assignment that minimizes total cost (classification + box loss).
- Remaining 295 predictions → matched to "no object" class.

**Q: What is Feature Pyramid Network (FPN)?**
- Combines features from multiple backbone scales (P3, P4, P5) for multi-scale detection.
- Small objects detected at high-resolution (early) layers.
- Large objects detected at low-resolution (late) layers.
- Used in Faster R-CNN, RetinaNet, YOLO.

---

## Deep Learning / PyTorch

**Q: What is the difference between `model.eval()` and `torch.no_grad()`?**
- `model.eval()`: Changes behaviour of BatchNorm (use running stats) and Dropout (disabled). Does NOT stop gradient computation.
- `torch.no_grad()`: Stops gradient computation and saves memory. Does NOT change model behaviour.
- For inference: use BOTH.

**Q: What is mixed precision training? When to use fp16 vs bf16?**
- Mixed precision: compute forward/backward in fp16/bf16 (faster, less memory), update weights in fp32.
- fp16: range ±65504, needs `GradScaler` to prevent underflow. Use on older GPUs (V100, T4).
- bf16: same range as fp32, no `GradScaler` needed. Use on A100, H100, newer GPUs.
- Speedup: 2-3x on CUDA with tensor cores.

**Q: What is gradient accumulation?**
- Simulates larger batch size when GPU memory is limited.
- Instead of one update per batch, accumulate gradients over N batches, then update.
- `effective_batch_size = per_device_batch × gradient_accumulation_steps × num_gpus`

**Q: Explain the difference between `DataLoader`'s `collate_fn` and HF's `data_collator`.**
- Both serve the same purpose: take a list of dataset items and combine into a batch.
- PyTorch `DataLoader`: pass as `collate_fn=my_fn`
- HF Trainer: pass as `data_collator=my_fn`
- `DefaultDataCollator`: just stacks tensors. Requires all items same shape.
- Custom collator: needed for variable-size inputs (object detection, variable-length sequences).

**Q: What is `requires_grad` vs `torch.no_grad()`?**
- `requires_grad=False`: permanently disables gradient tracking for that parameter (feature extraction).
- `torch.no_grad()`: context manager that temporarily disables gradient tracking (inference, eval).

---

## Data Augmentation

**Q: What augmentations are safe for object detection?**
- Safe: horizontal flip, color jitter, brightness/contrast, normalize, crop (with box adjustment)
- Unsafe without adjustment: resize (must scale boxes), rotate (must rotate boxes)
- Best library: `albumentations` (handles boxes/masks automatically)

**Q: What is MixUp / CutMix?**
- MixUp: blend two images and labels linearly. `x = λ*x1 + (1-λ)*x2`, `y = λ*y1 + (1-λ)*y2`
- CutMix: cut a patch from one image and paste into another. Adjust labels by area ratio.
- Both act as regularization and improve robustness. Used in ViT training.

**Q: What is test-time augmentation (TTA)?**
- Run inference on multiple augmented versions of same image (flip, scale), then average predictions.
- Free accuracy boost at inference time (cost: slower).

---

## Computer Vision Architecture

**Q: What are the main advantages of ViT over CNN?**
- Global attention from layer 1 (CNN builds up global context slowly through layers)
- Better scaling — performance keeps improving with more data/params
- Disadvantages: needs more data, less inductive bias, slower on small datasets

**Q: What is positional encoding in ViT?**
- ViT splits image into N patches. Transformer has no inherent sense of position.
- Learned or sinusoidal positional embeddings added to patch embeddings.
- Without it, ViT treats patches as a bag-of-patches (order doesn't matter).

**Q: What is the difference between semantic, instance, and panoptic segmentation?**
- Semantic: every pixel gets a class label. No instance distinction. `cat cat cat dog dog`
- Instance: each object gets unique ID. Different cats have different masks. `cat1 cat2 dog1`
- Panoptic: combines both. Stuff (sky, road) = semantic. Things (cat, dog) = instance.

---

## Model Deployment

**Q: What is ONNX and why export to it?**
- Open Neural Network Exchange: framework-agnostic model format.
- Allows running PyTorch models in TensorRT, OpenVINO, ONNX Runtime without Python.
- Typically 1.5-3x inference speedup with ONNX Runtime vs PyTorch.

**Q: What is quantization?**
- Reduce model weight precision: FP32 → INT8 or FP16.
- Post-training quantization (PTQ): quantize after training, fast but some accuracy loss.
- Quantization-aware training (QAT): simulate quantization during training, better accuracy.
- Speedup: INT8 is 4x smaller than FP32, 2-4x faster on CPU.

---

## Resources

- Deep Learning book (free): https://www.deeplearningbook.org/
- CS231n course: https://cs231n.github.io/
- Andrej Karpathy's Neural Networks Zero-to-Hero: https://karpathy.ai/zero-to-one.html
- PyTorch tutorials: https://pytorch.org/tutorials/
- Papers With Code (SOTA models): https://paperswithcode.com/
- Made With ML (MLOps): https://madewithml.com/
- Lilian Weng's blog: https://lilianweng.github.io/
