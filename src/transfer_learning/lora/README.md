# LoRA (Low-Rank Adaptation)

## Overview
LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method that adds trainable low-rank matrices to pre-trained model weights. It dramatically reduces the number of trainable parameters while maintaining performance.

## When to Use
- Fine-tuning large language models (LLMs)
- Limited GPU memory
- Want to fine-tune multiple tasks efficiently
- Need to store multiple fine-tuned versions
- Working with transformer models

## How It Works
1. Freeze pre-trained model weights
2. Add low-rank decomposition matrices (A and B) to specific layers
3. Train only the low-rank matrices
4. Combine with original weights during inference
5. Can merge LoRA weights back into base model

## Key Concepts
- **Rank (r)**: Controls the number of trainable parameters (typically 4-64)
- **Alpha (α)**: Scaling factor for LoRA weights
- **Target modules**: Usually attention matrices (Q, K, V, O)
- **Parameter efficiency**: Reduces trainable params by 1000x-10000x

## Advantages
- Extremely parameter-efficient (0.1%-1% of original params)
- Low memory footprint
- Fast training
- Easy to switch between different LoRA adapters
- No inference latency overhead when merged
- Great for multi-task learning

## Disadvantages
- Slight performance gap vs full fine-tuning (usually small)
- Requires LoRA-compatible implementations
- Hyperparameter tuning (rank, alpha) needed
- Mostly designed for transformer architectures

## Example Code Structure
```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

# Load base model
model = AutoModelForCausalLM.from_pretrained("base-model")

# Configure LoRA
lora_config = LoraConfig(
    r=16,  # rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

## Resources
- LoRA Paper: https://arxiv.org/abs/2106.09685
- Hugging Face PEFT: https://huggingface.co/docs/peft
- PEFT Documentation: https://huggingface.co/docs/peft/task_guides/lora
