import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, List
import os


class FineTuner:
    """Fine-tuning strategy for pretrained models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        checkpoint_dir: str = './checkpoints'
    ):
        """
        Args:
            model: Pretrained model to fine-tune
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def freeze_layers(self, layer_names: List[str]):
        """Freeze specific layers by name."""
        for name, param in self.model.named_parameters():
            if any(layer in name for layer in layer_names):
                param.requires_grad = False
                print(f'Frozen: {name}')
    
    def unfreeze_layers(self, layer_names: List[str]):
        """Unfreeze specific layers by name."""
        for name, param in self.model.named_parameters():
            if any(layer in name for layer in layer_names):
                param.requires_grad = True
                print(f'Unfrozen: {name}')
    
    def unfreeze_all(self):
        """Unfreeze all layers."""
        for param in self.model.parameters():
            param.requires_grad = True
        print('All layers unfrozen')
    
    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get trainable parameters."""
        return [p for p in self.model.parameters() if p.requires_grad]
    
    def count_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.get_trainable_params())
    
    def count_total_params(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.model.parameters())
    
    def gradual_unfreeze(
        self,
        num_epochs_per_stage: int,
        stages: List[List[str]],
        initial_lr: float = 0.001,
        lr_decay: float = 0.1
    ):
        """
        Gradually unfreeze layers in stages.
        
        Args:
            num_epochs_per_stage: Number of epochs per unfreezing stage
            stages: List of layer name patterns to unfreeze at each stage
            initial_lr: Initial learning rate
            lr_decay: Learning rate decay factor per stage
        """
        from .trainer import Trainer
        
        current_lr = initial_lr
        
        for stage_idx, layer_patterns in enumerate(stages):
            print(f'\n=== Stage {stage_idx + 1} ===')
            print(f'Unfreezing layers: {layer_patterns}')
            
            # Unfreeze layers for this stage
            self.unfreeze_layers(layer_patterns)
            
            # Create optimizer with current LR
            optimizer = torch.optim.Adam(
                self.get_trainable_params(),
                lr=current_lr
            )
            
            # Train for this stage
            trainer = Trainer(
                model=self.model,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                optimizer=optimizer,
                device=self.device,
                checkpoint_dir=self.checkpoint_dir
            )
            
            trainer.train(num_epochs_per_stage)
            
            # Decay learning rate
            current_lr *= lr_decay
            print(f'Learning rate: {current_lr}')
    
    def differential_lr(
        self,
        layer_groups: dict,
        base_lr: float = 0.001
    ):
        """
        Set different learning rates for different layer groups.
        
        Args:
            layer_groups: Dict mapping layer name patterns to LR multipliers
            base_lr: Base learning rate
            
        Example:
            layer_groups = {
                'layer4': 1.0,      # High LR for final layers
                'layer3': 0.5,      # Medium LR
                'layer2': 0.1,      # Low LR
                'layer1': 0.01      # Very low LR for early layers
            }
        """
        param_groups = []
        
        for pattern, multiplier in layer_groups.items():
            params = []
            for name, param in self.model.named_parameters():
                if pattern in name:
                    params.append(param)
            
            if params:
                param_groups.append({
                    'params': params,
                    'lr': base_lr * multiplier
                })
                print(f'Group {pattern}: LR = {base_lr * multiplier}')
        
        optimizer = torch.optim.Adam(param_groups)
        return optimizer
