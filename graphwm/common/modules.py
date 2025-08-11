import warnings
import torch
from torch.optim.lr_scheduler import _LRScheduler

class CustomScheduleLR(_LRScheduler):
    """
    The learning rate schedule used in GNS.
    Updated for PyTorch 1.11+ compatibility.
    """
    def __init__(self, optimizer, min_lr, decay_steps, decay_rate, last_epoch=-1, verbose=False):
        self.min_lr = min_lr
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        super().__init__(optimizer, last_epoch, verbose)  # Updated super call
    
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        
        # Calculate new learning rates for all parameter groups
        return [self.min_lr + (base_lr - self.min_lr) * 
                self.decay_rate ** (self.last_epoch / self.decay_steps)
                for base_lr in self.base_lrs]

    def _get_closed_form_lr(self):
        # Consistent with get_lr() using last_epoch instead of step_count
        return [self.min_lr + (base_lr - self.min_lr) * 
                self.decay_rate ** (self.last_epoch / self.decay_steps)
                for base_lr in self.base_lrs]


class StandardScalerTorch:
    """Normalizes the targets of a dataset. Updated for PyTorch 1.11+ compatibility."""
    def __init__(self, means=None, stds=None):
        self.means = means
        self.stds = stds

    def fit(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float)
        self.means = torch.mean(X, dim=0)
        self.stds = torch.std(X, dim=0, unbiased=False) + 1e-5  # Prevent division by zero

    def transform(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float)
        self.match_device(X)
        return (X - self.means) / self.stds

    def inverse_transform(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float)
        self.match_device(X)
        return X * self.stds + self.means

    def match_device(self, tensor):
        if self.means.device != tensor.device:
            self.means = self.means.to(tensor.device)
            self.stds = self.stds.to(tensor.device)
    
    def to(self, device):
        """Move scaler to specified device"""
        self.means = self.means.to(device)
        self.stds = self.stds.to(device)
        return self

    def copy(self):
        return StandardScalerTorch(
            means=self.means.clone().detach(),
            stds=self.stds.clone().detach())

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"means: {self.means.tolist() if self.means is not None else None}, "
            f"stds: {self.stds.tolist() if self.stds is not None else None})"
        )


def get_scaler_from_data_list(data_list, key):
    """Create scaler from list of data dictionaries"""
    # Handle empty lists safely
    if not data_list:
        raise ValueError("data_list is empty")
    
    # Concatenate values efficiently
    targets = torch.cat([d[key] if torch.is_tensor(d[key]) else torch.tensor(d[key])
                  for d in data_list])
    
    scaler = StandardScalerTorch()
    scaler.fit(targets)
    return scaler
