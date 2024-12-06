import torch
from torch.amp import GradScaler

# 动态选择设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 使用适合设备的 GradScaler
scaler = GradScaler(device=device)
