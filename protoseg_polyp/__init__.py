import torch

# GPU when available, otherwise CPU (evaluation and analysis run on CPU, just slower).
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
