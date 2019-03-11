import torch

from torch.nn import Module

def load_model(path: str, model: Module):
    state = torch.load(path)
    model.load_state_dict(state['model'])
