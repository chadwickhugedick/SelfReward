import torch
import logging

def get_device(use_cpu=False, config_device=None):
    """
    Centralized device selection utility.

    Args:
        use_cpu (bool): If True, force CPU usage.
        config_device (str): Specific device string from config (e.g., 'cuda:0').

    Returns:
        torch.device: The selected device.
    """
    if use_cpu:
        device = torch.device('cpu')
    elif config_device:
        device = torch.device(config_device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Log CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA is available. Using device: {device}")
    else:
        print(f"CUDA is not available. Using device: {device}")

    return device