"""
Device utilities for PyTorch model training.
"""
import torch


def get_device(use_cpu=False):
    """
    Get the appropriate device for training.
    
    Args:
        use_cpu (bool): Force CPU usage even if GPU is available
        
    Returns:
        str: Device string ('cuda' or 'cpu')
    """
    if use_cpu:
        device = 'cpu'
        print("Forced CPU usage")
    elif torch.cuda.is_available():
        device = 'cuda'
        print("CUDA is available. Using GPU")
    else:
        device = 'cpu'
        print("CUDA not available. Using CPU")
    
    return device


def get_device_info():
    """
    Get detailed device information.
    
    Returns:
        dict: Device information
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': torch.cuda.current_device() if torch.cuda.is_available() else None,
        'device_name': torch.cuda.get_device_name() if torch.cuda.is_available() else None
    }
    
    return info


def set_device(device_str):
    """
    Set the CUDA device.
    
    Args:
        device_str (str): Device string
    """
    if torch.cuda.is_available() and device_str.startswith('cuda'):
        torch.cuda.set_device(device_str)
        print(f"Set device to {device_str}")
    else:
        print(f"Using {device_str}")