import torch


def encode_RLE(mask):
    """
    Encode a mask into a RLE.
    Args:
        mask: torch.bool [H, W] on CUDA
    Returns:
        runs: torch.tensor [N] - run lengths
    """
    flat = mask.flatten() # [H*W]
    
    if flat.numel() == 0:
        return torch.tensor([], device=flat.device), False
    
    starts_with_true = flat[0].item()
    
    # Find transitions between True/False
    # Add dummy values at start and end to handle boundaries
    padded = torch.cat([torch.tensor([not flat[0]], device=flat.device), flat, torch.tensor([not flat[-1]], device=flat.device)])
    
    # Find where values change
    transitions = torch.nonzero(padded[1:] != padded[:-1], as_tuple=False).flatten()
    
    # Calculate run lengths
    runs = torch.diff(transitions)
    
    # append a 1 if starts_with_true else a 0 so that we don't have to return starts_with_true
    if starts_with_true:
        runs = torch.cat([torch.tensor([1], device=runs.device), runs])
    else:
        runs = torch.cat([torch.tensor([0], device=runs.device), runs])
        
    return runs

def decode_RLE(runs, shape):
    """
    Decode a RLE into a mask.
    Args:
        runs: torch.tensor [N] - run lengths
        shape: tuple - shape to reshape result to
    Returns:
        mask: torch.bool [H, W] on CUDA
    """
    
    start_with_true = runs[0].item()
    runs = runs[1:]
    
    if runs.numel() == 0:
        return torch.zeros(shape, dtype=torch.bool, device=runs.device)
    
    # Create alternating pattern: start_with_true determines first value
    start_val = 1 if start_with_true else 0
    vals = (torch.arange(runs.numel(), device=runs.device) + start_val) % 2
    
    # Expand runs into full sequence
    expanded = torch.repeat_interleave(vals, runs).bool()
    
    # Reshape to target shape
    total_elements = shape[0] * shape[1] if len(shape) == 2 else shape[0]
    if expanded.numel() != total_elements:
        # Pad or truncate if needed
        if expanded.numel() < total_elements:
            padding = torch.zeros(total_elements - expanded.numel(), dtype=torch.bool, device=runs.device)
            expanded = torch.cat([expanded, padding])
        else:
            expanded = expanded[:total_elements]
    
    mask_dec = expanded.view(shape)
    return mask_dec