import logging

def get_root_logger(log_file=None, log_level=logging.INFO):
    logger = logging.getLogger()
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        if log_file is not None:
            file_handler = logging.FileHandler(log_file, 'w')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    logger.setLevel(log_level)
    return logger
#下面的是我自己加的
import torch

def get_paddings_indicator(points, num_points, max_num_points=None):
    """
    Args:
        points [torch.Tensor]: (B, N, C), float tensor
        num_points [torch.Tensor]: (B,), int tensor
        max_num_points int: maximum number of points for the padded_tensor
    Returns:
        paddings_indicator [torch.Tensor]: (B, max_num_points), 0/1 tensor
    """
    B, N = points.shape[:2]
    if max_num_points is None:
        max_num_points = num_points.max()
    device = points.device
    paddings_indicator = torch.zeros((B, max_num_points), dtype=torch.float32, device=device)
    arange = torch.arange(max_num_points, dtype=torch.long, device=device)
    mask = arange[None, :] < num_points[:, None]
    paddings_indicator[mask] = 1
    return paddings_indicator


def example_convert_to_torch(example, device=None):
    example_torch = {}
    for key in example:
        if key == 'metadata':
            example_torch[key] = example[key]
        elif key == 'points':
            example_torch[key] = example[key].float().to(device)
        else:
            example_torch[key] = torch.from_numpy(example[key]).to(device)
    return example_torch