import random
import numpy as np
import torch

from certa.models.emt.logging_customized import setup_logging

setup_logging()


def initialize_gpu_seed(seed: int):
    device, n_gpu = setup_gpu()

    init_seed_everywhere(seed, n_gpu)

    return device, n_gpu


def init_seed_everywhere(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def setup_gpu():
    # Setup GPU parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    return device, n_gpu
