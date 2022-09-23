import torch
import random
import numpy as np
import time
import datetime


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))