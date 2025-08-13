import random

import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

if torch.cuda.is_available():
    _device = torch.device("cuda")
else:
    _device = torch.device("cpu")
print(f"Use Device: {_device}")


def get_device():
    return _device

def calculate_acc(outputs, labels):
    predicted_labels = torch.argmax(outputs, dim=1)
    correct = (predicted_labels == labels).sum().item()
    total = labels.size(0)
    epoch_acc = correct / total

    return epoch_acc

def calculate_rmse(predictions, targets):
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    return rmse

def calculate_pcc(predictions, targets):
    pcc, _ = pearsonr(predictions, targets)
    return pcc

def reset_seed(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic=True
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False
    # torch.use_deterministic_algorithms(True)
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    # os.environ['PYTHONHASHSEED'] = str(seed)


def calculate_mean_and_std(numbers):
    if not numbers:
        return None, None
    
    mean = np.mean(numbers)
    std_dev = np.std(numbers)
    
    return mean, std_dev
