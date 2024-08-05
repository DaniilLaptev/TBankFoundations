
import os
import json

import random
import numpy as np

import torch

def save(name, model, loss, eval, extr = [], path = '.'):
    """Сохраняет историю обучения и модель на диск.

    Args:
        name (str): название запуска, отображаемое в wandb и логах
        model (nn.Module): объект PyTorch модели
        loss (list): история функции потерь
        eval (list): история валидации во время тренировки
        extr (list, optional): результаты вычисления экстраполяции, по дефолту [].
        path (str, optional): путь сохранения результатов, по дефолту '.'.
    """
    
    os.makedirs(os.path.dirname(path + '/data/'), exist_ok=True)
    os.makedirs(os.path.dirname(path + '/models/'), exist_ok=True)
    
    result = json.dumps({
        'loss': loss,
        'eval': eval,
        'extr': extr
    })
    
    with open(f'{path}/data/{name}.json', 'w') as f:
        f.write(result)
    torch.save(model, f'{path}/models/{name}.pt')
        
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)