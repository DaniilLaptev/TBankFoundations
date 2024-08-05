

import torch
from torch.utils.data import Dataset

class LinregDataset(Dataset):
    def __init__(
        self, 
        n_dims, n_points,
        mean = 0, std = 1, 
        total = 10000, 
        xs = None, ys = None,
        random = True,
        device = 'cpu'
        ):
        """Датасет, который генерирует новый набор данных при каждом вызове
        или выдаёт данные из заранее переданного фиксированного датасета.
        
        Реализуется линейная регрессия.

        Args:
            n_dims (int): число измерений для данной задачи
            n_points (int): число точек, в которых будет вычисляться значение функции
            mean (float, optional): среднее значение для генерации случайных данных, по дефолту 0
            std (float, optional): стандартное отклонение для генерации случайных данных, по дефолту 1
            total (int, optional): суммарное число сгенерированных примеров в датасете, по дефолту 10000
            xs (list, optional): координаты фиксированного датасета, по дефолту None
            ys (list, optional): значения функции фиксированного датасета, по дефолту None
            random (bool, optional): генерировать ли случайный датасет, по дефолту True
            device (torch.device | str, optional): вычислительное устройство, на которое будут отправляться данные, по дефолту 'cpu'

        Raises:
            ValueError: если random = False, то должен быть указан датасет.
        """
        
        self.n_dims = n_dims
        self.n_points = n_points
        self.truncate = 0
        
        self.xs = xs # [N, n, d]
        self.ys = ys # [N, n]
        self.length = total if self.xs is None else len(xs)
        self.current = 0
        
        self.mean = mean
        self.std = std
        
        if random:
            self.sample = self._generate
        else:
            if self.xs is None:
                raise ValueError('You must provide data for random = False.')
            self.sample = self._getdata 
        
        self.device = device
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        x, y = self.sample(idx)
        return self.combine(x, y), y
    
    def _generate(self, idx):
        xs = torch.normal(self.mean, self.std, (self.n_points, self.n_dims))
        xs[..., self.n_dims - self.truncate:] = 0
        w_b = torch.normal(0, 1, (self.n_dims, 1))
        w_b[..., self.n_dims - self.truncate:] = 0
        ys = (xs @ w_b).sum(-1)
        return xs.to(self.device), ys.to(self.device)
    
    def _getdata(self, idx):
        xs = self.xs[idx]
        ys = self.ys[idx]
        return xs.to(self.device), ys.to(self.device)
    
    def combine(self, xs, ys):
        """Метод для формирования входной матрицы.

        Args:
            xs (torch.Tensor): 2D тензор координат функции
            ys (torch.Tensor): 1D тензор значений функции

        Returns:
            torch.Tensor: входная матрица
        """
        n, d = xs.shape
        device = ys.device

        ys_b_wide = torch.cat(
            (
                ys.view(n, 1),
                torch.zeros(n, d-1, device=device),
            ),
            axis = 1,
        )

        zs = torch.stack((xs, ys_b_wide), dim = 1)
        zs = zs.view(2 * n, d)

        return zs