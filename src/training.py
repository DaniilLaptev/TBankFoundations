
import torch
from tqdm.notebook import tqdm_notebook as tqdm

def evaluate(loader, model, b = None):
        
    with torch.no_grad():
        total = 0
        for (x, y) in loader:
            
            preds = model(x[:, :-1], b)
            preds = torch.stack(preds)
            targs = torch.stack([y] * b)
            
            # Сначала по итерациям, затем по батчам
            loss = (targs[:,:,-1] - preds[:,:,-1]).square().mean(dim=0).mean()
            
            total += loss.item() / loader.dataset.n_dims
    return total / len(loader)

def train(
    model,
    train_loader,
    test_loader,
    optimizer,
    steps,
    b = 1,
    run = None,
    log_every = 1
):
    """Стандартный цикл обучения модели.

    Args:
        model (nn.Module): объект модели PyTorch
        train_loader (torch.DataLoader): тренировочный даталоадер
        test_loader (torch.DataLoader): валидационный даталоадер
        optimizer (torch.Optim): оптимизатор PyTorch
        steps (int, optional): число шагов обучения; количество батчей, которое будет показано модели
        b (int, optional): число тренировочных итераций, по дефолту 1
        run (wandb.Run, optional): объект wandb Run, в который будет производиться логирование, по дефолту None
        log_every (int, optional): число итераций, после которых будет производиться очередной вызов логирования в wandb, по дефолту 1

    Returns:
        loss_history, eval_history: история значения функции потерь и валидационной метрики
    """
    
    train_loader.dataset.length = steps * train_loader.batch_size
    
    evaluate_every = steps // 5
    
    loss_history = []
    eval_history = []
    
    val_loss = evaluate(test_loader, model, b)
    eval_history.append(val_loss)
    
    if run is not None:
        run.log({'Eval Loss': val_loss}, commit = False, step = 1)
    
    pbar, step = tqdm(range(steps)), 0
    for (x, y) in train_loader:
        optimizer.zero_grad()

        preds = model(x[:, :-1], b)
        preds = torch.stack(preds)
        targs = torch.stack([y] * b)
        
        # Сначала по токенам, затем по итерациям, затем по батчам
        loss = (targs - preds).square().mean(dim=2).mean(dim=0).mean()
        loss = loss / train_loader.dataset.n_dims
        
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        pbar.set_description(f'Train loss: {loss.item():.5f}')
        pbar.update(1)
        
        step += 1
        
        if step % evaluate_every == 0:
            val_loss = evaluate(test_loader, model, b)
            eval_history.append(val_loss)
            if run is not None:
                run.log({'Eval Loss': val_loss}, commit = False, step = step)
                
        # Логирует нулевой шаг, а также каждый log_every
        if run is not None and (step % log_every == 0 or step == 1):
            run.log({'Train Loss': loss.item()}, step = step)
    
    return loss_history, eval_history