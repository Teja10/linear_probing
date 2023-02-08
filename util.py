import torch

def save_checkpoint(epoch: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer, loss: float, path: str):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

def save_model_inference(model: torch.nn.Module, path: str):
    torch.save(model, path)