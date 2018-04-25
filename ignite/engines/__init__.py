from ignite.engines.engine import Engine, Events, State
from ignite._utils import to_tensor
import torch

def _prepare_batch(batch, device):
    x, y = batch
    x = to_tensor(x, device)
    y = to_tensor(y, device)
    return x, y


def create_supervised_trainer(model, optimizer, loss_fn, device='cpu'):
    """
    Factory function for creating a trainer for supervised models

    Args:
        model (torch.nn.Module): the model to train
        optimizer (torch.optim.Optimizer): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        device (str or torch.Device, optional): where to transfer batch (default: 'cpu')

    Returns:
        Engine: a trainer engine with supervised update function
    """
    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = _prepare_batch(batch, device)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        return loss.item()

    return Engine(_update)


def create_supervised_evaluator(model, metrics={}, device='cpu'):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (torch.nn.Module): the model to train
        metrics (dict of str: Metric): a map of metric names to Metrics
        device (str or torch.Device, optional): where to transfer batch (default: 'cpu')

    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            x, y = _prepare_batch(batch, device)
            y_pred = model(x)
        return y_pred

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine
