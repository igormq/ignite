import collections

import torch
from torch._six import string_classes


def _to_hours_mins_secs(time_taken):
    mins, secs = divmod(time_taken, 60)
    hours, mins = divmod(mins, 60)
    return hours, mins, secs


def to_tensor(batch, device):
    if torch.is_tensor(batch):
        return torch.Tensor(batch, device=device)
    elif isinstance(batch, string_classes):
        return batch
    elif isinstance(batch, collections.Mapping):
        return {k: to_tensor(sample, device) for k, sample in batch.items()}
    elif isinstance(batch, collections.Sequence):
        return [to_tensor(sample, device) for sample in batch]
    else:
        raise TypeError(("batch must contain tensors, numbers, dicts or lists; found {}"
                         .format(type(batch[0]))))


def to_onehot(indices, num_classes):
    onehot = torch.zeros(indices.size(0), num_classes)
    if indices.is_cuda:
        onehot = onehot.cuda()
    return onehot.scatter_(1, indices.unsqueeze(1), 1)
