import torch


print(torch.device('cpu'), torch.cuda.device('cuda'), torch.cuda.device('cuda:1'))

print(torch.cuda.device_count())

print(torch.cuda.is_available())

print(torch.device('cuda'))


def try_gpu(i = 0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_all_gpus():
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


print(try_gpu())
