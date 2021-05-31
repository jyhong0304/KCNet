import argparse

import torch
import torchvision.datasets
import torchvision.transforms as transforms

from models.KCNet import KCNet
from helper_func import to_onehot
from helper_func import get_all_data
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='KCNet for MNIST')
parser.add_argument('--hsize', type=int, default=1000, help='The number of hidden units.', required=True)
parser.add_argument('--num_workers', type=int, default=4, help='The number of workers.')
parser.add_argument('--no_cuda', type=str2bool, default=False, help='No use cuda')
opt = parser.parse_args()

#################
# Parameters
#################
use_cuda = not opt.no_cuda and torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
image_size = 28 * 28
num_classes = 10

train_kwargs = {
    'num_workers': opt.num_workers
}
test_kwargs = {
    'num_workers': opt.num_workers
}
if use_cuda:
    cuda_kwargs = {
        'num_workers': 1,
        'pin_memory': True
    }
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

##################
# Datasets
##################
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
dataset = torchvision.datasets.MNIST(root='~/data/mnist/', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='~/data/mnist/', train=False, transform=transform, download=True)

train_images, train_labels = get_all_data(dataset, device, shuffle=True, **train_kwargs)
train_labels = to_onehot(batch_size=len(dataset), num_classes=num_classes, y=train_labels, device=device)

test_images, test_labels = get_all_data(test_dataset, device, shuffle=False, **test_kwargs)
test_labels = to_onehot(batch_size=len(test_dataset), num_classes=num_classes, y=test_labels, device=device)
print('Data done.')

#################
# Model
#################
start_time = time.time()
model = KCNet(input_size=image_size, h_size=opt.hsize, num_glom_inputs=7, num_classes=num_classes, device=device,
              reg=13)
model.fit(train_images, train_labels)
acc = model.evaluate(test_images, test_labels)
print('Training time: {}'.format(time.time() - start_time))
print('Model with hsize {} - Accuracy: {}'.format(opt.hsize, acc))
