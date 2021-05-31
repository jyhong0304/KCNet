import argparse
import torch
import torchvision.datasets
import torchvision.transforms as transforms
from models.KCNet import KCNet
from helper_func import to_onehot
from helper_func import get_all_data
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='KCNet with Ensemble DOA for EMNIST-Balanced')
parser.add_argument('--n_submodels', type=int, default=10, help='The number of submodels.')
parser.add_argument('--epoch', type=int, default=50, help='The number of epoch.', required=True)
parser.add_argument('--hsize', type=int, default=1000, help='The number of hidden units for each submodel.',
                    required=True)
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
parser.add_argument('--stop_metric', type=float, default=0.9, help='Stopping criteria metric.')
parser.add_argument('--num_workers', type=int, default=4, help='The number of workers.')
parser.add_argument('--no_cuda', type=str2bool, default=False, help='No use cuda')
parser.add_argument('--show_images', type=str2bool, default=False, help='Shows all the images of weights and scores')
opt = parser.parse_args()

#################
# Parameters
#################
use_cuda = not opt.no_cuda and torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
image_size = 28 * 28
num_classes = 47

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
    transforms.ToTensor()
])
dataset = torchvision.datasets.EMNIST(root='~/data/emnist/', split='balanced',
                                      train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.EMNIST(root='~/data/emnist/', split='balanced',
                                           train=False, transform=transform, download=True)

train_val_images, train_val_labels = get_all_data(dataset, device, shuffle=True, **train_kwargs)
train_val_labels = to_onehot(batch_size=len(dataset), num_classes=num_classes, y=train_val_labels, device=device)

test_images, test_labels = get_all_data(test_dataset, device, shuffle=False, **test_kwargs)
test_labels = to_onehot(batch_size=len(test_dataset), num_classes=num_classes, y=test_labels, device=device)
print('Data preparation done.')

#################
# Model
#################
results_val_acc = []
results_test_acc = []
results_W = []
for idx_submodel in range(opt.n_submodels):
    model = KCNet(input_size=image_size, h_size=opt.hsize, num_classes=num_classes, gen_S=True, device=device, reg=3)
    prev_acc = -1
    for epoch in range(opt.epoch):
        train, val = torch.utils.data.random_split(train_val_images, (94000, 18800))
        X_train = train_val_images[train.indices].to(device)
        y_train = train_val_labels[train.indices].to(device)
        X_val = train_val_images[val.indices].to(device)
        y_val = train_val_labels[val.indices].to(device)
        # 1. Forward Pass
        model.fit(X_train, y_train)
        # 2. Backward Pass
        model.update_W(X_val, y_val, opt.lr)
        # Evaluate model
        cur_acc = model.evaluate(X_val, y_val)
        results_val_acc.append({'Epoch': epoch + 1, 'Accuracy': cur_acc})
        print('Submodel. {}: Epoch {} - Val accuracy: {}'.format(idx_submodel + 1, epoch + 1, cur_acc))
        # Early stopping criteria
        if cur_acc > opt.stop_metric:
            print('Early Stopping.')
            break
        prev_acc = cur_acc
    # Evaluate using test
    model.fit(train_val_images, train_val_labels)
    if use_cuda:
        W = model.get_W().cpu()
    else:
        W = model.get_W()
    results_W.append(W)
    test_acc = model.evaluate(test_images, test_labels)
    results_test_acc.append(test_acc)
    if use_cuda:
        torch.cuda.empty_cache()

print('Average submodel Test accuracy : {} +- {}'.format(
    np.mean(results_test_acc),
    2 * np.std(results_test_acc)
))

print('Ensemble Model')
model = KCNet(input_size=image_size, h_size=opt.hsize * opt.n_submodels, num_classes=num_classes,
              init_W=torch.cat(results_W, dim=0), device=device, reg=3)
model.fit(train_val_images, train_val_labels)
test_acc = model.evaluate(test_images, test_labels)
print('Ensemble model Test Accuracy : {}'.format(test_acc))
if opt.show_images:
    if use_cuda:
        W = model.get_W().cpu()
    else:
        W = model.get_W()
    ax = sns.heatmap(W).set_title('Ensemble Weight Matrix')
    plt.show()
