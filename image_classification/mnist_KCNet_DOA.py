import argparse
import torch
import torchvision.datasets
import torchvision.transforms as transforms
from models.KCNet import KCNet
from helper_func import to_onehot
from helper_func import get_all_data
import matplotlib.pyplot as plt
import seaborn as sns


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='KCNet with DOA for MNIST')
parser.add_argument('--epoch', type=int, default=50, help='The number of epoch.', required=True)
parser.add_argument('--hsize', type=int, default=1000, help='The number of hidden units.', required=True)
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

train_val_images, train_val_labels = get_all_data(dataset, device, shuffle=True, **train_kwargs)
train_val_labels = to_onehot(batch_size=len(dataset), num_classes=num_classes, y=train_val_labels, device=device)

test_images, test_labels = get_all_data(test_dataset, device, shuffle=False, **test_kwargs)
test_labels = to_onehot(batch_size=len(test_dataset), num_classes=num_classes, y=test_labels, device=device)
print('Data preparation done.')

#################
# Model
#################
model = KCNet(input_size=image_size, h_size=opt.hsize, num_classes=num_classes, gen_S=True, device=device, reg=13)
if opt.show_images:
    if use_cuda:
        S = model.get_S().cpu()
        W = model.get_W().cpu()
    else:
        S = model.get_S()
        W = model.get_W()
    ax = sns.heatmap(S).set_title('Initial Preference Score Matrix')
    plt.show()
    ax = sns.heatmap(W).set_title('Initial Weight Matrix')
    plt.show()
prev_acc = -1
for epoch in range(opt.epoch):
    train, val = torch.utils.data.random_split(train_val_images, (50000, 10000))
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
    print('Epoch {} - Val Accuracy: {}'.format(epoch + 1, cur_acc))
    # Early stopping criteria
    if cur_acc > opt.stop_metric:
        print('Early Stopping.')
        break
    prev_acc = cur_acc
# Evaluate using test
model.fit(train_val_images, train_val_labels)
if opt.show_images:
    if use_cuda:
        W = model.get_W().cpu()
    else:
        W = model.get_W()
    ax = sns.heatmap(W).set_title('Final Weight Matrix after DOA')
    plt.show()
test_acc = model.evaluate(test_images, test_labels)
print('KCNet with DOA Test acc : {}'.format(test_acc))
