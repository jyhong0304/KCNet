import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from models.FSHN import FSHN
import argparse


def train(model, device, train_loader, optimizer, epoch, LOG_INTERVAL, DRY_RUN):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            if DRY_RUN:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


#################
# Parameters
#################
parser = argparse.ArgumentParser(description='Fully-trained Single-Hidden-layer Neural Network for MNIST')
parser.add_argument('--batch_size', type=int, default=64, help='The number of batch size for training.')
parser.add_argument('--test_batch_size', type=int, default=1000, help='The number of batch size for testing.')
parser.add_argument('--epoch', type=int, default=5, help='The number of epoch.')
parser.add_argument('--lr', type=float, default=0.1, help='Learning rate.')
parser.add_argument('--num_workers', type=int, default=4, help='The number of workers.')
parser.add_argument('--no_cuda', type=str2bool, default=False, help='No use cuda')
parser.add_argument('--dry_run', type=str2bool, default=False, help='Use dry run')
parser.add_argument('--log_interval', type=int, default=4, help='The number of steps for showing log.')
opt = parser.parse_args()

##################
# Datasets
##################
torch.manual_seed(1)
USE_CUDA = not opt.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

train_kwargs = {'batch_size': opt.batch_size,
                'num_workers': opt.num_workers}
test_kwargs = {'batch_size': opt.test_batch_size,
               'num_workers': opt.num_workers}
if USE_CUDA:
    cuda_kwargs = {'num_workers': 1,
                   'pin_memory': True,
                   'shuffle': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
dataset1 = datasets.FashionMNIST(root='~/data/fashion_mnist/', train=True, transform=transform, download=True)
dataset2 = datasets.FashionMNIST(root='~/data/fashion_mnist/', train=False, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

model = FSHN(28 * 28, 89, 10).to(device)
optimizer = optim.SGD(model.parameters(), lr=opt.lr)
start_time = time.time()
for epoch in range(1, opt.epoch + 1):
    train(model, device, train_loader, optimizer, epoch, opt.log_interval, opt.dry_run)
    test(model, device, test_loader)

print('Training time: {} sec'.format(time.time() - start_time))
print('Number of learnable parameters: {}'.format(count_parameters(model)))
