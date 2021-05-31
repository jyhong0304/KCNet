import torch
from torch.utils.data import DataLoader


#####################
# Helper Functions
#####################
def to_onehot(batch_size, num_classes, y, device):
    # One hot encoding buffer that you create out of the loop and just keep reusing
    y_onehot = torch.FloatTensor(batch_size, num_classes).to(device)
    # y = y.type(dtype=torch.long)
    y = torch.unsqueeze(y, dim=1)
    # In your for loop
    y_onehot.zero_()
    y_onehot.scatter_(1, y, 1)

    return y_onehot


def get_all_data(dataset, device, num_workers=0, shuffle=False, pin_memory=False):
    dataset_size = len(dataset)
    data_loader = DataLoader(dataset, batch_size=dataset_size, num_workers=num_workers, shuffle=shuffle,
                             pin_memory=pin_memory)
    for i_batch, sample_batched in enumerate(data_loader):
        images, labels = sample_batched[0].view(len(dataset), -1).to(device), sample_batched[1].to(device)
    return images, labels

