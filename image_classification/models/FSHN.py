import torch.nn as nn


class FSHN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FSHN, self).__init__()
        self.input_size = input_size
        self.hidden = nn.Linear(input_size, hidden_size)
        self.final = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, img):
        x = img.view(-1, self.input_size)
        x = self.relu(self.hidden(x))
        x = self.final(x)
        return x
