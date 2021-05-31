import torch
import numpy as np


class KCNet():
    def __init__(self,
                 input_size,
                 num_classes,
                 h_size=2000,
                 num_glom_inputs=7,
                 reg=1,
                 init_W=None,
                 init_S=None,
                 gen_S=False,
                 device=None):
        self._input_size = input_size
        self._h_size = h_size
        self._num_glom_inputs = num_glom_inputs
        self._output_size = num_classes
        self._device = device
        self._reg = reg

        if init_W is None:
            self._W = self.generate_sparse_weight().to(self._device)
        else:
            self._W = init_W.to(self._device)
        if gen_S and init_S is None:
            self._S = self.weight_to_score().to(self._device)
        self._beta = torch.zeros((self._h_size, self._output_size)).to(self._device)
        self._activation = torch.relu

    def weight_to_score(self, lb=-1, ub=0.02):
        on_S = np.random.uniform(0, ub, size=(self._h_size, self._input_size))
        off_S = np.random.uniform(lb, 0, size=(self._h_size, self._input_size))
        return torch.from_numpy(np.where(self._W.cpu() > 0, on_S, off_S)).float()

    def score_to_weight(self):
        return torch.where(self._S > 0, torch.ones_like(self._S), torch.zeros_like(self._S))

    def generate_sparse_weight(self):
        weight = np.zeros((self._h_size, self._input_size))
        for i in range(self._h_size):
            final_num_input = np.clip(self._num_glom_inputs, 1, self._input_size).item()
            indices = np.random.choice(self._input_size, final_num_input, replace=False)
            weight[i, indices] = 1.
        return torch.from_numpy(weight).float()

    def generate_feature_map(self, x):
        R_IN = x.mm(torch.transpose(self._W, 0, 1))
        R_OUT = self._activation(R_IN - torch.mean(R_IN, dim=1, keepdim=True))
        self._rep_h = R_OUT
        return R_OUT

    def predict(self, x):
        h = self.generate_feature_map(x)
        out = h.mm(self._beta)
        return torch.softmax(out, dim=1)

    def fit(self, x, t):
        H = self.generate_feature_map(x)
        H_tran = torch.transpose(H, 0, 1)
        HH = H_tran.mm(H)
        H_pinv = torch.pinverse(HH + self._reg * torch.eye(self._h_size).to(self._device))
        self._beta = H_pinv.mm(H_tran).mm(t)

    def update_W(self, x, t, lr):
        # dim of error = x.shape[0] * nClass
        preds = self.predict(x)
        # Cross entropy derivative
        errors = preds - t
        # dim of d_hid = x.shape[0] * nKC * (mask of hidden unit)
        d_hid = errors.mm(torch.transpose(self._beta, 0, 1))
        # # Active neuron is 1. Otherwise, 0
        mask_h = torch.where(self._rep_h > 0, torch.ones_like(self._rep_h), torch.zeros_like(self._rep_h))
        # Element-wise multiplication
        d_hid = d_hid * mask_h
        # dim of mask_x = x.shape[0] * input_dim
        # 1, if feature is used. 0, otherwise
        # dim of W = nKC * d
        mask_x = torch.sum(self._W, dim=0, keepdim=True)
        mask_x = torch.where(mask_x > 0, torch.ones_like(mask_x), torch.zeros_like(mask_x))
        mask_x = mask_x.repeat(x.size()[0], 1)
        # dim of d_Wih = nKC * d
        input = x * mask_x
        d_Wih = torch.transpose(torch.transpose(input, 0, 1).mm(d_hid), 0, 1)
        # mask_s = torch.where(model.get_W() > 0, 1, -1)
        self._S -= lr * d_Wih
        self._S = torch.clamp(self._S, min=-1, max=1)
        self._W = self.score_to_weight()

    def append_score(self, n_add_hsize, lb=-1, ub=0.02):
        add_score = torch.from_numpy(np.random.uniform(lb, ub, size=(n_add_hsize, self._input_size))).float().to(
            self._device)
        self._S = torch.cat((self._S, add_score), dim=0)
        self._h_size += n_add_hsize
        self._W = self.score_to_weight()

    def evaluate(self, x, t):
        y_pred = self.predict(x)
        acc = torch.sum(torch.argmax(y_pred, dim=1) == torch.argmax(t, dim=1)).item() / len(t)
        return acc

    def get_beta(self):
        return self._beta

    def get_rep_h(self):
        return self._rep_h

    def get_W(self):
        return self._W

    def get_S(self):
        return self._S
