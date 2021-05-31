from sklearn.linear_model import RidgeClassifier
from sklearn.base import BaseEstimator
import numpy as np
from scipy.special import expit


class KCNet(BaseEstimator):
    def __init__(self, input_size, h_size=2000, num_glom_inputs=7,
                 decoder=None, alpha=1., W=None, S=None, gen_S=False):
        self.input_size = input_size
        self.h_size = h_size
        self.num_glom_inputs = num_glom_inputs
        self.decoder = decoder
        self.alpha = alpha
        if W is None:
            self.W = self.generate_sparse_weight(self.input_size)
        else:
            self.W = W
        self.S = S
        if gen_S and self.S is None:
            self.S = self.weight_to_score()

    def generate_score(self, lb=-1, ub=0.02):
        return np.random.uniform(lb, ub, size=(self.h_size, self.input_size))

    def score_to_weight(self):
        return np.where(self.S > 0, 1, 0)

    def weight_to_score(self, lb=-1, ub=0.02):
        on_S = np.random.uniform(0, ub, size=(self.h_size, self.input_size))
        off_S = np.random.uniform(lb, 0, size=(self.h_size, self.input_size))
        return np.where(self.W > 0, on_S, off_S)

    def generate_sparse_weight(self, num_feature):
        weight = np.zeros((self.h_size, num_feature))
        for i in range(self.h_size):
            final_num_input = np.clip(self.num_glom_inputs, 1, num_feature).item()
            indices = np.random.choice(num_feature, final_num_input, replace=False)
            weight[i, indices] = 1.
        return weight

    def generate_feature_map(self, x):
        R_IN = np.dot(x, self.W.T)
        R_OUT = np.maximum(
            (R_IN - np.repeat(np.mean(R_IN, axis=1, keepdims=True), self.h_size, axis=1)), 0)
        return R_OUT

    def fit(self, x, y):
        if self.W is None:
            self.W = self.generate_sparse_weight(x.shape[1])
        if self.S is None:
            self.S = self.weight_to_score()
        if self.decoder is None:
            self.decoder = RidgeClassifier(alpha=self.alpha, fit_intercept=False)
        return self.decoder.fit(self.generate_feature_map(x), y)

    def predict_internal(self, x):
        self.rep_h = self.generate_feature_map(x)
        beta = self.decoder.coef_.T
        preds = expit(np.dot(self.rep_h, beta))
        return preds

    def predict(self, x):
        return self.decoder.predict(self.generate_feature_map(x))

    def score(self, x, y):
        return self.decoder.score(self.generate_feature_map(x), y)

    def predict_label(self, x):
        preds = self.predict_internal(x)
        return np.where(preds > .5, 1, 0)

    def update_W(self, x, y, lr):
        # dim of error = x.shape[0] * nClass
        preds = self.predict_internal(x)
        y = y.reshape(-1, 1)
        # Binary Cross entropy derivative
        errors = np.where(y > 0, preds - y, preds)
        beta = self.decoder.coef_.T
        # dim of d_hid = x.shape[0] * nKC * (mask of hidden unit)
        d_hid = np.dot(errors, beta.T)
        # # Active neuron is 1. Otherwise, 0
        mask_h = np.where(self.rep_h > 0, 1, 0)
        # Element-wise multiplication
        d_hid = d_hid * mask_h
        # dim of mask_x = x.shape[0] * input_dim
        # 1, if feature is used. 0, otherwise
        # dim of W = nKC * d
        mask_x = np.sum(self.W, axis=0, keepdims=True)
        mask_x = np.where(mask_x > 0, 1, 0)
        mask_x = mask_x.repeat(x.shape[0], 0)
        # dim of d_Wih = nKC * d
        x = x * mask_x
        d_Wih = np.dot(x.T, d_hid).T
        self.S -= lr * d_Wih
        self.S = np.clip(self.S, a_min=-1, a_max=1)
        self.W = self.score_to_weight()

    def append_score(self, n_add_hsize, lb=-1, ub=0.02):
        add_score = np.random.uniform(lb, ub, size=(n_add_hsize, self.input_size))
        self.S = np.concatenate((self.S, add_score), axis=0)
        self.h_size += n_add_hsize
        self.W = self.score_to_weight()

    def get_params(self, deep=True):
        return {
            'input_size': self.input_size,
            'h_size': self.h_size,
            'alpha': self.alpha
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_S(self):
        return self.S

    def get_W(self):
        return self.W
