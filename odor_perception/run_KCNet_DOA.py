import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from models.KCNet import KCNet
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import ADASYN


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='KCNet with DOA for Odor Perception')
parser.add_argument('--epoch', type=int, default=20, help='The number of epoch.')
parser.add_argument('--data', type=str, default=None, help='Dataset.', required=True)
parser.add_argument('--data_path', type=str, default=None, help='The path of dataset.', required=True)
parser.add_argument('--hsize', type=int, default=100, help='NThe number of hidden nodes.')
parser.add_argument('--lr', type=float, default=0.1, help='Learning rate.')
parser.add_argument('--stop_metric', type=float, default=0.9, help='Stopping criteria metric.')
parser.add_argument('--show_images', type=str2bool, default=False,
                    help='Shows all the images of weights and scores')
opt = parser.parse_args()

# Dataset
if opt.data == 'sweet':
    print('Dataset : Sweet')
    PATH_DATA_ODOR_SWEET = opt.data_path
    df_sweet = pd.read_csv(PATH_DATA_ODOR_SWEET)
    X, y = df_sweet[df_sweet.columns[:-1]].to_numpy(), df_sweet[df_sweet.columns[-1]].to_numpy()
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=1, stratify=y)
    sc = StandardScaler()
    X_train_val = sc.fit_transform(X_train_val)
    X_test = sc.transform(X_test)
    alpha = 100
else:
    print('Dataset : Musky')
    PATH_DATA_ODOR_MUSKY = opt.data_path
    df_musky = pd.read_csv(PATH_DATA_ODOR_MUSKY)
    X, y = df_musky[df_musky.columns[:-1]].to_numpy(), df_musky[df_musky.columns[-1]].to_numpy()
    # define oversampling strategy
    oversample = ADASYN(random_state=1)
    # fit and apply the transform
    X_over, y_over = oversample.fit_resample(X, y)
    X_train_val, X_test, y_train_val, y_test = train_test_split(X_over, y_over, test_size=0.1, random_state=1)
    alpha = 115
# apply standard normalization to dataset
sc = StandardScaler()
X_train_val = sc.fit_transform(X_train_val)
X_test = sc.transform(X_test)

model = KCNet(input_size=159, h_size=opt.hsize, alpha=alpha, gen_S=True)
if opt.show_images:
    init_score = model.get_S()
    ax = sns.heatmap(init_score).set_title('Initial Preference Score Matrix')
    plt.show()
    init_w = model.get_W()
    ax = sns.heatmap(init_w).set_title('Initial Weight Matrix')
    plt.show()

prev_f1 = -1
for epoch in range(opt.epoch):
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1, stratify=y_train_val)
    # 1. Forward Pass
    model.fit(X_train, y_train)
    # 2. Backward Pass
    model.update_W(X_val, y_val, opt.lr)
    # Evaluate model
    cr = classification_report(model.predict_label(X_val), y_val, output_dict=True)
    cur_f1 = cr['weighted avg']['f1-score']
    print('Epoch {} - Validation weighted F1 score: {}'.format(epoch + 1, cur_f1))
    # 3. Stopping Criteria check and Increase number of hidden units later
    if cur_f1 > opt.stop_metric:
        print('Stopping criteria satisfied.')
        break
    prev_f1 = cur_f1

model.fit(X_train_val, y_train_val)
cr = classification_report(model.predict_label(X_test), y_test, output_dict=True)
f1 = cr['weighted avg']['f1-score']
print('Test weighted F1 score: {}'.format(f1))
if opt.show_images:
    ax = sns.heatmap(model.get_W()).set_title('Final Weight Matrix after DOA')
    plt.show()
