import argparse
import pandas as pd
from models.KCNet import KCNet
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
from imblearn.over_sampling import ADASYN

parser = argparse.ArgumentParser(description='KCNet for Odor Perception')
parser.add_argument('--data', type=str, default=None, help='Dataset.', required=True)
parser.add_argument('--data_path', type=str, default=None, help='The path of dataset.', required=True)
parser.add_argument('--hsize', type=int, default=1000, help='The number of hidden nodes.')
opt = parser.parse_args()

# Dataset
if opt.data == 'sweet':
    print('Dataset : Sweet')
    PATH_DATA_ODOR_SWEET = opt.data_path
    df_sweet = pd.read_csv(PATH_DATA_ODOR_SWEET)
    X, y = df_sweet[df_sweet.columns[:-1]].to_numpy(), df_sweet[df_sweet.columns[-1]].to_numpy()
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=1, stratify=y)
    alpha = 100
else:
    print('Dataset : Musky')
    PATH_DATA_ODOR_MUSKY = opt.data_path
    df_musky = pd.read_csv(PATH_DATA_ODOR_MUSKY)
    X, y = df_musky[df_musky.columns[:-1]].to_numpy(), df_musky[df_musky.columns[-1]].to_numpy()
    # define oversampling strategy
    oversample = ADASYN(random_state=1)
    X_over, y_over = oversample.fit_resample(X, y)
    X_train_val, X_test, y_train_val, y_test = train_test_split(X_over, y_over, test_size=0.1, random_state=1)
    alpha = 115
# apply standard normalization to dataset
sc = StandardScaler()
X_train_val = sc.fit_transform(X_train_val)
X_test = sc.transform(X_test)

start_time = time.time()
print("Model training w/ hsize: {}".format(opt.hsize))
model = KCNet(input_size=159, h_size=opt.hsize, alpha=alpha)
model.fit(X_train_val, y_train_val)
preds = model.predict_label(X_test)
cr = classification_report(preds, y_test, output_dict=True)

print('time : {} sec'.format(time.time() - start_time))
print('Weighted F1 score: {}'.format(cr['weighted avg']['f1-score']))
