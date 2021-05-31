import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
from imblearn.over_sampling import ADASYN
from sklearn import ensemble
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report

parser = argparse.ArgumentParser(description='Other data-driven approaches for Odor Perception')
parser.add_argument('--data', type=str, default=None, help='Dataset.', required=True)
parser.add_argument('--data_path', type=str, default=None, help='The path of dataset.', required=True)
opt = parser.parse_args()

# Dataset
if opt.data == 'sweet':
    print('Dataset : Sweet')
    PATH_DATA_ODOR_SWEET = opt.data_path
    df_sweet = pd.read_csv(PATH_DATA_ODOR_SWEET)
    X, y = df_sweet[df_sweet.columns[:-1]].to_numpy(), df_sweet[df_sweet.columns[-1]].to_numpy()
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=1, stratify=y)
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
sc = StandardScaler()
X_train_val = sc.fit_transform(X_train_val)
X_test = sc.transform(X_test)

list_clfs = ['adaboost', 'gbm', 'xgboost', 'randomforest', 'knn', 'svm']
PATH_HYPERPARAM = './hyperparams/study_'
results = []
for name_clf in list_clfs:
    study = joblib.load(PATH_HYPERPARAM + opt.data + '_' + name_clf + '.pkl')
    if name_clf == 'adaboost':
        clf = ensemble.AdaBoostClassifier(**study.best_params)
    elif name_clf == 'gbm':
        clf = ensemble.GradientBoostingClassifier(**study.best_params)
    elif name_clf == 'xgboost':
        clf = xgb.XGBClassifier(**study.best_params)
    elif name_clf == 'randomforest':
        clf = ensemble.RandomForestClassifier(**study.best_params)
    elif name_clf == 'knn':
        clf = KNeighborsClassifier(**study.best_params)
    else:
        print(study.best_params)
        clf = SVC(**study.best_params)

    clf.fit(X_train_val, y_train_val)
    cr = classification_report(clf.predict(X_test), y_test, output_dict=True)
    f1 = cr['weighted avg']['f1-score']
    print('Classifier: {}, Test F1 Score: {}'.format(name_clf, f1))
