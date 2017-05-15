import pandas as pd
import numpy as np

from hyperopt import Trials, tpe, hp, fmin

import nn
import importlib
from keras_utils import run_hyperopt

import pickle

import sys
print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(sys.argv))


sigmas = np.exp(pd.read_csv('logsigmas.csv', header=None).values).T

v3_data = pd.read_csv('v3_data.csv')
cols_to_correct = ['subject', 'resp', 'contrast_id', 'g']

for col in cols_to_correct:
    v3_data[col] = v3_data[col] - 1


space = {'n_hidden_units': hp.quniform('n_hidden_units', 1, 200, 1),
         'dropout': hp.uniform('dropout', 0, .5),
         'n_train': hp.quniform('n_train', 1000, 1000000, 1),
         'optimizer': hp.choice('optimizer', ['adadelta', 'adagrad']),
         'batch_size': hp.quniform('batch_size', 10, 1000, 10),
         'l2': hp.loguniform('l2', -20, 0)
         }

subject = int(sys.argv[1])

print('subject =', subject)

x_train, y_train, x_test, y_test, df = nn.data(
       v3_data, subject, sigmas, n_data_repeat=100)

func = lambda x: nn.fit_nn(x, x_train, y_train, x_test, y_test)

while True:
	run_hyperopt(func, space, filename='trials{:02}.p'.format(subject))
