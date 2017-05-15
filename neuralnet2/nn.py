import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout

from keras import backend as K
import keras_utils as k_utils
import conf_utils
import pandas as pd
from hyperopt import STATUS_OK
import matplotlib.pyplot as plt
from keras.regularizers import l2
import pdb

from keras import callbacks as kcb
import seaborn as sns


def subject_data(df, subject, task='B'):
    return df.loc[(df['task'] == task) & (df['subject'] == subject)]


def _unnorm_gauss(x, mu, precision):
    return np.exp(-(x - mu)**2 * precision / 2)

def _pval_correction(pvals):
    if sum(pvals[:-1]) > 1:
        return pvals-(sum(pvals)-1)
    else:
        return pvals

def _generate_spikes(s, sigmas, n_input_neurons=50, baseline=.025, tc_precision=.01, spref_max=40):
    
    spref = np.linspace(-spref_max, spref_max, n_input_neurons)
    n_trials = len(s)

    k = sum(np.exp(-spref**2 * tc_precision / 2))

    if np.isscalar(sigmas):
        sigmas = sigmas * np.ones(n_trials)
    elif len(sigmas) != n_trials:
        sigmas = np.random.choice(sigmas, size=n_trials)

    gains = 1 / (tc_precision * sigmas**2 * k)

    lam = baseline + gains[:, np.newaxis] * _unnorm_gauss(
        s[:, np.newaxis], spref, tc_precision)
    # lambdas taken from gaussian tuning curves

    spikes = np.random.poisson(lam)
    # spikes = lam

    return spikes

    # adapt this code for optimal posterior, maybe for non-zero baseline
    #     if baseline == 0:
    #         ar1 = np.sum(data['spikes'], 1) * tc_precision
    #         br1 = np.sum(data['spikes'] * spref, 1) * tc_precision

    #         cat_var = np.array(cat_SD)**2

    #         data['opt_p'] = 1 / (1 + np.sqrt(
    #             (1 + cat_var[0] * ar1) / (1 + cat_var[1] * ar1)) * np.exp(-0.5 * (
    #                 (cat_var[0] - cat_var[1]) * br1**2) / (
    #                     (1 + cat_var[0] * ar1) * (1 + cat_var[1] * ar1))))
    #         data['opt_d'] = -np.log(1 / data['opt_p'] - 1)


def data(df, sigmas, n_train_trials=2160, baseline=.025, n_input_neurons=50, spref_max=40, shuffle=True, n_test_trials=2160):

    if shuffle:
        train_trials = np.random.choice(len(df), size=n_train_trials)
        df_train = df.iloc[train_trials]
    else:
        df_train = df

    x_train = _generate_spikes(df_train['s'],
                               sigmas[df_train['contrast_id']],
                               baseline=baseline,
                               n_input_neurons=n_input_neurons,
                               spref_max=spref_max)
    y_train = np_utils.to_categorical(df_train['resp'], num_classes=8)

    if n_test_trials == 0:
        return x_train, y_train, df_train
    elif n_test_trials != 0:
        test_trials = np.random.choice(len(df), size=n_test_trials)
        df_test = df.iloc[test_trials]
        x_test = _generate_spikes(df_test['s'],
                                  sigmas[df_test['contrast_id']],
                                  baseline=baseline,
                                  n_input_neurons=n_input_neurons,
                                  spref_max=spref_max)
        y_test = np_utils.to_categorical(df_test['resp'], num_classes=8)
        return x_train, y_train, x_test, y_test, df_train, df_test


# def pred_prob_of_resp(y_true, y_pred, keras=True):
#     if keras:
#         return K.mean(y_pred)
#     else:  # outside of Keras
#         # pdb.set_trace() 
#         return np.mean(np.log(np.sum(y_true * y_pred, axis=1)))


def fit_nn(params, x_train, y_train, x_test, y_test, callbacks=[]):
    n_input_neurons = x_train.shape[1]

    model = Sequential()
    model.add(Dense(int(params['nhu1']),
              input_dim=n_input_neurons,
              activation='relu',
              kernel_regularizer=l2(l=params['l2']),
              bias_regularizer=l2(l=params['l2'])))
    model.add(Dropout(params['dropout']))

    if params['nhu2'] != 0:
        model.add(Dense(int(params['nhu2']),
              activation='relu',
              kernel_regularizer=l2(l=params['l2']),
              bias_regularizer=l2(l=params['l2'])))
        model.add(Dropout(params['dropout']))

    model.add(Dense(8,
              activation='softmax'))  # previously 1, sigmoid

    model.compile(
        loss='categorical_crossentropy',  # previously binary_crossentropy
        optimizer=params['optimizer'],
        metrics=[])

    k_utils.reset_weights(model)

    n_train = int(params['n_train'])

    history = model.fit(x_train[:n_train], y_train[:n_train],
              epochs=params['n_epochs'],
              batch_size=int(params['batch_size']),
              verbose=0,
              callbacks=callbacks,
              validation_data=(x_test, y_test))

    cat_xe_train = model.evaluate(x_train, y_train, verbose=0)
    cat_xe_test = model.evaluate(x_test, y_test, verbose=0)
    
    return model, history
    # for hyperopt: {'loss': cat_xe, 'status': STATUS_OK}, model


def plot_comparison(df, spikes, model, ax, n_bins=7, comparison_score=None, max_n_samples=int(5e4), plot_reliabilities=[1,3,5]):
    n_trials = np.min([len(df), max_n_samples])
    trials = np.random.choice(len(df), size=n_trials, replace=False)

    df = df.iloc[trials]
    spikes = spikes[trials]
    
    edges, centers = conf_utils.quantile_bins(bins=n_bins)

    df = df.assign(s_bin=pd.cut(df['s'],
                                np.concatenate(([-np.inf], edges, [np.inf])),
                                labels=False))

    if model is not None:
        resp_pred = np.empty(len(spikes))
        for i, pvals in enumerate(model.predict(spikes)):
            resp_pred[i] = np.where(np.random.multinomial(1, pvals=_pval_correction(pvals)))[0][0]

        df = df.assign(resp_pred=resp_pred)

    c_s_bins = df.groupby(['contrast_id', 's_bin'])

    mean = c_s_bins['resp'].apply(np.mean)
    sem = c_s_bins['resp'].apply(lambda x: np.std(x)/np.sqrt(len(x)))

    if model is not None:
        mean_pred = c_s_bins['resp_pred'].apply(np.mean)
        sem_pred = c_s_bins['resp_pred'].apply(lambda x: np.std(x)/np.sqrt(len(x)))

    contrast_colors =[
    (0.0784,    0.0471,         0),
    (0.2588,    0.1529,         0),
    (0.4784,    0.3333,    0.0471),
    (0.6980,    0.5098,    0.0941),
    (0.8471,    0.6667,    0.1569),
    (1.0000,    0.8235,    0.2196)]
    
    
    for c in plot_reliabilities:     
               
        ax.errorbar(range(n_bins), mean[c], sem[c], ecolor=contrast_colors[c], capsize=6, capthick=2, elinewidth=2, fmt='none')

        if model is not None:
            ax.fill_between(range(n_bins), mean_pred[c] - sem_pred[c],
                            mean_pred[c] + sem_pred[c], color=contrast_colors[c], alpha=.5)
                
    # NN_score = -pred_prob_of_resp(
    #     np_utils.to_categorical(df['resp'], num_classes=8), prob_pred, keras=False)
    NN_score = model.evaluate(spikes, np_utils.to_categorical(df['resp'], num_classes=8), verbose=0)

    plt.text(0.05, 0.95, 'NN x-ent: {:.3}'.format(NN_score),
             horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
        
    ax.set_xticks(range(n_bins))
    ax.set_xticklabels(np.round(centers).astype(int))


    if comparison_score is not None:
        i = 0
        for score in comparison_score:
            plt.text(0.05, .88-.07*i, '{} x-ent: {:.3}'.format(score, comparison_score[score]),
                     horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
            i += 1

    ax.set_ylim([-0.5, 7.5])
    ax.invert_yaxis()
    sns.despine()

    # return df


class NBatchLogger(kcb.Callback):
    def __init__(self, batch_interval=10):
        self.batch_interval = batch_interval
        self.batches_seen = 0
        self.losses = []
        self.batches = []
        
    def on_batch_end(self, batch, logs={}):
        self.batches_seen += 1
        if self.batches_seen % self.batch_interval == 0:
            loss = logs.get('loss')
            self.losses.append(loss)
            self.batches.append(self.batches_seen)
