from keras import backend as K
from keras.layers import Dense
import numpy as np
import pickle

from hyperopt import Trials, tpe, fmin

def reset_weights(model):

    session = K.get_session()

    for layer in model.layers:
        if isinstance(layer, Dense):
            old = layer.get_weights()
            layer.kernel.initializer.run(session=session)
            layer.bias.initializer.run(session=session)
            
            
def shuffle_weights(model, weights=None):
    """Randomly permute the weights in `model`, or the given `weights`.
    This is a fast approximation of re-initializing the weights of a model.
    Assumes weights are distributed independently of the dimensions of the weight tensors
      (i.e., the weights have the same distribution along each dimension).
    :param Model model: Modify the weights of the given model.
    :param list(ndarray) weights: The model's weights will be replaced by a random permutation of these weights.
      If `None`, permute the model's current weights.
    """

    if weights is None:
        weights = model.get_weights()
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    # Faster, but less random: only permutes along the first dimension
    # weights = [np.random.permutation(w) for w in weights]
    model.set_weights(weights)


def run_hyperopt(func, space, filename='trials.p', trials_step=2):
    trials_step = 25
    max_trials = 5  # initial number of trials before save
    
    try:
        trials = pickle.load(open(filename, 'rb'))
        print('loading saved trials...')
        max_trials = len(trials.trials) + trials_step
        print('running trials {} to {}...'.format(len(trials.trials), max_trials))
    except:
        trials = Trials()
        
    fmin(func, space, algo=tpe.suggest, max_evals=max_trials, trials=trials)
    
    print('best score: {:.5}'.format(-np.min(trials.losses())))

    with open(filename, 'wb') as f:
        pickle.dump(trials, f)
