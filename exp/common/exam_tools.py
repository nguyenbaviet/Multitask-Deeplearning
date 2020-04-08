import numpy as np
import time
from keras.callbacks import Callback
from deephar.utils import *

def eval_singleClip(model, x_te, action_te, verbose=1):
    start = time.time()
    pred = model.predict(x_te, batch_size=1, verbose=verbose)
    dt = time.time() -start

    pred = pred[-1]
    correct = np.equal(np.argmax(action_te, axis=-1), np.argmax(pred, axis=-1), dtype=np.float)

    score = sum(correct) / len(correct)
    if verbose:
        printc(WARNING, 'Acc for action acc.%: %.1f' %(100 * score))
        printcn('', '\n%d samples in %.1f clips per sec' %(len(x_te), dt, len(x_te) /dt))

    return score

def eval_singleClip_generator(model, datagen, n_actions, verbose=1):
    num_samples = datagen.get_length()

    y_true = np.zeros((num_samples, n_actions))
    y_pred = np.zeros((num_samples, n_actions))
    out = [datagen.get_data(i) for i in range(num_samples)]
    start = time.time()
    for i in range(num_samples):
        # out = datagen.get_data(i)
        y_true[i, :] = out[i]['action']
        y_pred[i, :] = model.predict(np.expand_dims(out[i]['frame'], axis=0))[-1]
    dt = time.time() - start

    correct = np.equal(np.argmax(y_true, axis=-1), np.argmax(y_pred, axis=-1), dtype=np.float)

    score = sum(correct) / len(correct)

    if verbose:
        printc(WARNING, 'Acc for action acc: %.1f' %(100 * score))
        printcn('', '\n%d samples in %.1f sec: %.1f clips per sec' %(num_samples, dt, num_samples /dt))
    return score
class ExamActionEvalCallback(Callback):
    def __init__(self, data, batch_size=1,n_action=4, eval_model=None):
        self.n_action = n_action
        self.data = data
        self.batch_size = batch_size
        self.eval_model = eval_model
        self.score = {}

    def on_epoch_end(self, epoch):
        assert self.eval_model is not None, 'Do not have eval model'

        score = eval_singleClip_generator(self.eval_model, self.data, self.n_action)

        epoch +=1

        self.score[epoch] = score

        printcn(OKBLUE, 'Best score is %.1f at epoch %d' % \
                (100 * self.best_score, self.best_epoch))

    @property
    def best_epoch(self):
        if len(self.score) > 0:
            # Get the key of the maximum value from a dict
            return max(self.score, key=self.score.get)
        else:
            return np.inf

    @property
    def best_score(self):
        if len(self.score) > 0:
            # Get the maximum value from a dict
            return self.score[self.best_epoch]
        else:
            return 0