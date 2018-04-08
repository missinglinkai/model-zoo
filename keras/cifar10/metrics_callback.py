# -- coding: utf8 --

import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict, average=None)
        _val_recall = recall_score(val_targ, val_predict, average=None)
        _val_precision = precision_score(val_targ, val_predict, average=None)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print(' — val_f1: {}'.format(_val_f1))
        print(' — val_precision: {}'.format(_val_precision))
        print(' — val_f1: {} — val_precision: {} — val_recall {}'.format(_val_f1, _val_precision, _val_recall))
        return


class IntervalEvaluation(Callback):
    def __init__(self, callback):
        super(Callback, self).__init__()
        self.ml_callback = callback
        self.val_auc = []
        self.x_axis = []

    def on_epoch_end(self, epoch, logs={}):
        self.X_val, self.y_val, _, _ = self.validation_data

        y_pred = self.model.predict_proba(self.X_val, verbose=0)
        score = roc_auc_score(self.y_val, y_pred)
        self.val_auc.append(score)
        self.x_axis.append(epoch)
        print("interval evaluation - epoch: {:d} - score: {:.6f}".format(epoch, score))
        pass

    def on_train_end(self, epoch, logs=None):
        x = self.ml_callback.calculate_weights_hash(self.model)
        self.ml_callback.send_chart(name='AUC',
                                    x_values=self.x_axis, y_values=self.val_auc,
                                    x_legend='epoch', y_legends='AUC',
                                    scope='test', type='line', model_weights_hash=x)
        pass
