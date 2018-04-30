# -- coding: utf8 --
import logging
import math

import numpy
import numpy as np
from keras.callbacks import Callback


class IntervalEvaluation(Callback):
    def __init__(self, callback, test_generator, class_mapping, num_predictions):
        super(Callback, self).__init__()
        self.ml_callback = callback
        self.test_generator = test_generator
        self.class_mapping = class_mapping
        self.num_predictions = num_predictions
        self.test_items_x = None
        self.test_items_y = None

        self.val_recalls = []
        self.val_precisions = []

        for _ in range(len(self.test_generator)):
            data = next(self.test_generator)
            if self.test_items_x is None:
                self.test_items_x = data[0]
                self.test_items_y = data[1]
            else:
                self.test_items_x = numpy.concatenate((self.test_items_x, data[0]))
                self.test_items_y = numpy.concatenate((self.test_items_y, data[1]))

    def on_epoch_end(self, epoch, logs={}):

        predict_gen = self.model.predict_generator(self.test_generator,
                                                   steps=len(self.test_generator),
                                                   workers=4)

        confusion_matrix = np.zeros((len(self.class_mapping), len(self.class_mapping)), dtype='int32')

        for predict_index, predicted_y in enumerate(predict_gen):
            actual_label_index = numpy.argmax(self.test_items_y[predict_index])
            predicted_label_index = numpy.argmax(predicted_y)

            confusion_matrix[actual_label_index, predicted_label_index] += 1

            actual_label = self.class_mapping[actual_label_index]
            predicted_label = self.class_mapping[predicted_label_index]

        logging.info(confusion_matrix)
        # per class accuracy
        for k in range(len(self.class_mapping)):
            count = confusion_matrix[k].sum()
            precision = confusion_matrix[k, k] * 1. / confusion_matrix[:, k].sum()
            recall = confusion_matrix[k, k] * 1. / confusion_matrix[k, :].sum()
            logging.info('class {} {:>16}:\t count {}\t precision {:.2}%%\t recall {:.2}%%'.format(
                k, self.class_mapping[k], count, precision, recall))

        true_pos = np.diag(confusion_matrix)
        precision = np.sum(true_pos * 1.0 / np.sum(confusion_matrix, axis=0))
        recall = np.sum(true_pos * 1.0 / np.sum(confusion_matrix, axis=1))
        if not (math.isnan(precision) or math.isinf(precision) or math.isnan(recall) or math.isinf(recall)):
            self.val_recalls.append(recall)
            self.val_precisions.append(precision)
        else:
            logging.info('skipping precision recall due to NAN')

        logging.info('true_pos: {} Precision {} Recall {}'.format(np.sum(true_pos), precision, recall))

        self.ml_callback.send_chart(name='Precision Recall {}'.format(epoch),
                                    x_values=self.val_recalls, y_values=self.val_precisions,
                                    x_legend='Recall', y_legends='Precision',
                                    scope='test', type='line', experiment_id=self.ml_callback.experiment_id)

    def on_train_end(self, epoch, logs=None):
        logging.info('sending graph precision {} recall {}'.format(self.val_precisions, self.val_recalls))
        self.ml_callback.send_chart(name='Precision Recall',
                                    x_values=self.val_recalls, y_values=self.val_precisions,
                                    x_legend='Recall', y_legends='Precision',
                                    scope='test', type='line', experiment_id=self.ml_callback.experiment_id)
