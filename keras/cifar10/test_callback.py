from keras.callbacks import Callback


class TestCallback(Callback):
    def __init__(self, test_data, callback, datagen, batch_size):
        self.callback = callback
        self.loss_metrics = []
        self.acc_metrics = []
        self.x_test, self.y_test = test_data
        self.datagen = datagen
        self.batch_size = batch_size

    def on_epoch_end(self, batch, logs=None):
        with self.callback.test(self.model):
            # Evaluate model with test data set and share sample prediction results
            evaluation = self.model.evaluate_generator(self.datagen.flow(self.x_test, self.y_test,
                                                                         batch_size=self.batch_size,
                                                                         shuffle=False),
                                                       steps=self.x_test.shape[0] // self.batch_size,
                                                       workers=4)

            loss_index = self.model.metrics_names.index('loss')
            acc_index = self.model.metrics_names.index('acc')
            self.loss_metrics.append(evaluation[loss_index])
            self.acc_metrics.append(evaluation[acc_index])
        print('self.model.metrics_names: {}'.format(self.model.metrics_names))
        print('Model Accuracy = %.2f' % (evaluation[1]))
        print('evaluation: {}'.format(evaluation))

    def on_train_end(self, epoch, logs=None):
        x = self.callback.calculate_weights_hash(self.model)
        self.callback.send_chart(name='Precision Recall',
                                 x_values=self.loss_metrics, y_values=self.acc_metrics,
                                 x_legend='Precision', y_legends='Recall',
                                 scope='test', type='line', model_weights_hash=x)
        pass
