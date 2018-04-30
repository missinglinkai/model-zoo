from keras.callbacks import Callback


class TestCallback(Callback):
    def __init__(self, test_generator, callback):
        self.callback = callback
        self.test_generator = test_generator

    def on_epoch_end(self, batch, logs=None):
        print('starting test on epoch')
        with self.callback.test(self.model):
            # Evaluate model with test data set and share sample prediction results
            self.model.evaluate_generator(self.test_generator,
                                                       steps=len(self.test_generator),
                                                       workers=4)


