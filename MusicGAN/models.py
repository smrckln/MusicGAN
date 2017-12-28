from keras.models import Sequential
from keras.layers import Dense
from keras.layers.recurrent import LSTM

class Discriminator(object):
    def __init__(self):
        self.model = None


class GenreClassifier(object):
    def __init__(self):
        self.model = None

    def build_classifier(self, input_shape=(128, 33), label_shape=(-1, 1)):
        model = Sequential()
        model.add(LSTM(units=128, dropout=0.05, recurrent_dropout=0.35, return_sequences=True, input_shape=input_shape))
        model.add(LSTM(units=32, dropout=0.05, recurrent_dropout=0.35, return_sequences=False))
        model.add(Dense(units=label_shape[1], activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        model.summary()

        self.model = model

    def train(self, train_x=None, train_y=None, epochs=100, batch_size=32):
        self.model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size)

    def validate(self, val_x=None, val_y=None, batch_size=32):
        score, accuracy = self.model.evaluate(val_x, val_y, batch_size=batch_size)
        return (score, accuracy)

    def test_eval(self, test_x=None, test_y=None, batch_size=32):
        score, accuracy = self.model.evaluate(test_x, test_y, batch_size=batch_size)
        return (score, accuracy)