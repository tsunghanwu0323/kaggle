from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Conv1D, GlobalAvgPool1D, Dense, Dropout, Flatten, MaxPool1D, Input, Concatenate


class CNN(object):
    def __init__(self, classes, config, pretrained_embedding):
        self.models = {}
        self.classes = classes
        self.num_class = len(classes)
        self.config = config
        self.model = self._build(pretrained_embedding)

    def _build(self, pretrained_embedding):
        model = Sequential()
        if pretrained_embedding is not None:
            model.add(Embedding(self.config['vocab_size'], self.config['embedding_dim'],
                                weights=[pretrained_embedding],
                                input_length=self.config['max_len'], trainable=False))
        else:
            model.add(Embedding(self.config['vocab_size'], self.config['embedding_dim'], embeddings_initializer="uniform",
                            input_length=self.config['max_len'], trainable=True))
        model.add(Conv1D(128, 7, activation='relu', padding='same'))
        model.add(MaxPool1D())
        model.add(Conv1D(256, 5, activation='relu', padding='same'))
        model.add(MaxPool1D())
        model.add(Conv1D(512, 3, activation='relu', padding='same'))
        model.add(MaxPool1D())
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_class, activation=None))
        model.add(Dense(self.num_class, activation='sigmoid'))
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        model.summary()
        return model

    def fit_and_validate(self, train_x, train_y, validate_x, validate_y):
        history = self. model.fit(train_x, train_y,
                                  epochs=self.config['epochs'],
                                  verbose=True,
                                  validation_data=(validate_x, validate_y),
                                  batch_size=self.config['batch_size'])
        predictions = self.predict(validate_x)
        return predictions, history

    def predict_prob(self, test_x):
        return self.model.predict(test_x)

    def predict(self, test_x):
        probs = self.model.predict(test_x)
        return probs >= 0.5
