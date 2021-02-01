import numpy as np
from sklearn.naive_bayes import MultinomialNB


class Naivebayse:
    def __init__(self, classes):
        self.models = {}
        self.classes = classes
        for cls in self.classes:
            model = MultinomialNB()
            self.models[cls] = model

    def _fit(self, train_x, train_y):
        for idx, cls in enumerate(self.classes):
            class_labels = train_y.iloc[:, idx]
            self.models[cls].fit(train_x, class_labels)

    def fit_and_validate(self, train_x, train_y, validate_x, validate_y):
        self._fit(train_x, train_y)
        return self._predict(validate_x), None

    def _predict(self, test_x):
        predictions = np.zeros((test_x.shape[0], len(self.classes)))
        for idx, cls in enumerate(self.classes):
            predictions[:, idx] = self.models[cls].predict(test_x)
        return predictions

    def predict_prob(self, test_x):
        probs = np.zeros((test_x.shape[0], len(self.classes)))
        for idx, cls in enumerate(self.classes):
            probs[:, idx] = self.models[cls].predict_proba(test_x)[:, 1]
        return probs