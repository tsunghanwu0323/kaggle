from sklearn.metrics import accuracy_score, classification_report
from module.model import Naivebayse

class Trainer:
    def __init__(self, config, logger, classes, pretrained_embedding):
        self.config = config
        self.logger = logger
        self.classes = classes
        self.pretrained_embedding = pretrained_embedding
        self._create_model(classes)

    def _create_model(self, classes):
        if self.config['model_name'] == 'naivebayse':
            self.model = Naivebayse(classes)
        else:
            self.logger.warning("Model type {} is not supported.".format(self.config['model_name']))

    def _metrics(self, predictions, labels):
        accuracy = accuracy_score(labels, predictions)
        cls_report = classification_report(labels, predictions, zero_division=1)
        return accuracy, cls_report

    def fit_and_validate(self, train_x, train_y, validate_x, validate_y):
        predictions, history = self.model.fit_and_validate(train_x, train_y, validate_x, validate_y)
        accuracy, cls_report = self._metrics(predictions, validate_y)
        return self.model, accuracy, cls_report, history
