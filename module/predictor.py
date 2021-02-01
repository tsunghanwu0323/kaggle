import csv


class Predictor:
    def __init__(self, config, logger, model):
        self.config = config
        self.logger = logger
        self.model = model

    def predict_prob(self, test_x):
        prob = self.model.predict_prob(test_x)
        return prob

    def save_result(self, test_ids, probs):
        with open(self.config['output_path'], 'w') as output_csv_file:
            header = ['id', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
            writer = csv.writer(output_csv_file)
            writer.writerow(header)
            for test_id, prob in zip(test_ids, probs.tolist()):
                writer.writerow([test_id] + prob)
