import string
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class Preprocessor(object):
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.classes = self.config['classes']
        self._load_data()

    @staticmethod
    def clean_text(text):
        text = text.strip().lower().replace('\n', '')
        # tokenization
        words = text.split()
        # filter punctuation
        filter_table = str.maketrans('', '', string.punctuation)
        clean_words = [w.translate(filter_table) for w in words if len(w.translate(filter_table))]

        return clean_words

    def _parse(self, data, is_test=False):
        '''

        :param data:
        :param is_test:
        :return:
            tokenized_input (np.array)
            one_hot_label (np.array)
        '''
        X = data[self.config['input_text_column']].apply(Preprocessor.clean_text).values
        Y = None
        if not is_test:
            Y = data.drop([self.config['input_id_column'], self.config['input_text_column']], 1)
        else:
            Y = data.id.values
        return X, Y

    def _load_data(self):
        data = pd.read_csv(self.config['input_trainset'])
        # fill na data with 'unknown'
        data[self.config['input_text_column']].fillna("unknown", inplace=True)
        self.data_x, self.data_y = self._parse(data)
        self.train_x, self.validate_x, self.train_y, self.validate_y = train_test_split(
            self.data_x,
            self.data_y,
            test_size=self.config['split_ratio'],
            random_state=self.config['random_seed']
        )

        test_data = pd.read_csv(self.config['input_testset'])
        test_data[self.config['input_text_column']].fillna("unknown", inplace=True)
        self.test_x, self.test_ids = self._parse(test_data, is_test=True)

    def process(self):
        input_convertor = self.config.get('input_convertor', None)
        # label_convertor = self.config.get('label_convertor', None)
        data_x, data_y, train_x, train_y, validate_x, validate_y, test_x = self.data_y, self.data_y, self.train_x, self.train_y, self.validate_x, self.validate_y, self.test_x

        if input_convertor == 'count_vectorization':
            train_x, validate_x, test_x = self.count_vectorization(train_x, validate_x, test_x)
        elif input_convertor == 'tfidf_vectorization':
            train_x, validate_x, test_x = self.tfidf_vectorization(train_x, validate_x, test_x)

        return data_x, data_y, train_x, train_y, validate_x, validate_y, test_x

    def count_vectorization(self, train_x, validate_x, test_x):
        vectorizer = CountVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x)
        vectorized_train_x = vectorizer.fit_transform(train_x)
        vectorized_validate_x = vectorizer.transform(validate_x)
        vectorized_test_x = vectorizer.transform(test_x)
        return vectorized_train_x, vectorized_validate_x, vectorized_test_x

    def tfidf_vectorization(self, train_x, validate_x, test_x):
        vectorizer = TfidfVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x)
        vectorized_train_x = vectorizer.fit_transform(train_x)
        vectorized_validate_x = vectorizer.transform(validate_x)
        vectorized_test_x = vectorizer.transform(test_x)
        return vectorized_train_x, vectorized_validate_x, vectorized_test_x
