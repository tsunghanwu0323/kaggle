import io
import string
import pandas as pd
import numpy as np
from tensorflow import keras
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
        # fill na test data with 'unknown'
        test_data[self.config['input_text_column']].fillna("unknown", inplace=True)
        self.test_x, self.test_ids = self._parse(test_data, is_test=True)

    def process(self):
        input_convertor = self.config.get('input_convertor', None)
        data_x, data_y, train_x, train_y, validate_x, validate_y, test_x = self.data_y, self.data_y, self.train_x, self.train_y, self.validate_x, self.validate_y, self.test_x

        if input_convertor == 'count_vectorization':
            train_x, validate_x, test_x = self.count_vectorization(train_x, validate_x, test_x)
        elif input_convertor == 'tfidf_vectorization':
            train_x, validate_x, test_x = self.tfidf_vectorization(train_x, validate_x, test_x)
        elif input_convertor == 'nn_vectorization':
            train_x, validate_x, test_x = self.nn_vectorization(train_x, validate_x, test_x)

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

    def nn_vectorization(self, train_x, validate_x, test_x):
        self.word2ind = {}
        self.ind2word = {}
        special_tokens = ['<pad>', '<unk>']

        pretrained_embedding = self.config.get('pretrained_embedding', None)

        if pretrained_embedding is not None:
            word2embedding = Preprocessor.load_vectors(pretrained_embedding)
            vocabs = special_tokens + list(word2embedding.keys())
            vocabs_size = len(vocabs)
            self.embedding_matrix = np.zeros((vocabs_size, self.config['embedding_dim']))
            for token in special_tokens:
                word2embedding[token] = np.random.uniform(low=-1, high=1,
                                                          size=(self.config['embedding_dim'],))
            for idx, word in enumerate(vocabs):
                self.word2ind[word] = idx
                self.ind2word[idx] = word
                self.embedding_matrix[idx] = word2embedding[word]
        else:
            def add_word(word2ind, ind2word, word):
                if word in word2ind:
                    return
                ind2word[len(word2ind)] = word  # add word to ind2word
                word2ind[word] = len(word2ind)

            for token in special_tokens:
                add_word(self.word2ind, self.ind2word, token)

            for sent in train_x:
                for word in sent:
                    add_word(self.word2ind, self.ind2word, word)

        def get_ids(data_x):
            x_ids = []
            for sent_ in data_x:
                ind_sent = [self.word2ind.get(i, self.word2ind['<unk>']) for i in sent_]
                x_ids.append(ind_sent)
            x_ids = np.array(x_ids)

            return x_ids
        train_x_ids = get_ids(train_x)
        validate_x_ids = get_ids(validate_x)
        test_x_ids = get_ids(test_x)

        train_x_ids = keras.preprocessing.sequence.pad_sequences(train_x_ids, maxlen=self.config['max_len'], padding='post', value=self.word2ind['<pad>'])
        validate_x_ids = keras.preprocessing.sequence.pad_sequences(validate_x_ids, maxlen=self.config['max_len'], padding='post', value=self.word2ind['<pad>'])
        test_x_ids = keras.preprocessing.sequence.pad_sequences(test_x_ids, maxlen=self.config['max_len'], padding='post', value=self.word2ind['<pad>'])

        return train_x_ids, validate_x_ids, test_x_ids

    @staticmethod
    def load_vectors(fname):
        fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        data = {}
        for line in fin:
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = np.array(list(map(float, tokens[1:])))
        return data
