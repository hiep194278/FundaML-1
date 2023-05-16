import numpy as np
import random

class DataReader:
    def __init__(self, data_path, batch_size):
        self._batch_size = batch_size
        
        with open(data_path) as f:
            d_lines = f.read().splitlines()
        
        self._data = []
        self._labels = []
        self._sentence_lengths = []
        self._final_token = []

        for data, line in enumerate(d_lines):
            feature = line.split('<fff>')
            label, doc_id, sentence_length = int(feature[0]), int(feature[1]), int(feature[2])
            data = feature[3].split()
            
            self._data.append(data)
            self._labels.append(label)
            self._sentence_lengths.append(sentence_length)
            self._final_token.append(data[-1])

        self._data = np.array(self._data)
        self._labels = np.array(self._labels)
        self._sentence_lengths = np.array(self._sentence_lengths)
        self._final_token = np.array(self._final_token)

        self._num_epoch = 0
        self._batch_ID = 0

    def next_batch(self):
        start = self._batch_ID * self._batch_size
        end = start + self._batch_size
        self._batch_ID += 1

        if end + self._batch_size > len(self._data):
            self._num_epoch += 1
            self._batch_ID = 0
            indices = list(range(len(self._data)))
            random.seed(2023)
            random.shuffle(indices)
            self._data = self._data[indices]
            self._labels = self._labels[indices]
            self._sentence_lengths = self._sentence_lengths[indices]
            self._final_token = self._final_token[indices]
        
        return self._data[start:end], self._labels[start:end], self._sentence_lengths[start:end], self._final_token[start:end]