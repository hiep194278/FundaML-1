from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from collections import defaultdict
import numpy as np

WORD_IDF_FILE = '../datasets/20news-bydate/words_idfs.txt'

def load_data(data_path):

    # Map word index to its td_idf value
    def sparse_to_dense(sparse_r_d, vocab_size):
        r_d = [0.0 for _ in range(vocab_size)]
        indices_tfidfs = sparse_r_d.split()

        for index_tfidf in indices_tfidfs:
            index = int(index_tfidf.split(':')[0])
            tdidf = float(index_tfidf.split(':')[1])
            r_d[index] = tdidf

        return np.array(r_d)

    with open(data_path) as f:
        d_lines = f.read().splitlines()

    with open('../datasets/20news-bydate/words_idfs.txt') as f:
        vocab_size = len(f.read().splitlines())

    _data = []
    _labels = []
    _label_count = defaultdict(int)

    for data_id, data in enumerate(d_lines):
        features = data.split('<fff>')
        label, doc_id = int(features[0]), int(features[1])
        _label_count[label] += 1
        r_d = sparse_to_dense(sparse_r_d=features[2], vocab_size=vocab_size)
        _data.append(r_d)
        _labels.append(label)

    return np.array(_data), np.array(_labels)

def clustering_with_KMeans():
    data, labels  = load_data(data_path='../datasets/20news-bydate/20news-full-tf_idf.txt')

    # Use csr_matrix to create a sparse matrix with efficient row slicing
    X = csr_matrix(data)
    print("=========")

    kmeans = KMeans(
        n_clusters = 20,
        init='k-means++',
        n_init=5, # number of times that kmeans runs with differently initialized centroids
        tol=1e-3, # threshold for acceptable minimum error decrease
        random_state=2018 # set to get deterministic results
    ).fit(X)

    labels = kmeans.labels_
    print(labels)
    
def compute_accuracy(predicted_y, expected_y):
    matches = np.equal(predicted_y, expected_y)
    accuracy = np.sum(matches.astype(float))/expected_y.size
    return accuracy

def classifying_with_linear_SVMs():
    train_X, train_y = load_data(data_path='../datasets/20news-bydate/20news-train-tf_idf.txt')

    classifier = LinearSVC(
        C=10.0, # penalty coefficient
        tol=0.001, # tolerance for stopping criteria
        verbose=True # whether to print out logs or not
    )

    classifier.fit(train_X, train_y)

    test_X, test_y = load_data(data_path='../datasets/20news-bydate/20news-test-tf_idf.txt')
    predicted_y = classifier.predict(test_X)
    accuracy = compute_accuracy(predicted_y = predicted_y, expected_y = test_y)
    print("Accuracy:", accuracy)

def classifying_with_kernel_SVMs():
    train_X, train_y = load_data(data_path='../datasets/20news-bydate/20news-train-tf_idf.txt')

    classifier = SVC(
        C=50.0,
        kernel = 'rbf',
        gamma = 0.1,
        tol=0.001,
        verbose=True
    )

    classifier.fit(train_X, train_y)
    
    test_X, test_y = load_data(data_path='../datasets/20news-bydate/20news-test-tf_idf.txt')
    predicted_y = classifier.predict(test_X)
    accuracy = compute_accuracy(predicted_y=predicted_y, expected_y=test_y)
    print("Accuracy:", accuracy)

if __name__ == '__main__':
    # clustering_with_KMeans()
    classifying_with_linear_SVMs()
    # classifying_with_kernel_SVMs()
    