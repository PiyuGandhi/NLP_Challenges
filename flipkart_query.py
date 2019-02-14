# Importing Libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import re
import numpy as np


x_train = []
y_train = []


def words(text):
    """Get Words out of a string.

    Arguments:
        text {str} -- Text

    Returns:
        [list] -- Words
    """

    return re.findall(r'(?:[a-zA-Z]+[a-zA-Z\'\-]?[a-zA-Z]|[a-zA-Z]+)', text)


def load_data(filename='training.txt'):
    """Loads data from a file

    Keyword Arguments:
        filename {str} -- Name of the file to load data from (default: {'training.txt'})
    Returns:
        [list], [list] -- training features, training labels

    """
    count = 0
    with open(filename, 'r') as f:
        for line in f:
            count += 1
            if count == 1:
                continue
            x = [a for a in line.rstrip().split("\t")]
            sen = " ".join(word for word in words(x[0]))
            x_train.append(sen)
            y_train.append(x[1])
    return x_train, y_train


def preprocess_data(features, labels):
    """Preprocesses data and returns vectors and vectorizer

    Arguments:
        features {[list]} -- features of data
        labels {[list]} -- labels of data
    """
    # Step 1: Convert to numpy arrays
    x = np.array(features)
    y = np.array(labels)

    return x, y


def get_model(pipeline={'tfidf': TfidfVectorizer(), 'clf': LinearSVC(C=10.0, dual=False, tol=1e-05)}):
    """Gets Model from Pipeline

    Keyword Arguments:
        pipeline {dict} -- model pipleine to follow (default: {{'tfidf': TfidfVectorizer(), 'clf': LinearSVC()}})
    """
    model = Pipeline([
        (variable, pipeline[variable]) for variable in pipeline
    ])
    return model


if __name__ == '__main__':
    # Step 1: Load data from file
    features, labels = load_data(filename='training.txt')

    # Step 2: Preprocess data
    x_train, y_train = preprocess_data(features, labels)

    # Step 3: Get Model
    model_dict = {
        # Since the main keyword which will determine the label can (and most probably)
        # appears in less count (i.e. Some words are important than others), I decided
        # to go with Tfidf as it inverses the term frequency wrto. document frequency.
        # Since training and test aren't in same shape, we have to include the vectorizer
        # in the model pipeline
        'tfidf': TfidfVectorizer(),

        # LinearSVC classifier because Support Vector Classifiers have been known to
        # perform well in small data (personal experience) and hyperparameters obtained
        # from tuning.
        'clf': LinearSVC(C=10.0, dual=False, tol=1e-05)
    }
    model = get_model(pipeline=model_dict)

    # Step 3: Training the model
    model.fit(x_train, y_train)

    # Step 4: Load test data and predict
    test = []
    for i in range(int(input())):
        x = input()
        sen = " ".join(word for word in words(x))
        test.append(x)

    predicted = model.predict(np.array(test))
    for i in predicted:
        print(i)
