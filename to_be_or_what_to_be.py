# Importing Libraries
import fileinput
import re
from nltk.tokenize import wordpunct_tokenize
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline, make_pipeline

be_forms = ['am', 'are', 'were', 'was', 'is', 'been', 'being', 'be']
substitute = '----'


def find_targets(tokens):
    """Find targets i.e be_forms in tokens.

    Arguments:
        tokens {list} -- tokens

    Returns:
        [list] -- list of tokens
    """

    return [t for t in tokens if t in be_forms]


def remove_targets(tokens):
    """ Replace targets with a substitute in a tokenized text"""
    return [substitute if t in be_forms else t for t in tokens]


def create_windows(tokens, window_size=5):
    """Creating context windows

    Arguments:
        tokens {list} -- tokens of text

    Keyword Arguments:
        window_size {int} -- context window size (default: {5})

    Returns:
        contexts {list of lists} -- list of lists containing 2*window_size
    """

    contexts = []
    for i, word in enumerate(tokens):
        if word == substitute:
            window = tokens[i-window_size:i] + tokens[i+1:i+window_size+1]
            window = ' '.join(window)
            contexts.append(window)
    return contexts


def preprocess_data(text):
    """Preprocessing text

    Arguments:
        text {Raw text} -- text

    Returns:
        contexts[list], y[list], le[label encoder] -- features, labels, label encoder
    """
    # Step 1: Filter required text using regex
    # used to remove headers, toc, etc.
    title_match_regex = '\n{3,}\s+THE SECRET CACHE\n{3,}.*'
    corpus = re.search(title_match_regex, text, flags=re.M+re.S).group()
    corpus = corpus.replace('\n', ' ')
    corpus = re.sub(r' {2,}', ' ', corpus)  # replace multiple blanks by one
    # remove consecutive hyphens that we'll as a tag for the be verb
    corpus = corpus.replace('----', '')

    # Step 2: Tokenize
    tokens = wordpunct_tokenize(corpus)

    # Step 3: Find and remove targets from features
    targets = find_targets(tokens)
    tokens = remove_targets(tokens)

    # Step 4: Create context
    contexts = create_windows(tokens)

    # Step 5: Encode Targets
    counts = np.unique(targets, return_counts=True)

    le = LabelEncoder()
    y = le.fit_transform(targets)

    return contexts, y, le


def load_file(filepath='corpus.txt'):
    """Load training file

    Keyword Arguments:
        filepath {str} -- training filepath (default: {'corpus.txt'})

    Returns:
        text {str} -- Raw text
    """

    text = open(filepath, 'r', encoding='utf-8-sig').read()
    return text


def get_model(pipeline_dict={'vec': TfidfVectorizer(), 'clf': LogisticRegression()}):
    """Make model using Pipeline

    Keyword Arguments:
        pipeline_dict {dict} -- model pipeline dictionary (default: {{'vec': TfidfVectorizer(), 'clf': LogisticRegression()}})

    Returns:
        model [Pipeline] -- model (Pipeline object)
    """

    model = Pipeline([
        (variable, pipeline_dict[variable]) for variable in pipeline_dict
    ])
    return model


if __name__ == '__main__':
    # Step 1: Load Raw data
    text = load_file(filepath='corpus.txt')

    # Step 2: Preprocess data
    features, labels, encoder = preprocess_data(text)

    # Step 3: Get Model
    model_config = {
        # Tfidf because near context words matter more than far context
        'vec': TfidfVectorizer(),
        # Logistic Regression because multiclass labels and less data
        'clf': LogisticRegression()
    }
    model = get_model(pipeline_dict=model_config)

    # Step 3: Training
    model.fit(features, labels)

    # Step 4: Prediction
    test = fileinput.input()
    n = int(test.readline())
    test_text = test.readline()
    test_tokenization = wordpunct_tokenize(test_text)
    test_features = create_windows(test_tokenization)
    prediction = encoder.inverse_transform(model.predict(test_features))
    print('\n'.join(prediction))
