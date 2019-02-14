# Piyush Gandhi NLP Submission Mesh


## Flipkart Query

Tfidf Vectorizer + Linear SVC with hyperparameter tuning is used.

Hackerrank Score - 25/30.

Since the main keyword which will determine the label can (and most probably)
appears in less count (i.e. Some words are important than others), I decided
to go with Tfidf as it inverses the term frequency wrto. document frequency.
Since training and test aren't in same shape, we have to include the vectorizer
in the model pipeline.

LinearSVC classifier because Support Vector Classifiers have been known to
perform well in small data (personal experience) and hyperparameters obtained
from tuning.

Pros:

1. Fast Solution
2. Gets good accuracy
3. Easily Deployable

Cons:

1. Not much flexible (limited to training corpus which is very less)

Further and better Solution:

1. Use pretrained vector embeddings like glove or fasttext.


## To be or what to be

Tfidf Vectorizer + Logistic Regression is used.

Logistic Regression because multiclass labels and less data.

Hackerrank Score - 45/50.


Pros:

1. Fast Solution
2. Gets good accuracy
3. Easily Deployable

Cons:

1. Not much flexible (limited to training corpus which is very less)
2. Limited Vocabulary

Further and better Solution:

1. Use pretrained vector embeddings like glove or fasttext to directly predict the next word.

2. Use a larger corpus to train embeddings on our own using a word2vec or glove or fasttext architecture.

