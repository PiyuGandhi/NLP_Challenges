# Piyush Gandhi NLP Submission Mesh


## Flipkart Query

Tfidf Vectorizer + Linear SVC with hyperparameter tuning is used.

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

