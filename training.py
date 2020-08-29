#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 15:38:12 2020

@author: mkhoa
"""

import pandas as pd
import re
import nltk
import pickle

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Load data
df = pd.read_csv('train.csv', index_col='Unnamed: 0')

# Download stopwords
# nltk.download('stopwords')
stop_words = stopwords.words('english')

#Function to clean data
def preprocessor(text):
    """ Return a cleaned version of text
        
    """   
    # Remove HTML markup
    text = re.sub('<[^>]*>', '', text)
    # Remove emoticons
    text = re.sub('(?::|;|=)(?:-)?(?:\)|\(|D|P)', '', text)
    # Remove any non-word character and digit
    text = re.sub('[^A-Za-z ]+', '', text)
    # Also Convert to lower case
    text = (re.sub('[\W]+', ' ', text.lower()))    
    return text


def tokenizer_porter(text):
    """Split a text into list of words and apply stemming technic
    
    """
    porter = PorterStemmer()
    return [porter.stem(word) for word in text.split()]\
          
def convert(sentiment):
    """Convert from 5 sentiment to 3 sentiment
    
    """
    if sentiment < 2:
        sentiment = 0 # Negative
    if sentiment == 2:
        sentiment = 1 # Neutral
    if sentiment > 2:
        sentiment = 2 # Positive
    
    return sentiment
    
# Convert
df['converted_sentiment'] = df['Sentiment'].apply(convert)
df['preprocessed'] = df['Phrase'].apply(preprocessor)
df['preprocessed'] = df['preprocessed'].apply(tokenizer_porter)

y1 = df['Sentiment']
y2 = df['converted_sentiment']
X = df['Phrase']

# Training model on non-converted Sentiment
X_train, X_test, y_train, y_test = train_test_split(X, y1, test_size=0.2, random_state=0)
# tfid = TfidfVectorizer(stop_words=stop_words, tokenizer=tokenizer_porter, preprocessor=preprocessor)
count = CountVectorizer(stop_words=stop_words, tokenizer=tokenizer_porter, preprocessor=preprocessor)

# Construct pipeline
clf = Pipeline([('vect', count), ('clf', LogisticRegression(random_state=0, max_iter=400))])
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)

#Print Result
print('accuracy:',accuracy_score(y_test,prediction))
print('confusion matrix:\n',confusion_matrix(y_test,prediction))
print('classification report:\n',classification_report(y_test,prediction))

# df_test = pd.DataFrame({'Text': X_test, 'Actual': y_test, 'Predict': prediction})
# df_test = df_test[df_test['Actual'] != df_test['Predict']]
# df_test['processed'] = df_test['Text'].apply(preprocessor)
# df_test['processed'] = df_test['processed'].apply(tokenizer_porter)

# Save trained model to disk
pickle.dump(clf, open('model1.sav', 'wb')) 

# Train model on converted sentiment
X_train, X_test, y_train, y_test = train_test_split(X, y2, test_size=0.2, random_state=0)
# tfid = TfidfVectorizer(stop_words=stop_words, tokenizer=tokenizer_porter, preprocessor=preprocessor)
count = CountVectorizer(stop_words=stop_words, tokenizer=tokenizer_porter, preprocessor=preprocessor)

# Construct pipeline
clf = Pipeline([('vect', count), ('clf', LogisticRegression(random_state=0, max_iter=400))])
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)

#Print Result
print('accuracy:',accuracy_score(y_test,prediction))
print('confusion matrix:\n',confusion_matrix(y_test,prediction))
print('classification report:\n',classification_report(y_test,prediction))

# df_test = pd.DataFrame({'Text': X_test, 'Actual': y_test, 'Predict': prediction})
# df_test = df_test[df_test['Actual'] != df_test['Predict']]
# df_test['processed'] = df_test['Text'].apply(preprocessor)
# df_test['processed'] = df_test['processed'].apply(tokenizer_porter)

# Save trained model to disk
pickle.dump(clf, open('model2.sav', 'wb')) 




