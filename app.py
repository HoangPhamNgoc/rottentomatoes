# -*- coding: utf-8 -*-
import pickle

# Load saved model
clf = pickle.load(open('model.sav', 'rb'))

#predict function
def predict(text):
    return clf.predict([text])
