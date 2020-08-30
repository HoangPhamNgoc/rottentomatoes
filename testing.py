# -*- coding: utf-8 -*-
import pandas as pd
import re
import pickle
from pathlib import Path

script_location = Path(__file__).absolute().parent

def load_model():
    model1 = pickle.load(open(script_location / 'model1.sav', 'rb'))
    return model1

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

def predict(phrase):
    return int(model1.predict([phrase]))

model1 = load_model()
def main():  
    df_test = pd.read_csv(script_location / 'test.csv')
    df_test['Sentiment'] = df_test['Phrase'].apply(predict)
    df_test.to_csv('submission.csv')
    print('Completed')

if __name__ == '__main__':
    main()