# -*- coding: utf-8 -*-
import pandas as pd
import pickle
from pathlib import Path

script_location = Path(__file__).absolute().parent

def load_model():
    model1 = pickle.load(open(script_location / 'model1.sav', 'rb'))
    return model1

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