# -*- coding: utf-8 -*-
import streamlit as st
import re
import pickle
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
import wordcloud as wordcloud
import os, urllib

from collections import Counter
from pathlib import Path
from PIL import Image
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report




script_location = Path(__file__).absolute().parent
# Streamlit encourages well-structured code, like starting execution in a main() function.
def main():
    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    st.title('Movie Review Sentiment Analysis')
    st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Introduction", "Run the app", "Show the source code"])
    if app_mode == "Introduction":
        st.sidebar.success('To continue select "Run the app".')
        intro()
    elif app_mode == "Show the source code":
        st.write('Soure Code: https://github.com/mkhoa/rottentomatoes')
        show_source_code(df)
    elif app_mode == "Run the app":
        run_the_app()
        
# Load saved model
@st.cache(allow_output_mutation=True)
def load_model():
    model1 = pickle.load(open(script_location / 'model1.sav', 'rb'))
    model2 = pickle.load(open(script_location / 'model2.sav', 'rb'))
    return model1, model2

@st.cache(allow_output_mutation=True)
def load_dataframe():
    df = pickle.load(open(script_location / 'DataFrame.sav', 'rb'))
    return df

# Most common word
@st.cache
def most_common(df):    
    vocab = Counter()
    for phrase in df.preprocessed:
        for word in phrase.split(' '):
              vocab[word] += 1
              
    most_common = vocab.most_common(10)
    return most_common

# Download a single file and make its content available as a string.
@st.cache(show_spinner=False)
def get_file_content_as_string(path):
    url = 'https://raw.githubusercontent.com/mkhoa/rottentomatoes/master/' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")

def createWordCloud(df):
    word = []  
    for phrase in df.preprocessed:
        for i in phrase.split(' '):
              word.append(i.lower())   
    occurrences = Counter(word)
    cloud = wordcloud.WordCloud(background_color="white", width=1920, height=1080, min_font_size=8)
    cloud.generate_from_frequencies(occurrences)
    myimage = cloud.to_array()
    plt.imshow(myimage, interpolation = 'nearest')
    plt.axis('off')
    plt.show()    

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

# Predict function by model 1, non-convertred sentiment
def model1_predict(text):
    prediction = model1.predict([text])
    if prediction==0:  
        return 'Negative'
    elif prediction==1:
        return 'Somewhat negative'
    elif prediction==2:
        return 'Neutral'
    elif prediction==3:
        return 'Somewhat positive'
    elif prediction==4:
        return 'Positive'
    
# Predict function
def model2_predict(text):
    prediction = model2.predict([text])
    if prediction==0:  
        return 'Negative'
    elif prediction==1:
        return 'Neutral'
    elif prediction==2:
        return 'Positive'
  
   
# Introduction
def intro():
    st.title('About Dataset')
    image = Image.open(script_location / 'logo.jpg')
    st.image(image)
    st.markdown('''
                The Rotten Tomatoes movie review dataset is a corpus of movie reviews used for sentiment analysis. The dataset contain movie reviews from rottentomatoes.com website and had been labeled for sentiment.
                
                This is a multi-class classification problem, which simply means the data set have more than 2 classes(binary classifier). The five classes corresponding to sentiments:
                ```
                0 - Negative
                1 - Somewhat Negative
                2 - Neutral
                3 - Somewhat Positive
                4 - Positive
                ```
                ''')
    st.markdown('## Data Exploration')
    st.markdown('Let us have a look at first few phrases of the training dataset:')
    st.write(df.sample(10))
    st.markdown('## Number of comments across categories')
    st.write(df['Sentiment'].value_counts())
    sns.barplot(x=df['Sentiment'].value_counts().index, y=df['Sentiment'].value_counts())
    st.pyplot()
    st.markdown('## Most common word in review')
    st.write(most_common(df))
    st.markdown('**Word Cloud**')
    createWordCloud(df)
    st.pyplot()
    
# Function to run the prediction demo   
def run_the_app():
    user_input = st.text_input("Review Text", 'This is an example review for movie')
    st.write('Prediction by model 1, training by using original sentiment')
    st.write(model1_predict(user_input))
    st.write('Prediction by model 2, training by using converted sentiment')
    st.write(model2_predict(user_input))
    
def show_source_code(df):
    st.code(get_file_content_as_string("training.py")) 
    # Training model on non-converted Sentiment
    X = df['Phrase']
    y = df['Sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    prediction = model1.predict(X_test)
    
    #Print Result
    st.markdown('## Model 1')
    st.write('Model 1 accuracy:',accuracy_score(y_test,prediction))
    st.markdown('**Model 1 Confusion matrix:**')
    cm1 = confusion_matrix(y_test,prediction)
    sns.heatmap(cm1, annot=True,fmt='g', cmap='Blues', xticklabels=['0', '1', '2', '3', '4'], yticklabels=['0', '1', '2', '3', '4'])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    st.pyplot()
             
    
    # Train model on converted sentiment
    X = df['Phrase']
    y2 = df['converted_sentiment']
    X_train, X_test, y_train_2, y_test_2 = train_test_split(X, y2, test_size=0.2, random_state=0)
    prediction2 = model2.predict(X_test)
    
    #Print Result
    st.markdown('## Model 2')
    st.write('Model 2 accuracy:',accuracy_score(y_test_2,prediction2))
    st.markdown('**Model 2 Confusion matrix:**')
    cm2 = confusion_matrix(y_test_2,prediction2)
    sns.heatmap(cm2, annot=True,fmt='g', cmap='Blues', xticklabels=['0', '1', '2'], yticklabels=['0', '1', '2'])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    st.pyplot()
            
if __name__ == "__main__":
    # Load saved model
    model1, model2 = load_model()
    df = load_dataframe()
    main()
   

