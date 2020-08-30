# -*- coding: utf-8 -*-
import streamlit as st
import pickle
from pathlib import Path


script_location = Path(__file__).absolute().parent
# Streamlit encourages well-structured code, like starting execution in a main() function.
def main():
    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    st.title('Movie Review Sentiment Analysis')
    st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Show instructions", "Run the app", "Show the source code"])
    if app_mode == "Show instructions":
        st.sidebar.success('To continue select "Run the app".')
    elif app_mode == "Show the source code":
        st.write('Soure Code: https://github.com/mkhoa/rottentomatoes')
    elif app_mode == "Run the app":
        run_the_app()

# Load saved model
@st.cache(allow_output_mutation=True)
def load_model():
    model1 = pickle.load(open(script_location / 'model1.sav', 'rb'))
    model2 = pickle.load(open(script_location / 'model2.sav', 'rb'))
    return model1, model2

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

# Load saved model
model1, model2 = load_model()
def run_the_app():
    user_input = st.text_input("Review Text", 'This is an example review for movie')
    st.write('Prediction by model 1, training by using original sentiment')
    st.write(model1_predict(user_input))
    st.write('Prediction by model 2, training by using converted sentiment')
    st.write(model2_predict(user_input))
    
if __name__ == "__main__":
    main()
