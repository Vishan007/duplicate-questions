import streamlit as st
import utils
import pickle
import warnings
warnings.filterwarnings('ignore')

model = pickle.load(open('model.pkl','rb'))

st.header('Duplicate Questions')

q1 = st.text_input('Enter Question 1')
q2 = st.text_input('Enter Question 2')

if st.button('Find'):
    query = utils.query_point_creator(q1 ,q2)
    result = model.predict(query)[0]

    if result:
        st.header('Duplicate')
    else:
        st.header('Not Duplicate')