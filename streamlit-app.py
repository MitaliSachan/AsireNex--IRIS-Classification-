from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import pickle
import streamlit as st
from PIL import Image


# loading in the model to predict on the data
pickle_in=open('classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)

def welcome():
	return 'welcome all'

# defining the function which will make the prediction using
# the data which the user inputs
def prediction(sepal_length, sepal_width, petal_length, petal_width):

	prediction = classifier.predict(
		[[sepal_length, sepal_width, petal_length, petal_width]])
	print(prediction)
	return prediction

# this is the main function in which we define our webpage
def main():
	# giving the webpage a title
	st.title("Iris Flower Prediction")

	# here we define some of the front end elements of the web page like
	# the font and background color, the padding and the text to be displayed
	html_temp = """
	<div style ="background-color:teal;padding:13px">
	<h2 style ="color:white;text-align:center;">Streamlit Iris Flower Classifier</h2>
	</div><br>
	"""
  # this line allows us to display the front end aspects we have
	# defined in the above code
	st.markdown(html_temp, unsafe_allow_html = True)
	
	ColorMinMax = st.markdown(''' <style> div.stSlider > div[data-baseweb = "slider"] > div[data-testid="stTickBar"] > div {background: rgb(1 1 1 / 0%); } </style>''', unsafe_allow_html = True)
	Slider_Cursor = st.markdown(''' <style> div.stSlider > div[data-baseweb="slider"] > div > div > div[role="slider"]{background-color: rgb(14, 38, 74); box-shadow: rgb(14 38 74 / 20%) 0px 0px 0px 0.2rem;} </style>''', unsafe_allow_html = True)
	Slider_Number = st.markdown(''' <style> div.stSlider > div[data-baseweb="slider"] > div > div > div > div
                                { color: rgb(14, 38, 74); } </style>''', unsafe_allow_html = True)
	
	col = f''' <style> div.stSlider > div[data-baseweb = "slider"] > div > div {{background: linear-gradient(to right, rgb(1, 183, 158) 0%, 
                                rgb(1, 183, 158) {sl}%, 
                                rgba(151, 166, 195, 0.25) {sw}%, 
                                rgba(151, 166, 195, 0.25) 100%); }} </style>'''
	
	ColorSlider = st.markdown(col, unsafe_allow_html = True)

	# the following lines create text boxes in which the user can enter
	# the data required to make the prediction
	sl=st.slider('Select Sepal Length', 0.0, 10.0)
	sw=st.slider('Select Sepal Width', 0.0, 10.0)
	pl=st.slider('Select Petal Length', 0.0, 10.0)
	pw=st.slider('Select Petal Width', 0.0, 10.0)
	result =""

	# the below line ensures that when the button called 'Predict' is clicked,
	# the prediction function defined above is called to make the prediction
	# and store it in the variable result
	if st.button("Predict"):
		result = prediction(sl, sw, pl, pw)
	st.success('The output is {}'.format(result))

if __name__=='__main__':
	main()
