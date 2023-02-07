import os
import nltk
nltk.download('stopwords')
import random
import datetime
import numpy as np
import pandas as pd
from os import path
from utils import *
from PIL import Image
import streamlit as st
from unidecode import unidecode
from nltk.corpus import stopwords
import leafmap.foliumap as leafmap
from wordcloud import WordCloud, STOPWORDS

st.set_page_config(layout="wide")

# Read the dataset
df = load_dataset()

# Heading
st.markdown("<h1 style='text-align: center; color: DARK RED;'><b></b>📺 RISK LIVE</h1>", unsafe_allow_html=True)

# Get the gsr input from the user
chosen_gsr = gsr_input_layout()

# Get the date input from the user
start_date, end_date = date_input_layout()

# Filter the dataframe based on the user input
df = filter_dataframe(df, chosen_gsr, start_date, end_date)

# Create a slider to filter the dataframe based on the similarity distance
dis = st.slider('Similarity Distance: (Lower the better match)', 0.0, 1.5, 1.2)
df = df[df.distance <= dis]

# Plot the map 
plot_leafmap(df)

# Plot the Best themes for the selected GSR
# plot_theme(df)

