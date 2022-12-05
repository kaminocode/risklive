import datetime
import pandas as pd
import streamlit as st
from PIL import Image
import leafmap.foliumap as leafmap
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
from unidecode import unidecode
import numpy as np
from PIL import Image
from os import path
import os
import random
from wordcloud import WordCloud, STOPWORDS

bbc_df = pd.read_csv('./data/bbc_df.csv', encoding = "utf-8")
bbc_df = bbc_df.dropna()

def decode_description(line):
    return unidecode(line)
bbc_df['description'] = bbc_df['description'].apply(decode_description)

image = Image.open('./data/Nuclear_Decommissioning_Authority_logo.png')
st.set_page_config(layout="wide")

st.sidebar.image(image, use_column_width=True)
st.sidebar.title("About")
st.sidebar.info(
    """
    Risk Live is a collaborative effort of Nuclear Decommissioning Authority and University of Aberdeen.
    """
)

st.sidebar.title("Contact")
st.sidebar.info(
    """
    Rahul Baburajan:
    [GitHub](https://github.com/rahulbaburaj) | [LinkedIn](https://www.linkedin.com/in/rahul-edachali/)
    """
)

st.markdown("<h1 style='text-align: center; color: DARK RED;'><b></b>RISK LIVE</h1>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
gsr_dict = {'Insufficient Skill Supply': 'skills', 'Supply Chain': 'supplychain', 'Cyber Threat': 'cyberthreat', ' Health, Safety and Wellbeing':'hsw'}
gsr_description_dict = {
    'skills': 'The NDA Group or one of its Businesses has insufficient capability and capacity deliver the mission through not having the right people with the skills at the right time and place.',
    'supplychain': 'Risk that the existing supply chain may not have the capacity or capability to support NDA’s current targets, programmes & ultimately the mission, resulting in failure to deliver HMG policy/ targets, increased government interest & reduced value for money for the UK taxpayer.',
    'hsw': 'Key sources of Health, Safety and Wellbeing (HSW) risk with significant potential of loss of life, serious injury/ ill health or major property damage.',
    'cyberthreat': 'The NDA Group does not proactively deter, detect, defend against, recover from and be resilient to, cyber threats resulting in an adverse effect on delivery of the NDA mission.'
}

with col1:
    gsr = st.selectbox(
            'Which GSR would you like to analyse?',
            ('Insufficient Skill Supply', 'Supply Chain', 'Cyber Threat', ' Health, Safety and Wellbeing'))
chosen_gsr = gsr_dict[gsr]
st.markdown(f"<h4 style='text-align: center; color: DARK RED;'>{gsr_description_dict[chosen_gsr]}</h4>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col2:
    end_date = st.date_input(
        "Select end date",
        datetime.now(),
        min_value=datetime.strptime("2020-01-01", "%Y-%m-%d"),
        max_value=datetime.now(),
    )
with col1:
    start_date = st.date_input(
        "Select start date",
        datetime.now() - relativedelta(months=+6),
        min_value=datetime.strptime("2020-01-01", "%Y-%m-%d"),
        max_value=end_date,
    )

my_time = datetime.min.time()
start_date = datetime.combine(start_date, my_time)
end_date = datetime.combine(end_date, my_time)
bbc_df = bbc_df[bbc_df.retrieved_gsr ==chosen_gsr]
bbc_df.date = pd.to_datetime(bbc_df.date, format='%Y-%m-%d')
bbc_df = bbc_df[(bbc_df.date >=start_date) & (bbc_df.date <=end_date)]

dis = st.slider('Similarity Distance: (Lower the better match)', 0.0, 1.5, 1.3)
bbc_df = bbc_df[bbc_df.distance <= dis]

m = leafmap.Map(center=[51.5072, 0.1276], zoom=6)
cities = 'https://raw.githubusercontent.com/giswqs/leafmap/master/examples/data/us_cities.csv'
url = "https://raw.githubusercontent.com/giswqs/leafmap/master/examples/data/countries.geojson"
# m.add_basemap("CartoDB.DarkMatter")
m.add_geojson(
    url, layer_name="Countries", fill_colors=['blue', 'yellow', 'green', 'orange']
)
m.add_points_from_xy(
    bbc_df,
    x="longitude",
    y="latitude")

m.to_streamlit(height=700)

def grey_color_func(word, font_size, position, orientation, random_state=None,
                    **kwargs):
    return "hsl(0, 0%%, %d%%)" % random.randint(60, 100)


# get data directory (using getcwd() is needed to support running example in generated IPython notebook)
d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()

# read the mask image taken from
# http://www.stencilry.org/stencils/movies/star%20wars/storm-trooper.gif
mask = np.array(Image.open('./data/word_cloud_mask.png'))

# movie script of "a new hope"
# http://www.imsdb.com/scripts/Star-Wars-A-New-Hope.html
# May the lawyers deem this fair use.
text = ' '.join(list(bbc_df.title))

# pre-processing the text a little bit
text = text.replace("HAN", "Han")
text = text.replace("LUKE'S", "Luke")

# adding movie script specific stopwords
stopwords = set(STOPWORDS)
stopwords.add("int")
stopwords.add("ext")
stopwords.add("say")
stopwords.add("says")

wc = WordCloud(max_words=1000, mask=mask, stopwords=stopwords, margin=10,
               random_state=1).generate(text)
# store default colored image
default_colors = wc.to_array()

with st.expander("See word cloud"):
    st.image(wc.to_array(), caption=f'WordCloud representation of data for {gsr} GSR')
