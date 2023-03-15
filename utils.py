import pandas as pd
import streamlit as st
import plotly.express as px 
from unidecode import unidecode
import leafmap.foliumap as leafmap
from datetime import date, datetime
from dateutil.relativedelta import relativedelta


def load_dataset():
    df = pd.read_csv('./data/eiu_df.csv', encoding = "utf-8")
    return df
  
def gsr_input_layout():
    gsr_dict = {'Insufficient Skill Supply': 'skills', 'Supply Chain': 'supplychain', 'Cyber Threat': 'cyberthreat', 'Health, Safety and Wellbeing':'hsw'}
    gsr_description_dict = {
        'skills': 'The NDA Group or one of its Businesses has insufficient capability and capacity deliver the mission through not having the right people with the skills at the right time and place.',
        'supplychain': 'Risk that the existing supply chain may not have the capacity or capability to support NDA’s current targets, programmes & ultimately the mission, resulting in failure to deliver HMG policy/ targets, increased government interest & reduced value for money for the UK taxpayer.',
        'hsw': 'Key sources of Health, Safety and Wellbeing (HSW) risk with significant potential of loss of life, serious injury/ ill health or major property damage.',
        'cyberthreat': 'The NDA Group does not proactively deter, detect, defend against, recover from and be resilient to, cyber threats resulting in an adverse effect on delivery of the NDA mission.'
    }
    gsr = st.selectbox(
            'Which GSR would you like to analyse?',
            ('Insufficient Skill Supply', 'Supply Chain', 'Cyber Threat', 'Health, Safety and Wellbeing'))
    chosen_gsr = gsr_dict[gsr]
    st.markdown(f"<h4 style='text-align: center; color: DARK RED;'>{gsr_description_dict[chosen_gsr]}</h4>", unsafe_allow_html=True)
    return chosen_gsr

def date_input_layout():
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
    return start_date, end_date

def filter_dataframe(df, chosen_gsr, start_date, end_date):
    my_time = datetime.min.time()
    start_date = datetime.combine(start_date, my_time)
    end_date = datetime.combine(end_date, my_time)
    df = df[df.retrieved_gsr ==chosen_gsr]
    df.date = pd.to_datetime(df.date, format='%Y-%m-%d')
    df = df[(df.date >=start_date) & (df.date <=end_date)]
    return df

    
def theme_short_form(theme):
    # return the short form of the theme
    if len(theme.split(' ')) == 1:
        return theme
    return theme.split(' ')[0]+' '+theme.split(' ')[1]

def plot_theme(df):
    df_theme = df.best_theme.value_counts().rename_axis('best_theme').reset_index(name='counts')
    df_theme['theme'] = df_theme['best_theme'].apply(theme_short_form)
    # Create a horizontal barchat chart in streamlit using plotly for the df_theme dataframe
    fig = px.bar(df_theme, x='counts', y='theme', orientation='h', color='counts', color_continuous_scale='Viridis')
    st.plotly_chart(fig, use_container_width=True)

def decode_description(line):
    return unidecode(line)

def plot_leafmap(df):
    # df = df[['latitude', 'longitude', 'summary', 'date', 'retrieved_gsr']]
    # df['summary'] = df['summary'].apply(decode_description)
    df = df[['latitude', 'longitude', 'date', 'retrieved_gsr']]
    m = leafmap.Map(center=[51.5072, 0.1276], zoom=2)
    cities = 'https://raw.githubusercontent.com/giswqs/leafmap/master/examples/data/us_cities.csv'
    url = "https://raw.githubusercontent.com/giswqs/leafmap/master/examples/data/countries.geojson"
    # m.add_basemap("CartoDB.DarkMatter")
    m.add_geojson(
        url, layer_name="Countries", fill_colors=['blue', 'yellow', 'green', 'orange']
    )
    m.add_points_from_xy(
        df,
        x="longitude",
        y="latitude")
    
    m.to_streamlit(height=700)
