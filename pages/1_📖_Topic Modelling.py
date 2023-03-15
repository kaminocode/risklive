import pandas as pd
import numpy as np

# Text preprocessiong
import nltk
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('wordnet')
wn = nltk.WordNetLemmatizer()
from nltk import sent_tokenize
nltk.download('punkt')

# Topic model
from bertopic import BERTopic

# Dimension reduction
from umap import UMAP

from datetime import date, datetime
from dateutil.relativedelta import relativedelta
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import MaximalMarginalRelevance
from sentence_transformers import SentenceTransformer


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


def load_dataset():
    df = pd.read_csv('./data/eiu_df.csv', encoding = "utf-8")
    df.dropna(subset=['text_body'], inplace=True)
    df.drop_duplicates(subset=['heading'], inplace=True)
    return df

def filter_df(df, chosen_gsr):
    df = df[df.retrieved_gsr==chosen_gsr]
    # my_time = datetime.min.time()
    # start_date = datetime.now() - relativedelta(months=+6)
    # start_date = datetime.combine(start_date, my_time)
    # end_date = datetime.now()
    # end_date = datetime.combine(end_date, my_time)
    df.date = pd.to_datetime(df.date, format='%Y-%m-%d')
    # df = df[(df.date >= start_date) & (df.date <= end_date)]
    df = df.reset_index(drop=True)
    return df

def preprocess_df(df):
    stopwords = nltk.corpus.stopwords.words('english')
    df['text_body_without_stopwords'] = df['text_body'].apply(lambda x: ' '.join([w for w in x.split() if w.lower() not in stopwords]))

    # Lemmatization
    df['text_body_lemmatized'] = df['text_body_without_stopwords'].apply(lambda x: ' '.join([wn.lemmatize(w) for w in x.split() if w not in stopwords]))
    return df

@st.cache_data()
def get_sentences(df):
    docs = []
    titles = []
    timestamps = []
    for text, title, date in zip(df['text_body'], df['heading'], df['date']):
        sentences = sent_tokenize(text)
        docs.extend(sentences)
        titles.extend([title] * len(sentences))
        timestamps.extend([date.timestamp()] * len(sentences))
    return docs, titles, timestamps

@st.cache_data()
def topic_modeling(df):    
    # Vectorizer model
    vectorizer_model = CountVectorizer(stop_words="english")

    # Initiate BERTopic
    docs, titles, timestamps = get_sentences(df) 
    representation_model = MaximalMarginalRelevance(diversity=0.2)
    topic_model = BERTopic(language="english", representation_model=representation_model, vectorizer_model=vectorizer_model)
    
    # Run BERTopic model
    topics, probabilities = topic_model.fit_transform(docs)
    topic_model = topic_model.reduce_topics(docs, nr_topics=9)
    
    # Add topics over time
    topics_over_time = topic_model.topics_over_time(docs, timestamps, nr_bins=20)
    
    # if number of topics is 0, then return None
    if len(topic_model.topic_labels_)==1:
        return None, None
    
    # Visualize the Topic
    fig = topic_model.visualize_barchart(top_n_topics=12)
    fig_over_time = topic_model.visualize_topics_over_time(topics_over_time)
    fig_hierarchical = topic_model.visualize_hierarchy()

    return fig, fig_over_time, fig_hierarchical


# Heading
st.markdown("<h1 style='text-align: center; color: DARK RED;'><b></b>📖 Topic Modelling</h1>", unsafe_allow_html=True)


# st.write(
#     """
# # 📖 Topic Modelling
# """
# )

chosen_gsr = gsr_input_layout()
df = load_dataset()
orig_df = df.copy()
df = filter_df(df, chosen_gsr)

# Get the date input from the user
start_date, end_date = date_input_layout()

# df = preprocess_df(df)

# Visualize the Topic
fig, fig_over_time, fig_hierarchical = topic_modeling(df)

# st.write(type(fig))
if not fig:
    st.write("No topics found!")
else:
    st.plotly_chart(fig, use_container_width=True)
    st.plotly_chart(fig_over_time, use_container_width=True)
    st.plotly_chart(fig_hierarchical, use_container_width=True)

# st.pyplot(fig)


# plot histogram of distance column of the df to see the distribution of the distance seperaly for each GSR
# st.write(df.groupby('retrieved_gsr')['distance'].hist())

