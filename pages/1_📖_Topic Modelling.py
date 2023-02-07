import pandas as pd
import numpy as np

# Text preprocessiong
import nltk
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('wordnet')
wn = nltk.WordNetLemmatizer()

# Topic model
from bertopic import BERTopic

# Dimension reduction
from umap import UMAP

from datetime import date, datetime
from dateutil.relativedelta import relativedelta
import streamlit as st


def gsr_input_layout():
    gsr_dict = {'Insufficient Skill Supply': 'skills', 'Supply Chain': 'supplychain', 'Cyber Threat': 'cyberthreat', ' Health, Safety and Wellbeing':'hsw'}
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

def load_dataset():
    df = pd.read_csv('./data/eiu_df.csv', encoding = "utf-8")
    df = df.dropna()
    df = df[df.best_theme!='No theme found']  
    return df

def filter_df(df, chosen_gsr):
    df = df[df.retrieved_gsr==chosen_gsr]
    my_time = datetime.min.time()
    start_date = datetime.now() - relativedelta(months=+6)
    start_date = datetime.combine(start_date, my_time)
    end_date = datetime.now()
    end_date = datetime.combine(end_date, my_time)
    df.date = pd.to_datetime(df.date, format='%Y-%m-%d')
    df = df[(df.date >= start_date) & (df.date <= end_date)]
    df = df.reset_index(drop=True)
    return df

def preprocess_df(df):
    stopwords = nltk.corpus.stopwords.words('english')
    df['text_body_without_stopwords'] = df['text_body'].apply(lambda x: ' '.join([w for w in x.split() if w.lower() not in stopwords]))

    # Lemmatization
    df['text_body_lemmatized'] = df['text_body_without_stopwords'].apply(lambda x: ' '.join([wn.lemmatize(w) for w in x.split() if w not in stopwords]))
    return df

@st.cache()
def topic_modeling(df):
    # Initiate UMAP
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=100)

    # Initiate BERTopic
    topic_model = BERTopic(umap_model=umap_model, language="english", calculate_probabilities=True)

    # Run BERTopic model
    topics, probabilities = topic_model.fit_transform(df['text_body_lemmatized'])

    # Visualize the Topic
    fig = topic_model.visualize_barchart(top_n_topics=12)

    return fig

# Heading
st.markdown("<h1 style='text-align: center; color: DARK RED;'><b></b>📖 Topic Modelling</h1>", unsafe_allow_html=True)


# st.write(
#     """
# # 📖 Topic Modelling
# """
# )

chosen_gsr = gsr_input_layout()
df = load_dataset()
df = filter_df(df, chosen_gsr)
df = df[:1000]

df = preprocess_df(df)

# Visualize the Topic
fig = topic_modeling(df)

# st.write(type(fig))
st.plotly_chart(fig, use_container_width=True)
# st.pyplot(fig)




