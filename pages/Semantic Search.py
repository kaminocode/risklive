import pickle
import numpy as np
import pandas as pd
import streamlit as st
from annoy import AnnoyIndex
from sentence_transformers import SentenceTransformer, CrossEncoder

st.set_page_config(
    page_title="Semantic Search App", page_icon="🕵️‍♂️", initial_sidebar_state="expanded"
)

st.write(
    """
# 🕵️‍♂️ Semantic Search
Input your search text below.
"""
)

@st.cache(allow_output_mutation=True)
def load_all_models():
    with open("./models/file_value_list.pkl","rb") as f:
        file_value_list = pickle.load(f)
    with open("./models/reverse_file_dict.pkl","rb") as f:
        reverse_file_dict = pickle.load(f)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    f = 384
    u = AnnoyIndex(f, 'angular')
    u.load('./models/query_embeddings.ann')
    return model, u, file_value_list, reverse_file_dict

@st.cache(allow_output_mutation=True)
def get_embedding(model, text):
    embedding = model.encode(text, convert_to_tensor=False)
    return embedding

input_text = st.text_input('Input Text', 'What the health!')
model_name = st.selectbox('Which model would you like to use?',('Annoy', 'Annoy+Rerank'))
knn = st.slider('Number of Nearest Neighbours:', 1, 10, 1)
model, u, file_value_list, reverse_file_dict = load_all_models()
gsr_dict = {'Insufficient Skill Supply': 'skills', 'Supply Chain': 'supplychain', 'Cyber Threat': 'cyberthreat', ' Health, Safety and Wellbeing':'hsw'}
reverse_gsr_dict =  {v: k for k, v in gsr_dict.items()}
embedding = get_embedding(model, input_text)
index = u.get_nns_by_vector(embedding, knn, search_k=-1, include_distances=True)
retrieved_theme_list = [file_value_list[item] for item in index[0]]
retrieved_gsr_list = [reverse_file_dict[retrieved_theme] for retrieved_theme in retrieved_theme_list]
distance_list =  index[1]

if model_name=='Annoy+Rerank':
    df = pd.DataFrame(columns=['Retrieved Theme','GSR','Distance','Cross Distance'])
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    cross_inp = [[input_text, corpus_sentence] for corpus_sentence in retrieved_theme_list]
    cross_scores = cross_encoder.predict(cross_inp)
    cross_distance_list = [float(item) for item in list(cross_scores)]
    df['Retrieved Theme'] = retrieved_theme_list
    df['GSR'] = retrieved_gsr_list
    df['Distance'] = distance_list
    df['Cross Distance'] = cross_distance_list
    df = df.sort_values(by=['Cross Distance'], ascending=False)
else:
    df = pd.DataFrame(columns=['Retrieved Theme','GSR','Distance'])
    df['Retrieved Theme'] = retrieved_theme_list
    df['GSR'] = retrieved_gsr_list
    df['Distance'] = distance_list
st.table(df)
