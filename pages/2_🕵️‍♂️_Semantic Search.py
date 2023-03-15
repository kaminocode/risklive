import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from nltk import sent_tokenize

st.set_page_config(layout="wide")

# Heading
st.markdown("<h1 style='text-align: center; color: DARK RED;'><b></b>🕵️‍♂️ Semantic Search</h1>", unsafe_allow_html=True)

# Load the pre-trained SentenceTransformer model
@st.cache_data()
def load_model():
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    return model
model = load_model()


@st.cache_data()
def load_dataset():
    df = pd.read_csv('../data/eiu_df.csv', encoding = "utf-8")
    df.dropna(subset=['text_body'], inplace=True)
    df.drop_duplicates(subset=['heading'], inplace=True)
    return df
df = load_dataset()

# Sample corpus
corpus = df['text_body'].tolist()
heading = df['heading'].tolist()


# Preprocess the news corpus
@st.cache_data()
def preprocess_corpus(corpus):
    news_data = []
    news_line_info = []
    for idx, news in enumerate(corpus):
        lines = sent_tokenize(news)
        for line in lines:
            if line.strip():
                news_data.append(line.strip())
                news_line_info.append((idx, line.strip()))
    return news_data, news_line_info
news_data, news_line_info = preprocess_corpus(corpus)

# Create embeddings for the news data
@st.cache_data()
def create_embeddings(news_data):
    embeddings = model.encode(news_data, convert_to_tensor=True)
    return embeddings

embeddings = create_embeddings(news_data)

@st.cache_data()
def search_query(query, top_num=10):
    # Create an embedding for the query
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Compute similarity between query and all news data
    cos_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]

    # Get the index of the 10 most similar news data
    top_index_list = np.argpartition(-cos_scores, range(top_num))[0:top_num]
    top_index = np.argmax(cos_scores)

    # get the list of most similar news data and the similarity score
    score_list = [cos_scores[idx] for idx in top_index_list]
    news_list = [news_line_info[idx] for idx in top_index_list]
    return news_list, score_list

# Example search
query = st.text_input('Input Query', 'Covid19')
result, similarity_score = search_query(query)


# get the line prior and next to the matching line in the text
def subset_text(heading, text, highlight):
    lines = sent_tokenize(text)
    for line in lines:
        if highlight in line:
            index = lines.index(line)
            break
    line_1 = lines[index - 1]
    line_2 = lines[index]
    line_3 = lines[index + 1]
    return_text = line_1 + line_2 + line_3
    text2highlight = line_2
    return highlight_text(heading, return_text, text2highlight)
    

def highlight_text(heading, text, text2highlight):
    start_tag = "<mark>"
    end_tag = "</mark>"
    highlighted_text = text.replace(text2highlight, f"{start_tag}{text2highlight}{end_tag}")
    return f"<h3>{heading}</h3>{highlighted_text}"


highlight_text_list = []
for result_tuple in result:
    heading_text = heading[result_tuple[0]]
    heading_text = heading_text.replace('Download the numbers in Excel', '')
    highlighted_text = subset_text(heading_text, corpus[result_tuple[0]], result_tuple[1])
    if highlighted_text not in highlight_text_list:
        st.markdown(f"<style>mark {{ background-color: yellow; }}</style>{highlighted_text}<br><br><br>", unsafe_allow_html=True)
    highlight_text_list.append(highlighted_text)
    
