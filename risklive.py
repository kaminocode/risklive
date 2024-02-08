import streamlit as st
import streamlit.components.v1 as components
from bertopic import BERTopic
import pandas as pd
st.set_page_config(page_title="Risk Live", page_icon=':star', layout='wide')

margins_css = """
<style>
.appview-container .main .block-container{{
        padding-left: 0rem;
        }}
</style>
"""

st.markdown(margins_css, unsafe_allow_html=True)


@st.cache_resource()
def get_topic_model():
    embedding_model = "BAAI/bge-large-en-v1.5"
    loaded_topic_model = BERTopic.load("./models/topic_model", embedding_model)
    return loaded_topic_model


def clean_topic_info(topics_info):
    topics_info = topics_info.drop(columns=["Representative_Docs"])
    outlier_row = topics_info[topics_info["Topic"] == -1]
    topics_info = topics_info.drop(outlier_row.index)
    topics_info.loc[len(topics_info)] = outlier_row.values[0]
    return topics_info

@st.cache_data()
def get_df_docs():
    df = pd.read_csv("./data/newscatcher_df_with_response_and_topics.csv")
    docs = df['keywords_list'].tolist()
    return df, docs

@st.cache_data()
def get_figures(_topic_model, df, docs):
    fig0 = _topic_model.visualize_topics(width = 1000, height = 700)
    fig1 = _topic_model.visualize_barchart()
    timestamps = df['published_date'].tolist()
    topics_over_time = _topic_model.topics_over_time(docs, timestamps)
    fig2 = _topic_model.visualize_topics_over_time(topics_over_time)
    fig3 = _topic_model.visualize_documents(df['title'].tolist())
    
    hierarchical_topics = _topic_model.hierarchical_topics(docs)
    fig4 = _topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
    tree = _topic_model.get_topic_tree(hierarchical_topics)
    return fig0, fig1, fig2, fig3, fig4, tree

def main():
    st.title("Risk Live Topics")
    st.write("This app applies topic modeling on news articles from the past 48hours and visualizes them. There is a seperate tab for summary and alerts")

    # Load the topic model
    topic_model = get_topic_model()

    num_topics = len(topic_model.get_topics())
    topics_info = clean_topic_info(topic_model.get_topic_info())
    with st.expander("See Metadata"):
        st.write(f"Number of Topics: {num_topics}")
        st.write(topics_info)
    
    df, docs = get_df_docs()
    fig0, fig1, fig2, fig3, fig4, tree = get_figures(topic_model, df, docs)
    with st.expander("Topic Tree"):
        st.text(tree)
    st.write("Visualizations")
    st.plotly_chart(fig0)
    st.plotly_chart(fig1)
    st.plotly_chart(fig2)
    st.plotly_chart(fig3)
    st.plotly_chart(fig4)
    
if __name__ == "__main__":
    main()

