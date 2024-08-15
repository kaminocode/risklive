import os
import pickle
import streamlit as st
import plotly.io as pio
st.set_page_config(page_title="Risk Live", page_icon=':star', layout='wide')

margins_css = """
<style>
.appview-container .main .block-container{{
        padding-left: 0rem;
        }}
</style>
"""

st.markdown(margins_css, unsafe_allow_html=True)
IMG_DIR = "./results/images"

def get_figures():
    with open(os.path.join(IMG_DIR, '3d_time_plot.pkl'), 'rb') as f:
        fig1 = pickle.load(f)
    
    with open(os.path.join(IMG_DIR, 'treemap.pkl'), 'rb') as f:
        fig2 = pickle.load(f)
    
    json_files = ['topics.json', 'barchart.json', 'topics_over_time.json', 'documents.json', 'hierarchy.json']
    json_figures = [fig1]
    for file in json_files:
        with open(os.path.join(IMG_DIR, file), 'r') as f:
            fig = pio.from_json(f.read())
            json_figures.append(fig)
    json_figures.append(fig2)
    tree_path = os.path.join(IMG_DIR, 'topic_tree.txt')
    with open(tree_path, 'r') as f:
        tree = f.read()
    return json_figures, tree


def main():
    st.title("Risk Live: Topic Modeling")
    st.write("This app applies topic modeling on news articles from the past 72hours and visualizes them. There is a seperate tab for summary and alerts")

    json_figures, tree = get_figures()
    with st.expander("Topic Tree"):
        st.text(tree)
        
    st.plotly_chart(json_figures[6])
    
    st.plotly_chart(json_figures[0])
    st.plotly_chart(json_figures[1])
    st.plotly_chart(json_figures[2])
    st.plotly_chart(json_figures[3])
    st.plotly_chart(json_figures[4])
    st.plotly_chart(json_figures[5])
    
    
if __name__ == '__main__':
    main()