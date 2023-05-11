import streamlit as st
import pandas as pd
from glob import glob
import plotly.express as px
import pickle
import time

def get_last_part_name(file_name):
    # return file_name.split('\\')[-1].split('.')[0].split('_')[-2], file_name.split('\\')[-1].split('.')[0].split('_')[-1]
    return file_name.split('/')[-1].split('.')[0].split('_')[-2], file_name.split('/')[-1].split('.')[0].split('_')[-1]
    
def get_first_name(file_name):
    # results\\bubblefig_cyberthreat_2018.pickle should return bubblefig
    # return file_name.split('\\')[-1].split('_')[0]
    return file_name.split('/')[-1].split('_')[0]


def load_files(pickle_data_path):
    files = glob(pickle_data_path + '/*.pickle')
    data = {}
    gsr_list = []
    year_list = []
    data_name = {}
    for year in ['2018', '2019', '2020', '2021', '2022']:
        data[year] = {}
        data_name[year] = {}
        for gsr in ['supplychain', 'skills', 'hsw', 'cyberthreat']:
            data[year][gsr] = []
            data_name[year][gsr] = []
    for file in files:
        extracted_gsr, year = get_last_part_name(file)
        with open(file, 'rb') as f:
            data[year][extracted_gsr].append(pickle.load(f))
            data_name[year][extracted_gsr].append(file)
    return data, data_name

def create_sample_graph(title, gsr, data, file_num):
    fig = data[gsr][file_num]
    if file_num==4:
        return fig
    fig.update_layout(
        plot_bgcolor='rgba(30, 30, 30, 0.9)',
        paper_bgcolor='rgba(30, 30, 30, 0.9)',
        font=dict(color='white')
    )
    return fig

def write_summary(txt_files, gsr, year):
    for file in txt_files:
        # import pdb;pdb.set_trace()
        # extracted_gsr= file.split('\\')[-1].split('.')[0]
        extracted_gsr= file.split('/')[-1].split('.')[0]
        if extracted_gsr == gsr:
            with open(file, 'r') as f:
                summary_text = f.read()

                # if number of lines is less than 10, then display the text in a text area
                if len(summary_text.splitlines()) < 5:
                    height = 310
                else:
                    height = 500
                return summary_text, height
                


def main():
    gsr_list = ['supplychain', 'skills', 'hsw', 'cyberthreat']

    # Set up the Streamlit app
    st.set_page_config(layout='wide', page_title='Risk Live', page_icon=None, initial_sidebar_state='auto')
    st.markdown("<h1 style='text-align: center;'>🤖 Risk Live </h1>", unsafe_allow_html=True)
    
    # Center-align the selectbox
    empty_space1, center_column, empty_space2 = st.columns([1, 2, 1])
    with center_column:
        gsr = st.selectbox('Select GSR', gsr_list, label_visibility='collapsed')

    # Load the text files in summary folder
    txt_files = glob('summary/*.txt')

    # load images in images folder
    img_files = glob('images/*.png')
            

    data, data_name = load_files('results')
    data_duplicate = data.copy()
    # Display the graphs in a 3-column layout

    i = 0
    year = 2019

    placeholder1 = st.empty()
    placeholder2 = st.empty()
    placeholder3 = st.empty()
    placeholder4 = st.empty()
    placeholder5 = st.empty()
    placeholder6 = st.empty()
    file_name_list = [get_first_name(file_name) for file_name in data_name[str(year)][gsr]]
    file_name_order = ['figovertime', 'bubblefig', 'topicfig', 'wordcloud', 'map']
    col1_1, col2_1 = st.columns([1, 1])
    col2_1, col2_2 = st.columns([1, 1])
        


    while True:
        file_name_list = [get_first_name(file_name) for file_name in data_name[str(year)][gsr]]

        with col1_1:
            fig = data[str(year)][gsr][file_name_list.index('figovertime')]
            fig.update_layout(title_text="")
            fig.update_layout(
                width=800,
                height=800,
                margin=dict(
                    l=0,   # left margin
                    r=0,   # right margin
                    t=0,   # top margin
                    b=0    # bottom margin
                    )
                )
            
            placeholder1.plotly_chart(fig, use_container_width=True)
            
        with col2_1:
            fig = data[str(year)][gsr][file_name_list.index('bubblefig')]
            fig.update_layout(
                width=800,
                height=800,
                margin=dict(
                    l=0,   # left margin
                    r=0,   # right margin
                    t=0,   # top margin
                    b=0    # bottom margin
                    )
                )
            placeholder2.plotly_chart(fig, use_container_width=True)
        

        fig = data[str(year)][gsr][file_name_list.index('map')]
        fig.update_layout(
                width=800,
                height=800,
                margin=dict(
                    l=0,   # left margin
                    r=0,   # right margin
                    t=0,   # top margin
                    b=0    # bottom margin
                    )
                )
        placeholder5.plotly_chart(fig, use_container_width=True)


        with col2_1:
            fig = data[str(year)][gsr][file_name_list.index('topicfig')]
            fig.update_layout(title_text="")
            fig.update_layout(
                width=800,
                height=800,
                margin=dict(
                    l=0,   # left margin
                    r=0,   # right margin
                    t=0,   # top margin
                    b=0    # bottom margin
                    )
                )
            placeholder3.plotly_chart(fig, use_container_width=True)

        with col2_2:
            for img in img_files:
                img_gsr, img_year = get_last_part_name(img)
                if img_gsr == gsr and img_year==str(year):
                    placeholder6.image(img, use_column_width=True)
            
        
        
        summary_text, height = write_summary(txt_files, gsr, year)
        placeholder4.text_area('examples', summary_text, height=height, disabled = False, label_visibility='collapsed', key=f'{gsr}_{year}_{i}')


        year = year + 1
        if year == 2023:
            year = 2019
        i = i + 1
        time.sleep(2)
        

        

if __name__ == '__main__':
    main()