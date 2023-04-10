import streamlit as st
import pandas as pd
from glob import glob
import plotly.express as px
import pickle

def get_last_part_name(file_name):
    return file_name.split('/')[-1].split('.')[0].split('_')[-1]

def get_first_name(file_name):
    # results\\bubblefig_cyberthreat.pickle should return bubblefig
    return file_name.split('\\')[-1].split('_')[0]

def load_files(pickle_data_path):
    files = glob(pickle_data_path + '/*.pickle')
    data = {}
    gsr_list = []
    data_name = {}
    for file in files:
        extracted_gsr = get_last_part_name(file)
        if extracted_gsr not in gsr_list:
            data[extracted_gsr]=[]
            gsr_list.append(extracted_gsr)
            data_name[extracted_gsr] = []
        with open(file, 'rb') as f:
            data[extracted_gsr].append(pickle.load(f))
            data_name[extracted_gsr].append(file)
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

def main():
    gsr_list = ['skills', 'supplychain', 'cyberthreat', 'hsw']

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
    # Display the graphs in a 3-column layout

    file_name_list = [get_first_name(file_name) for file_name in data_name[gsr]]
    file_name_order = ['topicfig', 'figovertime', 'bubblefig', 'wordcloud', 'map']
    # change the order of the file_name_list and data[gsr] to match the order of file_name_order
    file_name_list = [file_name_list[file_name_order.index(file_name)] for file_name in file_name_order]
    data[gsr] = [data[gsr][file_name_list.index(file_name)] for file_name in file_name_order]

    for fig, first_name in zip(data[gsr], file_name_order):
        if first_name == 'wordcloud':
            # Display the generated image:
            # plt.imshow(fig, interpolation='bilinear')
            # plt.axis("off")
            # plt.show()
            # st.pyplot()
            continue
        else:
            st.plotly_chart(fig, use_container_width=True)

    # plot the image
    # st.markdown("<h2 style='text-align: center; color: black;'>WordCloud </h2>", unsafe_allow_html=True)
    # align the image to the center
    for img in img_files:
        if get_last_part_name(img) == gsr:
            st.image(img, use_column_width=True)
    
    #     st.write(get_first_name(file_name))
    # st.plotly_chart(create_sample_graph("Graph 1", gsr, data, file_num=0), use_container_width=True)
    # st.plotly_chart(create_sample_graph("Graph 2", gsr, data, file_num=1), use_container_width=True)
    # st.plotly_chart(create_sample_graph("Graph 3", gsr, data, file_num=2), use_container_width=True)
    # st.plotly_chart(create_sample_graph("Graph 4", gsr, data, file_num=3), use_container_width=True)
    # st.plotly_chart(create_sample_graph("Graph 5", gsr, data, file_num=4), use_container_width=True)
    # st.plotly_chart(create_sample_graph("Graph 6", gsr, data, file_num=5), use_container_width=True)


    st.markdown("<h2 style='text-align: center;'>Summary </h2>", unsafe_allow_html=True)
    for file in txt_files:
        if get_last_part_name(file.split('\\')[-1]) == gsr:
            with open(file, 'r') as f:
                summary_text = f.read()

                # if number of lines is less than 10, then display the text in a text area
                if len(summary_text.splitlines()) < 5:
                    height = 310
                else:
                    height = 500
                st.text_area('examples', summary_text, height=height, disabled = False, label_visibility='collapsed')


if __name__ == '__main__':
    main()