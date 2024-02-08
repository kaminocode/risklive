import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

def get_news_last_2h(df):
    time_now = datetime.now()
    time_2h_ago = time_now - timedelta(hours=2)
    df['published_date'] = pd.to_datetime(df['published_date'])
    df = df[df['published_date'] > time_2h_ago]
    return df

def alert():
    st.header("Risk Alerts")
    df = pd.read_csv("./data/newscatcher_df_with_response.csv")
    df_red = get_news_last_2h(df[df['alertflag']=="Red"])
    df_yellow = get_news_last_2h(df[df['alertflag']=="Yellow"])

    if not df_red.empty:
        with st.expander("Moderate Risk News in past 2 hours"):
            for index, row in df_red.iterrows():
            # write in bullet points {row['title']}\n{row['link']}\n{row['llm_summary']}
                st.subheader(f"{row['title']}")
                st.markdown(f"({row['link']})")
                st.markdown(f"{row['llm_summary']}")
                st.write("\n\n")
                
    if not df_yellow.empty:
        with st.expander("Moderate to Low Risk News in past 2 hours"):
            for index, row in df_yellow.iterrows():
                st.subheader(f"{row['title']}")
                st.markdown(f"({row['link']})")
                st.write(f"{row['llm_summary']}")
                st.write("\n\n")
            
            
def load_txt_file(file):
    with open(file, 'r') as f:
        data = f.read()
    return data.replace("Risk Analysis Summary:", "").strip()

def summary():
    st.header("Risk Summary")
    summary_folder = "./results/summary"
    red_summary = load_txt_file(f"{summary_folder}/red/summary_conscise.txt")
    yellow_summary = load_txt_file(f"{summary_folder}/yellow/summary_conscise.txt")
    green_summary = load_txt_file(f"{summary_folder}/green/summary.txt")
    
    if red_summary:
        with st.expander("Moderate Risk News Summary"):
            st.write(red_summary)
    if yellow_summary:
        with st.expander("Moderate to Low Risk News Summary"):
            st.write(yellow_summary)
    if green_summary:
        with st.expander("Low Risk News Summary"):
            st.write(green_summary)
            
if __name__ == "__main__":
    st.title("Risk Alerts and Summary")
    alert()
    summary()