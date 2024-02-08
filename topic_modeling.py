import os
import json
import pandas as pd
import schedule
import time
import logging
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from dotenv import load_dotenv
from openai import AzureOpenAI
from utils import load_environment_variables, initialize_client, api_call

# Ensure logging is set up to capture info level logs
logging.basicConfig(filename='./logs/topic_modeling.log', filemode='a', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
# logging.getLogger().setLevel(logging.DEBUG)

def get_keyword_list(keyword_list_str):
    """Converts a string representation of a list into a list of keywords."""
    logging.debug("Starting to convert keyword list string to list.")
    keyword_list = [str(item[1:-1]) for item in keyword_list_str[1:-1].split(", ")]
    keyword_str = ", ".join(keyword_list)
    logging.debug("Keyword list conversion completed.")
    return keyword_str

def clean_dataframe(df):
    """Cleans the dataframe by filtering relevant rows and handling missing values."""
    df_filtered = df[df.relevance == 'Yes'].dropna(subset=['keywords'])
    return df_filtered

def load_and_prepare_data(filepath):
    """Loads data from a CSV file, cleans it, and prepares it for modeling."""
    logging.info(f"Loading data from {filepath}.")
    df = pd.read_csv(filepath)
    cleaned_df = clean_dataframe(df)
    cleaned_df['keywords_list'] = cleaned_df['keywords'].apply(get_keyword_list)
    docs = cleaned_df['keywords_list'].tolist()
    logging.info("Data preparation completed.")
    return cleaned_df, docs

def initialize_models(embedding_model_name, len_docs):
    """Initializes and returns sentence transformer and BERTopic models along with sub-models."""
    sentence_model = SentenceTransformer(embedding_model_name)
    vectorizer = CountVectorizer(stop_words="english")
    if len_docs<100:
        umap_model = UMAP(n_neighbors=2, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
        hdbscan_model = HDBSCAN(min_cluster_size=2, min_samples=2, metric='euclidean', cluster_selection_method='eom')
    elif len_docs<200:
        umap_model = UMAP(n_neighbors=5, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
        hdbscan_model = HDBSCAN(min_cluster_size=5, min_samples=5, metric='euclidean', cluster_selection_method='eom')
    else:
        umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
        hdbscan_model = HDBSCAN(min_cluster_size=15, min_samples=15, metric='euclidean', cluster_selection_method='eom')
    topic_model = BERTopic(embedding_model=sentence_model, umap_model=umap_model, hdbscan_model=hdbscan_model, vectorizer_model=vectorizer)
    return sentence_model, topic_model

def generate_embeddings(sentence_model, docs):
    """Encodes documents into embeddings."""
    logging.debug("Generating embeddings for documents.")
    try:
        embeddings = sentence_model.encode(docs)
        logging.debug("Embeddings generated successfully.")
        return embeddings
    except Exception as e:
        logging.error(f"Error generating embeddings: {e}")
        raise


def train_topic_model(topic_model, docs, embeddings):
    """Fits the topic model with the documents and embeddings."""
    logging.info("Starting to train the topic model.")
    topic_model.fit(docs, embeddings)
    logging.info("Topic model training completed.")

def save_and_visualize(topic_model, docs, df, output_dir="results/images"):
    """Saves the topic model and generates visualizations."""
    logging.info("Starting to save the topic model and generate visualizations.")
    embedding_model = "BAAI/bge-large-en-v1.5"
    try:
        topic_model.save(f"./models/topic_model", serialization="safetensors", save_ctfidf=True, save_embedding_model=embedding_model)
        logging.debug("Topic model saved successfully.")
    except Exception as e:
        logging.error(f"Error saving topic model: {e}")
        
    df['topic'] = topic_model.topics_
    df.to_csv("./data/newscatcher_df_with_response_and_topics.csv", index=False)
    logging.debug("Dataframe with topics saved.")
    
    fig1 = topic_model.visualize_barchart()
    fig1.write_html(f"{output_dir}/barchart.html")
    logging.debug("Barchart visualization saved.")
    
    fig2 = topic_model.visualize_topics(width=1000, height=1000)
    fig2.write_html(f"{output_dir}/topics.html")
    
    fig3 = topic_model.visualize_documents(df['title'].tolist())
    fig3.write_html(f"{output_dir}/documents.html")
    
    fig4 = topic_model.visualize_topics_over_time(topic_model.topics_over_time(docs, df['published_date'].tolist()))
    fig4.write_html(f"{output_dir}/topics_over_time.html")
    
    hierarchical_topics= topic_model.hierarchical_topics(docs)
    fig5 = topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
    fig5.write_html(f"{output_dir}/hierarchy.html")
    logging.debug("Hierarchy visualization saved.")

    logging.info("All visualizations generated and saved.")

def get_news_articles_with_metadata(df):
    logging.debug("Extracting news articles with metadata.")
    news_articles_with_metadata = ""
    for i, row in df.iterrows():
        # news_articles_with_metadata = news_articles_with_metadata + f"Published Date: {row['published_date']}\nFlag: {row['alertflag']}\nNews: {row['llm_summary']}\n\n\n"
        news_articles_with_metadata = news_articles_with_metadata + f"News: {row['llm_summary'].strip()}\n"
    logging.debug("Extraction of news articles with metadata completed.")
    return news_articles_with_metadata
        
def initialize():
    logging.info("Initializing environment and loading prompts.")
    try:
        with open("./prompt/summarization_prompt.txt", "r") as f:
            prompt = f.read()
        with open("./prompt/summary_format.txt", "r") as f:
            summary_format = f.read()
        config = load_environment_variables()
        client = initialize_client(config)
        logging.info("Initialization successful.")
        return client, prompt, summary_format
    except Exception as e:
        logging.error(f"Error during initialization: {e}")
        raise

def parse_response(response):
    logging.debug("Parsing API response.")
    try:
        response_json = json.loads(response)
        info = {
            'summary': response_json['Summary'],
            'insights': response_json['Insights'],
            'recommendations': response_json['Recommendations'],
        }
        return info
    except json.JSONDecodeError as e:
        logging.info(f"Response causing the error: {response}")
        logging.error(f"Error parsing response: {e}")
        raise

def make_summary_conscise(flag):
    config = load_environment_variables()
    client = initialize_client(config)
    with open("./prompt/conscise.txt", "r") as f:
        prompt = f.read()
    with open(f"./results/summary/{flag.lower()}/summary.txt", "r") as f:
        summary = f.read()
    if summary.strip() != "":    
        prompt = prompt.format(news_data=summary)
        response, price, token_usage = api_call(client, prompt)
        if response:    
            with open(f"./results/summary/{flag.lower()}/summary_conscise.txt", "w") as f:
                f.write(response)
    else:
        logging.info(f"No summary found for flag {flag}.")
        with open(f"./results/summary/{flag.lower()}/summary_conscise.txt", "w") as f:
            f.write("")
            
def writeorappend(filename, text):
    logging.debug(f"Writing or appending to file: {filename}.")
    try:
        if os.path.exists(filename):
            with open(filename, "r") as f:
                old_str = f.read()
            with open(filename, "w") as f:
                f.write(text + "\n" + old_str)
        else:
            with open(filename, "w") as f:
                f.write(text)
        logging.debug(f"Content successfully written to {filename}.")
    except IOError as e:
        logging.error(f"Error writing to file {filename}: {e}")
        raise

def save_summary_data(summary_str, flag):
    logging.info(f"Saving summary data for flag: {flag.lower()}.")
    try:
        folder_path = f"./results/summary/{flag.lower()}"
        summary_file_path = f"{folder_path}/summary.txt"
        writeorappend(summary_file_path, summary_str)
        logging.info("Summary data saved successfully.")
        make_summary_conscise(flag)
        logging.info("Concise summary generated successfully.")
    except Exception as e:
        logging.error(f"Error saving summary data for flag {flag}: {e}")
        raise

def save_summary(df, flag):
    logging.info(f"Saving summary for flag: {flag}.")
    logging.info(f"Processing {len(df)} rows.")
    try:
        client, prompt, summary_format = initialize()
        summary_list = []
        for _, df_topic in df.groupby('topic'):
            news_articles_with_metadata = get_news_articles_with_metadata(df_topic)
            input_prompt = prompt.format(news_articles_with_metadata=news_articles_with_metadata, summary_format=summary_format)
            response_output, price, token_usage = api_call(client, input_prompt)
            if response_output:
                summary_list.append(response_output.strip())
        summary_str = "\n".join(summary_list)
        save_summary_data(summary_str, flag)
        logging.info("Summary for flag completed successfully.")
    except Exception as e:
        logging.error(f"Error saving summary for flag {flag}: {e}")
        raise

def get_save_summary(df_new):
    logging.info("Getting and saving summary for new data.")
    try:
        df = pd.read_csv("./data/newscatcher_df_with_response_and_topics.csv")
        df = df[df.link.isin(df_new.link)]  
        for flag in ['Red', 'Yellow', 'Green']:
            df_flag = df[df.alertflag == flag]
            save_summary(df_flag, flag)
        logging.info("Summary for new data saved successfully.")
    except Exception as e:
        logging.error(f"Error in getting and saving summary: {e}")
        raise
   
    
def check_and_process_new_data():
    """Checks for new data and processes it if found."""
    logging.info("Checking for new data to process...")
    try:
        processed_df = pd.read_csv("./data/newscatcher_df_with_response_and_topics.csv")
        processed_df = clean_dataframe(processed_df)
    except FileNotFoundError:
        processed_df = pd.DataFrame()

    try:
        new_data_df = pd.read_csv("./data/newscatcher_df_with_response.csv")
        new_data_df = clean_dataframe(new_data_df)
    except FileNotFoundError as e:
        logging.error("New data file not found.")
        return

    if not processed_df.empty:
        new_links = ~new_data_df['link'].isin(processed_df['link'])
        new_rows_df = new_data_df[new_links]
    else:
        new_rows_df = new_data_df

    if not new_rows_df.empty:
        logging.info(f"Found {len(new_rows_df)} new rows. Processing...")
        # Adjusted to pass dataframe directly to main
        main(new_data_df, new_rows_df)  # Process only new rows
    else:
        logging.info("No new rows found.")

def schedule_checks():
    """Schedules data checks to run every minute."""
    logging.info("Scheduling data checks to run every minute.")
    try:
        schedule.every(1).minutes.do(check_and_process_new_data)
        logging.info("Data checks scheduled successfully. Entering the loop to run pending.")
        while True:
            schedule.run_pending()
            time.sleep(1)
    except Exception as e:
        logging.error(f"Error in scheduling or running data checks: {e}")
        raise

def main(df=None, df_new=None):
    """Main function to process data."""
    logging.info("Starting main data processing function.")
    try:
        filepath = "./data/newscatcher_df_with_response.csv"
        logging.debug(f"Loading and preparing data from {filepath}.")
        cleaned_df, docs = load_and_prepare_data(filepath)
        
        logging.info("Initializing models.")
        sentence_model, topic_model = initialize_models("BAAI/bge-large-en-v1.5", len(docs))
        
        logging.info("Generating embeddings.")
        embeddings = generate_embeddings(sentence_model, docs)
        
        logging.info("Training topic model.")
        train_topic_model(topic_model, docs, embeddings)
        
        logging.info("Saving and visualizing topic model and data.")
        save_and_visualize(topic_model, docs, cleaned_df)
        
        if df_new is not None:
            logging.debug("Processing new dataset for summary generation.")
            get_save_summary(df_new)
        
        logging.info("Main data processing function completed successfully.")
    except Exception as e:
        logging.error(f"Error in main processing function: {e}")
        raise

if __name__ == "__main__":
    schedule_checks()
