import time
import json
import os
import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI
import logging
import schedule
from utils import load_environment_variables, initialize_client, api_call

# Configure logging
logging.basicConfig(filename='./logs/topic_gpt.log', filemode='a', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
# log debug messages
logging.getLogger().setLevel(logging.DEBUG)


def clean_df(df):
    remove_summary_list = ["Continue reading on Medium »", "Please enable JS and disable any ad blocker"]
    df = df[~df.summary.isin(remove_summary_list)]
    df = df.drop_duplicates(subset=['link'])
    return df
    

def check_and_process_new_data():
    logging.info("Checking for new data to process...")
    try:
        processed_df = pd.read_csv("./data/newscatcher_df_with_response.csv")
        processed_df = clean_df(processed_df)
    except FileNotFoundError:
        processed_df = pd.DataFrame()

    new_data_df = pd.read_csv("./data/newscatcher_df.csv")
    new_data_df = clean_df(new_data_df)
    
    # Check for new rows based on the 'link' column
    if not processed_df.empty:
        # Find links in new_data_df that are not in processed_df
        new_links = ~new_data_df['link'].isin(processed_df['link'])
        new_rows_df = new_data_df[new_links]
    else:
        new_rows_df = new_data_df

    if not new_rows_df.empty:
        logging.info(f"Found {len(new_rows_df)} new rows. Processing...")
        main(clean_df(new_rows_df))  # Call the main processing function with only the new rows
    else:
        logging.info("No new rows found.")


def parse_info(response):
    try:
        response_json = json.loads(response)
        info = {
            'keywords': str(response_json['RelevantKeywords']),
            'llm_summary': response_json['ShortSummary'],
            'relevance': response_json['Relevance'],
            'relevance_reason': response_json['RelevanceReason'],
            'alertflag': response_json['AlertFlag'],
        }
        logging.info("Parsed response successfully")
        return info
    except Exception as e:
        logging.error(f"Error parsing response: {e}")
        logging.error(f"Response: {response}")
        return {'keywords': "Error", 'summary': "Error", 'relevance': "Error", 'relevance_reason': "Error", 'alertflag': "Error"}

def read_files():
    with open("./prompt/response_format.txt", "r") as f:
        response_format = f.read()
    with open("./prompt/prompt.txt", "r") as f:
        prompt = f.read()
    logging.info("Read files successfully")
    return response_format, prompt

def process_dataframe(df, client, prompt, response_format):
    results = {
        'keywords': [], 'llm_summary': [], 'relevance': [], 'relevance_reason': [], 
        'alertflag': [], 'llm_response': [], 'pricing': []
    }

    for _, row in df.iterrows():
        news_article = row["summary"]
        input_prompt = prompt.format(news_article=news_article, response_format=response_format)
        response, price, token_usage = api_call(client, input_prompt)
        logging.debug(response)
        
        if response:
            parsed_info = parse_info(response)

            for key in results:
                if key in parsed_info:
                    results[key].append(parsed_info[key])
                elif key == 'llm_response':
                    results[key].append(response)
                elif key == 'pricing':
                    results[key].append(price)

    logging.info("Processed dataframe successfully")
    for key, value in results.items():
        df[key] = value

    return df

def save_dataframe(df, filename="./data/newscatcher_df_with_response.csv"):
    if os.path.exists(filename):
        df_old = pd.read_csv(filename)
        df_new = pd.concat([df_old, df], ignore_index=True)
        df_new.drop_duplicates(inplace=True)
        df = df_new
    df.to_csv(filename, index=False)
    logging.info(f"Saved dataframe to {filename}")

def main(df):
    logging.info("Starting main process")
    config = load_environment_variables()
    client = initialize_client(config)
    response_format, prompt = read_files()
    processed_df = process_dataframe(df, client, prompt, response_format)
    save_dataframe(processed_df)
    logging.info("Main process completed")
    
if __name__ == "__main__":
    logging.info("Script started")
    # Schedule the check_and_process_new_data function to run every minute
    schedule.every(1).minutes.do(check_and_process_new_data)

    # Keep running in a loop to execute scheduled tasks
    while True:
        schedule.run_pending()
        time.sleep(1)
