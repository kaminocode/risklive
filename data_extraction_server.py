import os
import logging
import requests
import pandas as pd
import json
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv
from requests.exceptions import RequestException
import time
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='./logs/news_fetcher.log'
)
logger = logging.getLogger(__name__)
num_api_calls = 0 
# Load environment variables
load_dotenv()
X_API_KEY = os.getenv("NEWSCATCHER_API_KEY")

def clean_df(df):
    remove_summary_list = ["Continue reading on Medium »", "Please enable JS and disable any ad blocker"]
    df = df[~df.summary.isin(remove_summary_list)]
    df = df.drop_duplicates(subset=['link'])
    return df

def get_headlines():
    global num_api_calls
    if not X_API_KEY:
        logger.error("API key is not set. Please check your .env configuration.")
        return

    base_url = 'https://api.newscatcherapi.com/v2/latest_headlines'
    headers = {'x-api-key': X_API_KEY}
    full_data_list = []
    params = {
        'when': '1h',
        'page': 1,
        'page_size': 100,
        'lang': 'en',
        'countries': 'US, UK',
        'sources': 'theguardian.com, forbes.com, wsj.com, economist.com, bbc.com, wired.com, theverge.com, reuters.com', 
    }
    try:
        response = requests.get(base_url, headers=headers, params=params)
        num_api_calls += 1
        time.sleep(2)  # Respectful delay between requests
        response.raise_for_status()  # Raises HTTPError for bad responses
        results = response.json()
        if 'articles' in results:
            full_data_list.extend(results['articles'])
            logger.info(f"Successfully fetched {len(full_data_list)} headline articles.")
        else:
            logger.warning(f"No headlines articles found or bad response structure.")
    except RequestException as e:
        logger.error(f"Request to get headlines failed: {e}")

    if full_data_list:
        df = pd.DataFrame(full_data_list)
        df = clean_df(df)
        file_path = './data/newscatcher_df.csv'
        try:
            if os.path.exists(file_path):
                df.to_csv(file_path, mode='a', header=False, index=False)
                logger.info("Appended new data to existing CSV file.")
            else:
                df.to_csv(file_path, index=False)
                logger.info("Created new CSV file and saved data.")
        except Exception as e:
            logger.error(f"Failed to save data to CSV: {e}")
    else:
        logger.info("No new data to append to the CSV file.")
    logger.info(f"Total API calls so far: {num_api_calls}")

def get_daily_news(keyword):
    global num_api_calls
    df = pd.DataFrame()
    base_url = 'https://api.newscatcherapi.com/v2/search'
    headers = {'x-api-key': X_API_KEY}
    params_page = 1
    while True:
        time.sleep(2)  # Respectful delay between requests
        params = {
            'q': keyword,
            'lang': 'en',
            'to_rank': 500,
            'page_size': 100,
            'page': params_page,
            'from': '2 day',
            'sources': 'theguardian.com, forbes.com, wsj.com, aljazeera.com, economist.com, bbc.co.uk, reuters.com, wired.com, theverge.com'
        }

        try:
            response = requests.get(base_url, headers=headers, params=params)
            num_api_calls += 1
            results = response.json()  # Directly use .json() method
            df_dict = results.get('articles', [])
            params_page += 1
            if not df_dict:  # No more data to process
                break
            df = pd.concat([df, pd.DataFrame(df_dict)], ignore_index=True)

            if response.status_code == 200 and (params_page > results.get('total_pages', 0) or params_page > 4):
                logger.info(f"Keyword '{keyword}' has {len(df)} articles")
                break
        except RequestException as e:
            logger.error(f"Failed to fetch data for keyword '{keyword}': {e}")
            break

    if not df.empty:
        file_path = './data/newscatcher_df.csv'
        df = clean_df(df)
        try:
            if os.path.exists(file_path):
                df.to_csv(file_path, mode='a', header=False, index=False)
            else:
                df.to_csv(file_path, index=False)
            logger.info(f"Data for keyword '{keyword}' appended to {file_path}")
            logger.info(f"Total API calls so far: {num_api_calls}")
        except Exception as e:
            logger.error(f"Failed to append data for keyword '{keyword}' to CSV: {e}")

def fetch_data_by_keywords():
    keywords = ['supply chain', 'job crisis', 'health and safety', 'cyber attacks', 'cyber security', 'layoffs', 'economic instability', 'war', \
        'unemployment', 'health crisis', 'cybersecurity risk', 'economic crisis', 'economy', 'supply chain crisis', 'living crisis']
    for keyword in keywords:
        get_daily_news(keyword)

def start_scheduler():
    now = datetime.now()
    next_hour = now + timedelta(hours=1)
    top_of_next_hour = next_hour.replace(minute=0, second=0, microsecond=0)
    delay_seconds = (top_of_next_hour - now).total_seconds()

    scheduler = BackgroundScheduler()
    # scheduler.add_job(get_headlines, 'interval', hours=1, next_run_time=datetime.now() + timedelta(seconds=delay_seconds))
    # scheduler.add_job(get_headlines, 'interval', hours=1)
    scheduler.add_job(get_headlines, 'interval', hours=1, next_run_time=datetime.now())
    # scheduler.add_job(fetch_data_by_keywords, 'interval', hours=24)
    # scheduler.add_job(fetch_data_by_keywords, trigger='cron', hour=8)
    scheduler.add_job(fetch_data_by_keywords, 'interval', hours=24, next_run_time=datetime.now()+ timedelta(seconds=30))

    scheduler.start()
    logger.info("Scheduler started. Fetching hourly news and daily keyword data.")

    try:
        while True:
            time.sleep(2)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        logger.info("Scheduler shut down successfully.")

if __name__ == '__main__':
    start_scheduler()
