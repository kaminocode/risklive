readme_content = """
# Data Extraction Server README

## Overview
This server is designed to fetch the latest news headlines and articles based on specific keywords by making API calls to Newscatcher API. It is implemented in Python and uses scheduled tasks to periodically update the data.

## Features
- **Automated News Fetching**: Fetches latest headlines and searches for articles based on predefined keywords.
- **Data Cleaning**: Removes unwanted summaries and duplicates from the fetched data.
- **Scheduled Fetching**: Utilizes APScheduler to schedule fetching tasks on an hourly and daily basis.
- **Logging**: Maintains logs of the fetching process and errors.

## News Sources
The server fetches news from the following sources:
- The Guardian (`theguardian.com`)
- Forbes (`forbes.com`)
- The Wall Street Journal (`wsj.com`)
- The Economist (`economist.com`)
- BBC (`bbc.com` and `bbc.co.uk`)
- Wired (`wired.com`)
- The Verge (`theverge.com`)
- Reuters (`reuters.com`)
- Al Jazeera (`aljazeera.com`)

## Endpoints Used
1. **Latest Headlines Endpoint**: Fetches the latest headlines within the last hour from specified news sources.
   - URL: `https://api.newscatcherapi.com/v2/latest_headlines`
   - Parameters: `when`, `page`, `page_size`, `lang`, `countries`, `sources`
2. **Search Endpoint**: Searches for articles based on keywords within the last 2 days from specified news sources.
   - URL: `https://api.newscatcherapi.com/v2/search`
   - Parameters: `q` (keyword), `lang`, `to_rank`, `page_size`, `page`, `from`, `sources`

## Requirements
- Python 3.x
- Requests library
- Pandas library
- APScheduler
- python-dotenv

## Setup
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Set your Newscatcher API key in a `.env` file as `NEWSCATCHER_API_KEY=your_api_key_here`.
4. Run the server: `python path/to/server_script.py`

## Usage
The server automatically starts fetching data based on the configured schedule:
- **Hourly News Fetch**: Every hour, the latest headlines are fetched.
- **Daily Keyword Fetch**: Once every 24 hours, articles related to predefined keywords are fetched.

Data is saved in CSV format in the `./data` directory.

## License
[Specify License]

## Contribution
Contributions are welcome. Please submit a pull request or open an issue for any features or bug fixes.
