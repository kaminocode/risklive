# Topic Modeling Server README

## Overview
This server is designed to perform advanced topic modeling and summarization on news articles using BERTopic for topic modeling and GPT-4 for summarization. It processes updated datasets to identify topics within the news articles and generates concise summaries for each topic.

## Features
- **Topic Modeling**: Utilizes BERTopic to extract topics from news articles based on their content.
- **GPT-4 Summarization**: Generates summaries for each identified topic using Azure's implementation of OpenAI's GPT-4.
- **Continuous Monitoring**: Scheduled checks for new or updated data files and processes them as needed.
- **Data Cleaning**: Filters relevant rows and handles missing values to prepare data for modeling.
- **Visualization**: Generates and saves visualizations of the topic models, including topic hierarchies and topic distributions over time.
- **Logging**: Comprehensive logging of processing steps and errors for debugging and monitoring.

## Requirements
- Python 3.x
- Pandas library
- Sentence Transformers library
- BERTopic library
- UMAP-learn
- HDBSCAN
- Scikit-learn
- Dotenv for environment variable management
- OpenAI library for Azure OpenAI integration

## Setup
1. Clone the repository to your local machine.
2. Install the required Python libraries: `pip install pandas sentence-transformers bertopic umap-learn hdbscan scikit-learn python-dotenv openai`.
3. Create a `.env` file in the root directory and add your Azure OpenAI credentials and any other necessary configuration variables.
4. Ensure initial news data CSV file is located in the `./data` directory.

## Usage
To start the server and begin processing:
1. Navigate to the server's directory.
2. Run the script: `python topic_modeling.py`.
3. The script will continuously monitor for new or updated news data files and process them as needed.

The server checks for new updates, performs topic modeling to identify distinct topics within the news articles, and generates summaries for each topic using GPT-4. The process enriches the dataset with insights into the thematic structures of the news data and provides summaries that capture the essence of each topic.

## How It Works
1. **Data Preparation**: Loads and cleans the data, preparing it for topic modeling.
2. **Model Initialization**: Initializes BERTopic and sentence transformer models based on the size of the dataset.
3. **Topic Modeling**: Identifies topics within the news articles using BERTopic.
4. **Summarization**: For each identified topic, generates a summary using GPT-4.
5. **Visualization and Saving**: Generates visualizations of the topic models and saves the enriched dataset with identified topics and summaries.

## Customization
- Adjust the BERTopic and sentence transformer model parameters based on the specific characteristics of your dataset.
- Modify the scheduled check frequency to suit your data update frequency.
