# README

## Overview
This system streamlines the extraction, tagging, and analysis of news articles. It operates through three interconnected stages: extracting news data, tagging articles with LLM-generated metadata, and performing topic modeling with summarization for analysis.

## Process Flow
1. **Data Extraction**: Automated fetching of latest news and articles based on specific keywords, updated hourly and daily.
2. **News Processing**: Tagging of each article with summaries, relevance, alert flags, and other metadata for insight generation.
3. **Topic Modeling and Summarization**: Identifying topics within relevant articles using BERTopic and summarizing them with GPT-4 for risk analysis.

## Components Overview
- **Data Extraction Server**: Extracts news data using Newscatcher API, focusing on predefined keywords and sources.
- **News Processing Server**: Enhances articles with GPT-4 generated summaries and relevance tagging.
- **Topic Modeling and Summarization Server**: Applies BERTopic for modeling and GPT-4 for summarizing topics, aiding in intuitive risk analysis.

## Usage
Follow individual component READMEs in docs/ for setup and usage instructions. Ensure each server is correctly configured for integrated operation.
