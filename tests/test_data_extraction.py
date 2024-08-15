import unittest
from unittest.mock import patch
import requests_mock
import pandas as pd
from datetime import datetime
from risklive.data_extraction.bing_api import BingAPI, extract_trending_topics, extract_news_by_category, search_news, search_news_for_trending_topics, aggregate_news_data

class TestBingAPI(unittest.TestCase):

    @patch('data_extraction.bing_api.BING_API_KEY', 'fake_api_key')
    @requests_mock.Mocker()
    def test_get_trending_topics(self, mock_request):
        mock_response = {
            "value": [
                {
                    "name": "Sample Trending Topic",
                    "webSearchUrl": "http://example.com/search",
                    "newsSearchUrl": "http://example.com/news",
                    "isBreakingNews": False
                }
            ]
        }
        mock_request.get("https://api.bing.microsoft.com/v7.0/news/trendingtopics", json=mock_response)
        
        since = int((datetime.now() - pd.DateOffset(days=3)).timestamp())
        bing_api = BingAPI(api_key='fake_api_key')
        response = bing_api.get_trending_topics(since)
        
        self.assertEqual(response, mock_response)

    @patch('data_extraction.bing_api.BING_API_KEY', 'fake_api_key')
    @requests_mock.Mocker()
    def test_get_news_by_category(self, mock_request):
        mock_response = {
            "value": [
                {
                    "name": "Sample News Article",
                    "url": "http://example.com/article",
                    "description": "This is a sample news article."
                }
            ]
        }
        mock_request.get("https://api.bing.microsoft.com/v7.0/news", json=mock_response)
        
        since = int((datetime.now() - pd.DateOffset(days=3)).timestamp())
        bing_api = BingAPI(api_key='fake_api_key')
        response = bing_api.get_news_by_category('Business', since)
        
        self.assertEqual(response, mock_response)

    @patch('data_extraction.bing_api.BING_API_KEY', 'fake_api_key')
    @requests_mock.Mocker()
    def test_search_news(self, mock_request):
        mock_response = {
            "value": [
                {
                    "name": "Sample Search Article",
                    "url": "http://example.com/search_article",
                    "description": "This is a sample search article."
                }
            ]
        }
        mock_request.get("https://api.bing.microsoft.com/v7.0/news/search", json=mock_response)
        
        since = int((datetime.now() - pd.DateOffset(days=3)).timestamp())
        bing_api = BingAPI(api_key='fake_api_key')
        response = bing_api.search_news('nuclear', since=since)
        
        self.assertEqual(response, mock_response)


class TestDataExtractionFunctions(unittest.TestCase):

    @patch('data_extraction.bing_api.BING_API_KEY', 'fake_api_key')
    @requests_mock.Mocker()
    def test_extract_trending_topics(self, mock_request):
        mock_response = {
            "value": [
                {
                    "name": "Sample Trending Topic",
                    "webSearchUrl": "http://example.com/search",
                    "newsSearchUrl": "http://example.com/news",
                    "isBreakingNews": False
                }
            ]
        }
        mock_request.get("https://api.bing.microsoft.com/v7.0/news/trendingtopics", json=mock_response)
        
        since = int((datetime.now() - pd.DateOffset(days=3)).timestamp())
        df = extract_trending_topics(since)
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1)
        self.assertListEqual(list(df.columns), ['Name', 'WebSearchURL', 'NewsSearchURL', 'IsBreakingNews', 'Timestamp'])

    @patch('data_extraction.bing_api.BING_API_KEY', 'fake_api_key')
    @requests_mock.Mocker()
    def test_extract_news_by_category(self, mock_request):
        mock_response = {
            "value": [
                {
                    "name": "Sample News Article",
                    "url": "http://example.com/article",
                    "description": "This is a sample news article."
                }
            ]
        }
        mock_request.get("https://api.bing.microsoft.com/v7.0/news", json=mock_response)
        
        since = int((datetime.now() - pd.DateOffset(days=3)).timestamp())
        df = extract_news_by_category('Business', since)
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1)
        self.assertListEqual(list(df.columns), ['Title', 'URL', 'Description', 'Timestamp'])

    @patch('data_extraction.bing_api.BING_API_KEY', 'fake_api_key')
    @requests_mock.Mocker()
    def test_search_news(self, mock_request):
        mock_response = {
            "value": [
                {
                    "name": "Sample Search Article",
                    "url": "http://example.com/search_article",
                    "description": "This is a sample search article."
                }
            ]
        }
        mock_request.get("https://api.bing.microsoft.com/v7.0/news/search", json=mock_response)
        
        since = int((datetime.now() - pd.DateOffset(days=3)).timestamp())
        df = search_news('nuclear', since)
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1)
        self.assertListEqual(list(df.columns), ['Title', 'URL', 'Description', 'Timestamp'])

    @patch('data_extraction.bing_api.BING_API_KEY', 'fake_api_key')
    @requests_mock.Mocker()
    def test_search_news_for_trending_topics(self, mock_request):
        mock_response_trending = {
            "value": [
                {
                    "name": "Sample Trending Topic",
                    "webSearchUrl": "http://example.com/search",
                    "newsSearchUrl": "http://example.com/news",
                    "isBreakingNews": False
                }
            ]
        }
        mock_response_search = {
            "value": [
                {
                    "name": "Sample Search Article",
                    "url": "http://example.com/search_article",
                    "description": "This is a sample search article."
                }
            ]
        }
        mock_request.get("https://api.bing.microsoft.com/v7.0/news/trendingtopics", json=mock_response_trending)
        mock_request.get("https://api.bing.microsoft.com/v7.0/news/search", json=mock_response_search)
        
        since = int((datetime.now() - pd.DateOffset(days=3)).timestamp())
        df = search_news_for_trending_topics(since)
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        self.assertListEqual(list(df.columns), ['Title', 'URL', 'Description', 'Timestamp'])

    @patch('data_extraction.bing_api.BING_API_KEY', 'fake_api_key')
    @patch('data_extraction.bing_api.CATEGORIES', ['Business'])
    @patch('data_extraction.bing_api.QUERIES', ['nuclear'])
    @requests_mock.Mocker()
    def test_aggregate_news_data(self, mock_request):
        mock_response_trending = {
            "value": [
                {
                    "name": "Sample Trending Topic",
                    "webSearchUrl": "http://example.com/search",
                    "newsSearchUrl": "http://example.com/news",
                    "isBreakingNews": False
                }
            ]
        }
        mock_response_news = {
            "value": [
                {
                    "name": "Sample News Article",
                    "url": "http://example.com/article",
                    "description": "This is a sample news article."
                }
            ]
        }
        mock_request.get("https://api.bing.microsoft.com/v7.0/news/trendingtopics", json=mock_response_trending)
        mock_request.get("https://api.bing.microsoft.com/v7.0/news", json=mock_response_news)
        mock_request.get("https://api.bing.microsoft.com/v7.0/news/search", json=mock_response_news)
        
        df = aggregate_news_data()
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        self.assertListEqual(list(df.columns), ['Title', 'URL', 'Description', 'Timestamp'])

if __name__ == '__main__':
    unittest.main()
