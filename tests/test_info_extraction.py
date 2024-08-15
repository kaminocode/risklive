import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from risklive.data_processing import extract_information, parse_json_info, process_df, extract_tokens
from risklive.data_processing import initialize_client, api_call

class TestInfoExtractionFunctions(unittest.TestCase):

    @patch('data_processing.initialize_client')
    @patch('data_processing.api_call')
    def test_extract_information(self, mock_api_call, mock_initialize_client):
        mock_initialize_client.return_value = MagicMock()
        mock_api_call.return_value = ({"key": "value"}, 0.01, MagicMock(prompt_tokens=10, completion_tokens=20))
        
        client = initialize_client()
        news_content = "This is a test news content."
        response, price, token_usage = extract_information(client, news_content)
        self.assertIsNotNone(response)
        self.assertIsNotNone(price)
        self.assertIsNotNone(token_usage)

    def test_parse_json_info(self):
        response = {
            'RelevantKeywords': ['test', 'news'],
            'ShortSummary': 'This is a test summary.',
            'Relevance': 'Yes',
            'RelevanceReason': 'Test reason.',
            'AlertFlag': 'Red',
            'AlertReason': 'Test alert reason.'
        }
        parsed_info = parse_json_info(response)
        self.assertEqual(parsed_info['RelevantKeywords'], 'test, news')
        self.assertEqual(parsed_info['ShortSummary'], 'This is a test summary.')
        self.assertEqual(parsed_info['Relevance'], 'Yes')
        self.assertEqual(parsed_info['RelevanceReason'], 'Test reason.')
        self.assertEqual(parsed_info['AlertFlag'], 'Red')
        self.assertEqual(parsed_info['AlertReason'], 'Test alert reason.')

    def test_extract_tokens(self):
        token_usage = MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        tokens = extract_tokens(token_usage)
        self.assertEqual(tokens['PromptTokens'], 10)
        self.assertEqual(tokens['CompletionTokens'], 20)
        self.assertEqual(tokens['TotalTokens'], 30)

    @patch('data_processing.extract_information')
    @patch('data_processing.initialize_client')
    def test_process_df(self, mock_initialize_client, mock_extract_information):
        mock_initialize_client.return_value = MagicMock()
        mock_extract_information.return_value = (
            {
                'RelevantKeywords': 'test, news',
                'ShortSummary': 'This is a test summary.',
                'Relevance': 'Yes',
                'RelevanceReason': 'Test reason.',
                'AlertFlag': 'Red',
                'AlertReason': 'Test alert reason.'
            },
            0.01,
            MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        )
        
        data = {'Description': ["This is a test news content."]}
        df = pd.DataFrame(data)
        processed_df = process_df(df)
        
        self.assertIn('LLM_Response', processed_df.columns)
        self.assertIn('LLM_Price', processed_df.columns)
        self.assertIn('LLM_Token_Usage', processed_df.columns)
        self.assertIn('RelevantKeywords', processed_df.columns)
        self.assertIn('ShortSummary', processed_df.columns)
        self.assertIn('Relevance', processed_df.columns)
        self.assertIn('RelevanceReason', processed_df.columns)
        self.assertIn('AlertFlag', processed_df.columns)
        self.assertIn('AlertReason', processed_df.columns)
        self.assertIn('PromptTokens', processed_df.columns)
        self.assertIn('CompletionTokens', processed_df.columns)
        self.assertIn('TotalTokens', processed_df.columns)

if __name__ == '__main__':
    unittest.main()
