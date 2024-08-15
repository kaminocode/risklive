import unittest
from unittest.mock import patch, MagicMock
import openai
from data_processing import initialize_client, pricing, load_prompt_template, format_prompt, api_call

class TestLLMFunctions(unittest.TestCase):

    @patch('openai.AzureOpenAI')
    def test_initialize_client(self, mock_azure_openai):
        mock_azure_openai.return_value = MagicMock()
        client = initialize_client()
        self.assertIsNotNone(client)
        mock_azure_openai.assert_called_once()

    def test_pricing(self):
        token_usage = MagicMock()
        token_usage.prompt_tokens = 1000
        token_usage.completion_tokens = 500
        
        price = pricing(token_usage, "openai_chat")
        self.assertLess(price, 0.01)

        price = pricing(token_usage, "gpt4")
        self.assertLess(price, 0.1)

        price = pricing(token_usage, "gpt4o")
        self.assertLess(price, 0.1)

    def test_load_prompt_template(self):
        with patch('builtins.open', unittest.mock.mock_open(read_data="prompt content")) as mock_file:
            content = load_prompt_template("dummy_path")
            self.assertEqual(content, "prompt content")
            mock_file.assert_called_once_with("dummy_path", 'r')

    def test_format_prompt(self):
        prompt_template = "Hello, {name}!"
        formatted_prompt = format_prompt(prompt_template, name="World")
        self.assertEqual(formatted_prompt, "Hello, World!")

    @patch('data_processing.pricing')
    @patch('openai.AzureOpenAI')
    def test_api_call(self, mock_azure_openai, mock_pricing):
        client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"key": "value"}'
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20)
        
        client.chat.completions.create.return_value = mock_response
        mock_pricing.return_value = 0.005

        response, price, token_usage = api_call(client, "gpt4o", "user_prompt")
        
        self.assertEqual(response, {"key": "value"})
        self.assertLess(price, 0.005)
        self.assertEqual(token_usage.prompt_tokens, 10)
        self.assertEqual(token_usage.completion_tokens, 20)

if __name__ == '__main__':
    unittest.main()
