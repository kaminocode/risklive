from dotenv import load_dotenv
from openai import AzureOpenAI
import os 
import logging

def load_environment_variables():
    load_dotenv()
    return {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "api_version": "2023-05-15",
        "azure_endpoint": os.getenv("OPENAI_API_BASE")
    }

def initialize_client(config):
    client = AzureOpenAI(
        api_key=config["api_key"],
        api_version=config["api_version"],
        azure_endpoint=config["azure_endpoint"]
    )
    return client

def pricing(token_usage):
    price = (token_usage.prompt_tokens*0.003 + token_usage.completion_tokens*0.0004)/1000
    return price

def api_call(client, prompt, temperature=0, max_tokens=500, top_p=0):
    
    try:
        response = client.chat.completions.create(
            model="openai_chat",
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            top_p=float(top_p),
            messages=[
                {"role": "system", "content": ""},
                {"role": "user", "content": prompt},
            ],
        )
        response_output = response.choices[0].message.content
        token_usage = response.usage
        price = pricing(token_usage)
        logging.info("API call successful")
        logging.info(f"Price: {price} USD")
        logging.info(f"Response: {response_output}")
        return response_output, price, token_usage
    except Exception as e:
        logging.error(f"API call failed: {e}")
        return None, None, None