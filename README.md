# RiskLive

RiskLive is a real-time risk analysis dashboard for the nuclear industry. It aggregates news and data from various sources, processes the information using advanced natural language processing techniques, and presents insights through an interactive web interface.

## Features

- Real-time news aggregation from multiple sources
- Automated information extraction using LLM (Large Language Models)
- Topic modeling for trend analysis
- Interactive web dashboard for data visualization
- Scheduled tasks for regular data updates and maintenance

## Technology Stack

- Python
- Flask for web server
- Pandas for data manipulation
- OpenAI's API for LLM-based processing
- Bing API for news aggregation
- APScheduler for task scheduling

## Project Structure
```
risklive/
├── src/
│   └── risklive/
│       ├── data_extraction/
│       ├── data_processing/
│       ├── topic_modeling/
│       └── server/
├── tests/
├── notebooks/
├── config/
├── .env
├── setup.py
├── requirements.txt
└── README.md
```

- `src/risklive/`: Main package source code
- `tests/`: Unit and integration tests
- `notebooks/`: Jupyter notebooks for analysis and development
- `config/`: Configuration files
- `.env`: Environment variables (not version controlled)

## Installation

You can install RiskLive using either `pip` with the `requirements.txt` file or by using `setup.py`. Choose the method that best suits your needs.

### Method 1: Using requirements.txt

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/risklive.git
   cd risklive
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

### Method 2: Using setup.py

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/risklive.git
   cd risklive
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the package:
   ```
   pip install .
   ```

   Or, for development mode:
   ```
   pip install -e .
   ```

## Configuration

- `config/config.yml`: Main configuration file
- `.env`: Environment-specific secrets and API keys

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.