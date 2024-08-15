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

## Configuration

- `config/config.yml`: Main configuration file
- `.env`: Environment-specific secrets and API keys

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.