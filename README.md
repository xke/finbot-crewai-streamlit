# finbot-crewai-streamlit

deployed at https://finbot-crewai.streamlit.app/

environment variables needed:

* GROQ_API_KEY: For the `llama3-8b-8192`, `mixtral-8x7b-32768`, and `gemma-7b-it` AI models
* OPENAI_API_KEY: For the OpenAI `gpt` models
* SEC_API_API_KEY: For the SEC filings agent (see `agents/sec_filings_agent.py`)
* SERPER_API_KEY: For the news analysis agent (see `agents/news_analysis_agent.py`)
* WANDB_API_KEY: For the Weights & Biases' Weave integration to log inputs & outputs
