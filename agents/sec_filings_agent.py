import os

import requests

from langchain.tools import tool
from langchain.text_splitter import CharacterTextSplitter
#from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
#from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from sec_api import QueryApi
from unstructured.partition.html import partition_html
from crewai import Agent

from handler import CustomHandler
#import weave

#@weave.op()
def get_sec_filings_agent(chosen_llm):
  return Agent(
          role='The More Insightful Edgar SEC Filings Researcher and Analyst',
          goal="""Being the best at gathering and interpreting Edgar SEC Filings data relevant to the
                  future prospects of the company that the customer is interested in""",
          backstory="""The most seasoned and experienced Edgar SEC filings researcher and analyst with
      lots of expertise in understanding which information is most relevant to the future prospects of 
      companies that customers are interested in. Objective, unbiased approach to sorting through 
      various information in the SEC filings.""",
          verbose=True,
          allow_delegation=False,
          tools=[
              SECTools.search_10q,
              SECTools.search_10k
          ],
          llm=chosen_llm,
          callbacks=[CustomHandler("SEC Filings Agent")]
      )

# TODO: why is it that this doesn't work with GPT-3.5 but does work with GPT-4?
# seems to work with gemma also

class SECTools():
  @tool("Search 10-Q form")
  def search_10q(stock, question):
    """
    Useful to search information from the latest 10-Q form for a
    given stock.

    The input to this tool should be a stock ticker representing the company
    you are interested in. Come up with one or two questions that would be
    highly relevant to someone wanting to know the future
    prospect of the company with this ticker symbol. 
    """
    queryApi = QueryApi(api_key=os.environ['SEC_API_API_KEY'])
    query = {
      "query": {
        "query_string": {
          "query": f"ticker:{stock} AND formType:\"10-Q\""
        }
      },
      "from": "0",
      "size": "1",
      "sort": [{ "filedAt": { "order": "desc" }}]
    }

    fillings = queryApi.get_filings(query)['filings']
    if len(fillings) == 0:
      return "Sorry, I couldn't find any filling for this stock, check if the ticker is correct."
    link = fillings[0]['linkToFilingDetails']
    answer = SECTools.__embedding_search(link, question)
    return answer

  @tool("Search 10-K form")
  def search_10k(stock, question):
    """
    Useful to search information from the latest 10-K form for a
    given stock.
    
    The input to this tool should be a stock ticker representing the company
    you are interested in. Come up with one or two questions that would be
    highly relevant to someone wanting to know the future
    prospect of the company with this ticker symbol.
    """
    queryApi = QueryApi(api_key=os.environ['SEC_API_API_KEY'])
    query = {
      "query": {
        "query_string": {
          "query": f"ticker:{stock} AND formType:\"10-K\""
        }
      },
      "from": "0",
      "size": "1",
      "sort": [{ "filedAt": { "order": "desc" }}]
    }

    fillings = queryApi.get_filings(query)['filings']
    if len(fillings) == 0:
      return "Sorry, I couldn't find any filling for this stock, check if the ticker is correct."
    link = fillings[0]['linkToFilingDetails']
    answer = SECTools.__embedding_search(link, question)
    return answer

  def __embedding_search(url, ask):
    text = SECTools.__download_form_html(url)
    elements = partition_html(text=text)
    content = "\n".join([str(el) for el in elements])
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap  = 150,
        length_function = len,
        is_separator_regex = False,
    )

     
    docs = text_splitter.create_documents([content])

    # Faiss retriever with OpenAI embeddings
    #
    #retriever = FAISS.from_documents(
    #  docs, OpenAIEmbeddings()
    #).as_retriever()

    # alternative embedding; Chroma takes a long time to load and times out the CrewAI telemetry
    #embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    #retriever = Chroma.from_documents(docs, embedding_function).as_retriever()

    retriever = FAISS.from_documents(
       docs, HuggingFaceEmbeddings()
    ).as_retriever()

    answers = retriever.invoke(ask, top_k=4)
    answers = "\n\n".join([a.page_content for a in answers])
    return answers

  def __download_form_html(url):
    headers = {
      'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
      'Accept-Encoding': 'gzip, deflate, br',
      'Accept-Language': 'en-US,en;q=0.9,pt-BR;q=0.8,pt;q=0.7',
      'Cache-Control': 'max-age=0',
      'Dnt': '1',
      'Sec-Ch-Ua': '"Not_A Brand";v="8", "Chromium";v="120"',
      'Sec-Ch-Ua-Mobile': '?0',
      'Sec-Ch-Ua-Platform': '"macOS"',
      'Sec-Fetch-Dest': 'document',
      'Sec-Fetch-Mode': 'navigate',
      'Sec-Fetch-Site': 'none',
      'Sec-Fetch-User': '?1',
      'Upgrade-Insecure-Requests': '1',
      'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    response = requests.get(url, headers=headers)
    return response.text