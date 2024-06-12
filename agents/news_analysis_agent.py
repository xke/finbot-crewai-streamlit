import json
import os

import requests
from langchain.tools import tool
from crewai import Agent

# TODO: seems like some issue with using YahooFinanceNewsTool
# from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from handler import CustomHandler

#import weave

#@weave.op()
def get_news_analysis_agent(chosen_llm):
  return Agent(
          role='The More Insightful News Researcher and Analyst',
          goal="""Being the best at gathering and interpreting news data relevant to the
                  future prospects of the company that the customer is interested in""",
          backstory="""The most seasoned and experienced news researcher and analyst with
      lots of expertise in understanding which news, company announcements, and market sentiments
      are most relevant to the future prospects of companies that customers are interested in.
      Objective, unbiased approach to sorting through various news and insights.""",
          verbose=True,
          allow_delegation=False,
          tools=[
              SearchTools.search_internet,
              SearchTools.search_news,
              #YahooFinanceNewsTool(),
          ],
          llm=chosen_llm,
          callbacks=[CustomHandler("News Analysis Agent")]
      )


class SearchTools():
  @tool("Search the internet")
  def search_internet(query):
    """Useful to search the internet
    about a a given topic and return relevant results"""
    top_result_to_return = 4
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query})
    headers = {
        'X-API-KEY': os.environ['SERPER_API_KEY'],
        'content-type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    results = response.json()['organic']
    string = []
    for result in results[:top_result_to_return]:
      try:
        string.append('\n'.join([
            f"Title: {result['title']}", f"Link: {result['link']}",
            f"Snippet: {result['snippet']}", "\n-----------------"
        ]))
      except KeyError:
        next

    return '\n'.join(string)

  @tool("Search news on the internet")
  def search_news(query):
    """Useful to search news about a company, stock or any other
    topic and return relevant results"""""
    top_result_to_return = 4
    url = "https://google.serper.dev/news"
    payload = json.dumps({"q": query})
    headers = {
        'X-API-KEY': os.environ['SERPER_API_KEY'],
        'content-type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    results = response.json()['news']
    string = []
    for result in results[:top_result_to_return]:
      try:
        string.append('\n'.join([
            f"Title: {result['title']}", f"Link: {result['link']}",
            f"Snippet: {result['snippet']}", "\n-----------------"
        ]))
      except KeyError:
        next

    return '\n'.join(string)