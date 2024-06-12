import os

import requests

from crewai import Agent
from langchain.tools import tool
from handler import CustomHandler

def get_technical_indicators_agent(chosen_llm):
  return Agent(
          role='The Best Technical Indicators Researcher and Analyst for Company Stocks',
          goal="""Being the best at gathering and interpreting Technical Indicators relevant to the
                  future prospects of the company that the customer is interested in""",
          backstory="""The most seasoned and experienced technical indicators researcher and analyst with
      lots of expertise in understanding which technical indicators 
      are most relevant to the future prospects of companies that customers are interested in.
      Objective, unbiased approach to sorting through various technical indicators.""",
          verbose=True,
          allow_delegation=False,
          tools=[
              TechnicalIndicatorsTools.get_technical_indicators,
          ],
          llm=chosen_llm,
          callbacks=[CustomHandler("Technical Indicators Agent")]
      )

class TechnicalIndicatorsTools():
  @tool("Get technical indicators")
  def get_technical_indicators(stock_ticker):
    """
    Get technical indicators from the Internet for this company.
    The input to this tool is the company stock ticker symbol.
    """
    url_nasdaq = f"https://scanner.tradingview.com/symbol?symbol=NASDAQ:{stock_ticker}&fields=Recommend.Other,Recommend.All,Recommend.MA,RSI,RSI%5B1%5D,Stoch.K,Stoch.D,Stoch.K%5B1%5D,Stoch.D%5B1%5D,CCI20,CCI20%5B1%5D,ADX,ADX+DI,ADX-DI,ADX+DI%5B1%5D,ADX-DI%5B1%5D,AO,AO%5B1%5D,AO%5B2%5D,Mom,Mom%5B1%5D,MACD.macd,MACD.signal,Rec.Stoch.RSI,Stoch.RSI.K,Rec.WR,W.R,Rec.BBPower,BBPower,Rec.UO,UO,EMA10,close,SMA10,EMA20,SMA20,EMA30,SMA30,EMA50,SMA50,EMA100,SMA100,EMA200,SMA200,Rec.Ichimoku,Ichimoku.BLine,Rec.VWMA,VWMA,Rec.HullMA9,HullMA9,Pivot.M.Classic.S3,Pivot.M.Classic.S2,Pivot.M.Classic.S1,Pivot.M.Classic.Middle,Pivot.M.Classic.R1,Pivot.M.Classic.R2,Pivot.M.Classic.R3,Pivot.M.Fibonacci.S3,Pivot.M.Fibonacci.S2,Pivot.M.Fibonacci.S1,Pivot.M.Fibonacci.Middle,Pivot.M.Fibonacci.R1,Pivot.M.Fibonacci.R2,Pivot.M.Fibonacci.R3,Pivot.M.Camarilla.S3,Pivot.M.Camarilla.S2,Pivot.M.Camarilla.S1,Pivot.M.Camarilla.Middle,Pivot.M.Camarilla.R1,Pivot.M.Camarilla.R2,Pivot.M.Camarilla.R3,Pivot.M.Woodie.S3,Pivot.M.Woodie.S2,Pivot.M.Woodie.S1,Pivot.M.Woodie.Middle,Pivot.M.Woodie.R1,Pivot.M.Woodie.R2,Pivot.M.Woodie.R3,Pivot.M.Demark.S1,Pivot.M.Demark.Middle,Pivot.M.Demark.R1&no_404=true"
    url_nyse = f"https://scanner.tradingview.com/symbol?symbol=NYSE:{stock_ticker}&fields=Recommend.Other,Recommend.All,Recommend.MA,RSI,RSI%5B1%5D,Stoch.K,Stoch.D,Stoch.K%5B1%5D,Stoch.D%5B1%5D,CCI20,CCI20%5B1%5D,ADX,ADX+DI,ADX-DI,ADX+DI%5B1%5D,ADX-DI%5B1%5D,AO,AO%5B1%5D,AO%5B2%5D,Mom,Mom%5B1%5D,MACD.macd,MACD.signal,Rec.Stoch.RSI,Stoch.RSI.K,Rec.WR,W.R,Rec.BBPower,BBPower,Rec.UO,UO,EMA10,close,SMA10,EMA20,SMA20,EMA30,SMA30,EMA50,SMA50,EMA100,SMA100,EMA200,SMA200,Rec.Ichimoku,Ichimoku.BLine,Rec.VWMA,VWMA,Rec.HullMA9,HullMA9,Pivot.M.Classic.S3,Pivot.M.Classic.S2,Pivot.M.Classic.S1,Pivot.M.Classic.Middle,Pivot.M.Classic.R1,Pivot.M.Classic.R2,Pivot.M.Classic.R3,Pivot.M.Fibonacci.S3,Pivot.M.Fibonacci.S2,Pivot.M.Fibonacci.S1,Pivot.M.Fibonacci.Middle,Pivot.M.Fibonacci.R1,Pivot.M.Fibonacci.R2,Pivot.M.Fibonacci.R3,Pivot.M.Camarilla.S3,Pivot.M.Camarilla.S2,Pivot.M.Camarilla.S1,Pivot.M.Camarilla.Middle,Pivot.M.Camarilla.R1,Pivot.M.Camarilla.R2,Pivot.M.Camarilla.R3,Pivot.M.Woodie.S3,Pivot.M.Woodie.S2,Pivot.M.Woodie.S1,Pivot.M.Woodie.Middle,Pivot.M.Woodie.R1,Pivot.M.Woodie.R2,Pivot.M.Woodie.R3,Pivot.M.Demark.S1,Pivot.M.Demark.Middle,Pivot.M.Demark.R1&no_404=true"

    data_nasdaq = requests.get(url=url_nasdaq)
    data_nyse = requests.get(url=url_nyse)

    # combine both data (since the company could be either on NYSE or NASDAQ)
    return str(data_nasdaq) + " " + str(data_nyse)

  
