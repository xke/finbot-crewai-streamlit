import streamlit as st
#import agentops
import os 

from crewai import Crew, Process, Agent, Task
from langchain_openai import ChatOpenAI  # triggers "generator" error on Python decorators https://github.com/microsoft/promptflow/pull/3179
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic

#from langchain_community.llms import HuggingFaceHub
#from langsmith import traceable
import weave

from agents.news_analysis_agent import get_news_analysis_agent
from agents.sec_filings_agent import get_sec_filings_agent
from agents.technical_indicators_agent import get_technical_indicators_agent

from textwrap import dedent
import os
from handler import CustomHandler

from dotenv import load_dotenv
load_dotenv()

from datetime import date

#@weave.op()
#@agentops.record_function('log_run')
def log_run(state, model, company, historical_horizon_in_years, prediction_time_horizon_in_years,
             news_analysis_agent_enabled, sec_filings_agent_enabled, technical_indicators_agent_enabled, result):     
    return result

#@weave.op()
#@agentops.record_function("run_crew")
def run_crew(model, company, historical_horizon_in_years, prediction_time_horizon_in_years,
                    news_analysis_agent_enabled, sec_filings_agent_enabled, technical_indicators_agent_enabled):

    st.session_state.messages.append({"role": "user", "content": company})
    st.chat_message("user").write(company)

    # Define tasks to run, and the agent to run it with

    tasksList = []
    agentsList = []

    if news_analysis_agent_enabled:
        agentsList.append(news_analysis_agent)
        analyze_news_task = Task(
            description=dedent(f"""
                Collect and summarize important news articles, press
                releases, and market analyses related to the company and
                its industry. Look for the most recent news possible, but also
                include any historical news that is very relevant to the future
                prospects of the company.

                Selected company by the customer: {company}
                Today's date: {date.today()}
                Historical time horizon for news: {historical_horizon_in_years} years
            """),
            expected_output=dedent(f"""
                Your output is a concise report that includes a
                summary of the latest news, any notable
                shifts in market sentiment, and impacts on
                the future of the company.

                Also include in the output the company's stock ticker symbol,
                whether you believe the company's stock price will increase
                in the next {prediction_time_horizon_in_years} years, and the confidence
                level of your prediction/belief."""),
            agent=news_analysis_agent
        )
        tasksList.append(analyze_news_task)

    if sec_filings_agent_enabled:
        agentsList.append(sec_filings_agent)
        analyze_sec_filings_task = Task(
            description=dedent(f"""
                Analyze the latest 10-Q and 10-K filings from EDGAR for
                the company in question.
                Focus on key sections like Management's Discussion and
                Analysis, financial statements, insider trading activity,
                and any disclosed risks.
                Extract relevant data and insights that could influence
                the company's future performance.

                Selected company by the customer: {company}
                Today's date: {date.today()}
            """),
            expected_output=dedent(f"""
                Your output is a concise report that
                highlights significant findings from these filings,
                including any red flags or positive indicators for your
                customer.

                Also include in the output the company's stock ticker symbol,
                whether you believe the company's stock price will increase
                in the next {prediction_time_horizon_in_years} years, and the confidence
                level of your prediction/belief."""),
            agent=sec_filings_agent
        )
        tasksList.append(analyze_sec_filings_task)

    if technical_indicators_agent_enabled:
        agentsList.append(technical_indicators_agent)
        analyze_technical_indicators_task = Task(
            description=dedent(f"""
                Analyze the latest technical indicators for
                the company in question.
                               
                Selected company by the customer: {company}
                Today's date: {date.today()}

                Please convert the company name to its stock ticker symbol to get 
                the technical indicator data.
            """),
            expected_output=dedent(f"""
                Your output is a concise report that includes a
                summary of the technical indicators, and impacts on
                the future of the company. This report should be in readable text,
                not a JSON structure.
                                   
                Include in the output the company's stock ticker symbol,
                whether you believe the company's stock price will increase
                in the next {prediction_time_horizon_in_years} years, and the confidence
                level of your prediction/belief.
                """),
            agent=technical_indicators_agent
        )
        tasksList.append(analyze_technical_indicators_task)

    #print(agentsList)
    #print(tasksList)

    if len(agentsList)==0:
        return "No agents found. Please choose at least one agent."
    
    # Set up the crew and process tasks hierarchically
    project_crew = Crew(
        tasks=tasksList,
        agents=agentsList,
        process=Process.sequential,
        #manager_llm=chosen_llm, # only required for hierarchical process
        manager_callbacks=[CustomHandler("Manager")]
    )

    report = project_crew.kickoff()
    return report


def icon(emoji: str):
    """Shows an emoji as a Notion-style page icon."""
    st.write(
        f'<span style="font-size: 78px; line-height: 1">{emoji}</span>',
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    weave.init('finbot-crewai-streamlit')

    st.set_page_config(page_title="Finbot", page_icon="📈", layout="wide")

    # Main Streamlit UI setup
    icon("📈 **Finbot**")

    st.subheader("A Company Analysis Tool for Curious Investors",
                 divider="violet", anchor=False)
    
    # Set up the Streamlit UI customization sidebar
    st.sidebar.title('Customizations')

    #TODO: issue with using gpt-3.5-turbo for some reason
    model = st.sidebar.selectbox(
       'Choose AI model to use',
       ['claude-3-5-sonnet-20240620', 'claude-3-haiku-20240307', 'llama3-8b-8192', 'mixtral-8x7b-32768', 'gemma-7b-it'],
       index=1, # default to claude-3-haiku-20240307
    )

    # Set up model (automatically called again when model is changed)
    if model!=None and model.startswith('gpt'):
        chosen_llm = ChatOpenAI(model=model, temperature=0.1) # gpt models
    elif model!=None and model.startswith('claude'):
        chosen_llm = ChatAnthropic(model=model, temperature=0.1) # claude models
    else:
        chosen_llm = ChatGroq(
            temperature=0.1, 
            groq_api_key = os.environ['GROQ_API_KEY'], 
            model_name=model)

    # else:
    #     # some of these HF models are too big to run as is: 'meta-llama/Meta-Llama-3-8B-Instruct', 'mistralai/Mixtral-8x22B-Instruct-v0.1', 'google/gemma-7b-it'
    #     chosen_llm = HuggingFaceHub(
    #         repo_id=model,
    #         huggingfacehub_api_token=os.environ['HF_TOKEN'],
    #         task="text-generation")

    historical_horizon_in_years = st.sidebar.number_input(
        'Historical time horizon (in years)',
        value=1.0, min_value=0.0, max_value=10.0, step=0.5, format="%.1f"
    )

    prediction_time_horizon_in_years = st.sidebar.number_input(
        'Prediction time horizon (in years)',
        value=0.5, min_value=0.0, max_value=10.0, step=0.5, format="%.1f"
    )

    # Define agents with their specific roles and goals

    st.sidebar.write("")
    st.sidebar.write("Choose CrewAI agent(s) to use:")

    news_analysis_agent_enabled = st.sidebar.checkbox(
        'News analysis agent',
        value=True
    )

    if news_analysis_agent_enabled:
        news_analysis_agent = get_news_analysis_agent(chosen_llm)

    sec_filings_agent_enabled = st.sidebar.checkbox(
        'SEC filings agent (alpha version)',
        value=False
    )

    if sec_filings_agent_enabled:
        sec_filings_agent = get_sec_filings_agent(chosen_llm)

    technical_indicators_agent_enabled = st.sidebar.checkbox(
        'Technical indicators agent',
        value=True
    )

    if technical_indicators_agent_enabled:
        technical_indicators_agent = get_technical_indicators_agent(chosen_llm)


    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.markdown(
    """
    *This is an [open-source demo app](https://github.com/xke/finbot-crewai-streamlit). Use the AI output at your own risk.*
    
    *The [S&P 500](https://en.wikipedia.org/wiki/S%26P_500) is a diversified investment option that includes 500 companies.*
    """,
        unsafe_allow_html=True
    )
  
    
    # Initialize the message log in session state if not already present
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "##### What company do you want to analyze?"}]


    # Display existing messages
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Handle user input

    if company := st.chat_input():
        #agentops.init(tags=["finbot-crewai-streamlit", company, model])

        #agentops.record(agentops.ActionEvent([company, model]))

        log_run("start", model, company, historical_horizon_in_years, prediction_time_horizon_in_years,
                news_analysis_agent_enabled, sec_filings_agent_enabled, technical_indicators_agent_enabled, None)

        report = run_crew(model, company, historical_horizon_in_years, prediction_time_horizon_in_years,
                        news_analysis_agent_enabled, sec_filings_agent_enabled, technical_indicators_agent_enabled)

        # Display the final result
        result = f"##### Manager's Final Report: \n\n {report}"
        st.session_state.messages.append({"role": "assistant", "content": result})
        st.chat_message("assistant").write(result)

        log_run("finish", model, company, historical_horizon_in_years, prediction_time_horizon_in_years,
                news_analysis_agent_enabled, sec_filings_agent_enabled, technical_indicators_agent_enabled, result)

        #if result: 
        #    agentops.end_session('Success')
        #else:
        #    agentops.end_session('Failure')