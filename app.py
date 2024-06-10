import streamlit as st
import weave

from crewai import Crew, Process, Agent, Task
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

from agents.news_analysis_agent import get_news_analysis_agent
from agents.sec_filings_agent import get_sec_filings_agent
#from agents.technical_indicators_agent import get_technical_indicators_agent

#from agents.news_analysis_agent import SearchTools
#from agents.sec_filings_agent import SECTools

from textwrap import dedent
import os
from handler import CustomHandler

from dotenv import load_dotenv
load_dotenv()

from datetime import date


#__import__('pysqlite3')
#import sys
#sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')



weave.init('finbot-crewai-streamlit')

@weave.op()
def process_user_input(company, chosen_llm, historical_horizon_in_years, prediction_time_horizon_in_years):

    st.session_state.messages.append({"role": "user", "content": company})
    st.chat_message("user").write(company)


    # Define tasks to run, and the agent to run it with

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

    # analyze_technical_indicators_task = Task(
    #     description=dedent(f"""
    #         [update prompt]

    #         Selected company by the customer: {company}
    #         Today's date: {date.today()}
    #     """),
    #     expected_output=dedent(f"""
    #         Your output is a concise report that
    #         highlights significant findings from these filings,
    #         including any red flags or positive indicators for your
    #         customer.

    #         Also include in the output the company's stock ticker symbol,
    #         whether you believe the company's stock price will increase
    #         in the next {prediction_time_horizon_in_years} years, and the confidence
    #         level of your prediction/belief."""),
    #     agent=technical_indicators_agent
    #  )

    # Set up the crew and process tasks hierarchically

    project_crew = Crew(
        tasks=[analyze_news_task, analyze_sec_filings_task],
        agents=[news_analysis_agent, sec_filings_agent],
        process=Process.hierarchical,
        manager_llm=chosen_llm,
        manager_callbacks=[CustomHandler("Manager")]
    )

    # project_crew = Crew(
    #     tasks=[analyze_news_task, analyze_sec_filings_task, analyze_technical_indicators_task],
    #     agents=[news_analysis_agent, sec_filings_agent, technical_indicators_agent],
    #     process=Process.hierarchical,
    #     manager_llm=chosen_llm,
    #     manager_callbacks=[CustomHandler("Manager")]
    # )

    report = project_crew.kickoff()
    return report


# Set up the Streamlit UI customization sidebar
st.sidebar.title('Customizations')
model = st.sidebar.selectbox(
    'Choose AI model to use',
    ['gpt-3.5-turbo', 'gpt-4o', 'llama3-8b-8192', 'mixtral-8x7b-32768', 'gemma-7b-it'],
    index=4, # default to gemma-7b-it
)

# Set up model (automatically called again when model is changed)
if model!=None and model.startswith('gpt'):
    chosen_llm = ChatOpenAI(model=model, temperature=0.1) # openai
else:
    chosen_llm = ChatGroq(
            temperature=0, 
            groq_api_key = os.environ['GROQ_API_KEY'], 
            model_name=model
        )
    
historical_horizon_in_years = st.sidebar.number_input(
    'Historical time horizon (in years)',
    value=1.0, min_value=0.0, max_value=10.0, step=0.5, format="%.1f"
)

prediction_time_horizon_in_years = st.sidebar.number_input(
    'Prediction time horizon (in years)',
    value=0.5, min_value=0.0, max_value=10.0, step=0.5, format="%.1f"
)

# Main Streamlit UI setup
st.title("Finbot: A Company Analysis Tool")

# Define agents with their specific roles and goals
news_analysis_agent = get_news_analysis_agent(chosen_llm)
sec_filings_agent = get_sec_filings_agent(chosen_llm)
#technical_indicators_agent = get_technical_indicators_agent(chosen_llm)

# Initialize the message log in session state if not already present
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "##### What company do you want us to analyze?"}]


# Display existing messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Handle user input
if company := st.chat_input():

    report = process_user_input(company, chosen_llm, historical_horizon_in_years, prediction_time_horizon_in_years)

    # Display the final result
    result = f"##### Manager's Final Report: \n\n {report}"
    st.session_state.messages.append({"role": "assistant", "content": result})
    st.chat_message("assistant").write(result)

