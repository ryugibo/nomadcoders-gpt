import streamlit as st
from utils.openai import check_openai_api_key
from typing import Type
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.tools import WikipediaQueryRun, BaseTool, DuckDuckGoSearchResults
from langchain.utilities import WikipediaAPIWrapper
from langchain.schema import SystemMessage
from pydantic import BaseModel, Field


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.write(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], False)


class SearchToolArgsSchema(BaseModel):
    keyword: str = Field(description="The keyword you will search for.")


class CrawlToolArgsSchema(BaseModel):
    url: str = Field(description="The url you will crawl for.")


class WriteTxtToolArgsSchema(BaseModel):
    information: str = Field(description="The information wroten into txt file.")


class WikipediaSearchTool(BaseTool):
    name = "WikipediaSearch"
    description = """
    Use this tool to find information about keyword in Wikipedia.
    It takes a query as an argument.
    """
    args_schema: Type[SearchToolArgsSchema] = SearchToolArgsSchema

    def _run(self, keyword):
        wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        return wikipedia.run(keyword)


class DuckDuckGoSearchTool(BaseTool):
    name = "DuckDuckGoSearch"
    description = """
    Use this tool to find sites about keyword in DuckDuckGo.
    It takes a query as an argument.
    """
    args_schema: Type[SearchToolArgsSchema] = SearchToolArgsSchema

    def _run(self, keyword):
        ddg = DuckDuckGoSearchResults()
        return ddg.run(keyword)


class WebsiteCrawlTool(BaseTool):
    name = "WebsiteCrawl"
    description = """
    Use this tool to find information in sites.
    It takes a address of site as an argument.
    """
    args_schema: Type[CrawlToolArgsSchema] = CrawlToolArgsSchema

    def _run(self, url):
        return url


st.set_page_config(page_title="SearchGPT", page_icon="🌥️")
st.title("SearchGPT")

with st.sidebar:
    st.link_button(
        label="https://github.com/ryugibo/nomadcoders-gpt/blob/main/pages/03_SearchGPT.py",
        url="https://github.com/ryugibo/nomadcoders-gpt/blob/main/pages/03_SearchGPT.py",
    )

    api_key = st.text_input("OPENAI API KEY")

    is_valid_api_key = check_openai_api_key(api_key) if api_key else False

message = st.chat_input("Enter a question", disabled=not is_valid_api_key)
if is_valid_api_key:
    if "langchain" not in st.session_state:
        st.session_state["agent"] = initialize_agent(
            llm=ChatOpenAI(
                api_key=api_key,
                temperature=1e-1,
            ),
            verbose=True,
            agent=AgentType.OPENAI_FUNCTIONS,
            handle_parsing_errors=True,
            tools=[
                WikipediaSearchTool(),
                DuckDuckGoSearchTool(),
                WebsiteCrawlTool(),
            ],
            agent_kwargs={
                "system_message": SystemMessage(
                    content="""
                You are a information collector.
                
                You find a information about keyword.

                You Wikipedia or DuckDuckGo.

                if you used DuckDuckGo, crawl informations from each websites.
                """
                )
            },
        )

    send_message("I'm ready! Ask away", "ai", False)
    paint_history()
    if message:
        send_message(message, "human")
        response = st.session_state["agent"].invoke(message)
        send_message(response["output"], "ai")
else:
    st.session_state["messages"] = []
    st.warning("Input open ai api key in sidebar")
