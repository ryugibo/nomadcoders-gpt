import streamlit as st
from utils.openai import check_openai_api_key
from typing import Type
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.tools import WikipediaQueryRun, BaseTool, DuckDuckGoSearchResults
from langchain.utilities import WikipediaAPIWrapper
from langchain.schema import SystemMessage
from pydantic import BaseModel, Field


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

if check_openai_api_key(api_key):
    llm = ChatOpenAI(
        api_key=api_key,
        temperature=1e-1,
    )
    agent = initialize_agent(
        llm=llm,
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

    agent.invoke("Research about the XZ backdoor")
else:
    st.warning("Input open ai api key in sidebar")
