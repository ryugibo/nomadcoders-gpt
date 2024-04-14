import streamlit as st
from utils.openai import check_openai_api_key
import openai as client
from langchain.tools import WikipediaQueryRun, DuckDuckGoSearchResults
from langchain.utilities import WikipediaAPIWrapper
from langchain.document_loaders import WebBaseLoader
import time
import json
import requests
from bs4 import BeautifulSoup
import re


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def print_message(message, role, save=True):
    with st.chat_message(role):
        st.write(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        print_message(message["message"], message["role"], False)


def research_wikipedia(inputs):
    keyword = inputs["keyword"]
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    return wikipedia.run(keyword)


def research_duckduckgo(inputs):
    keyword = inputs["keyword"]
    ddg = DuckDuckGoSearchResults()
    results = ddg.run(keyword)
    urls = re.findall(r"link: (https?://\S+)", results)
    urls = [url[:-1] if url.endswith("]") else url for url in urls]
    return str(urls)


def crawl_site(inputs):
    url = inputs["url"]
    loader = WebBaseLoader([url])
    return " ".join(loader.load()[0].page_content.split())


def get_run(run_id, thread_id):
    return client.beta.threads.runs.retrieve(run_id=run_id, thread_id=thread_id)


def get_messages(thread_id):
    return list(
        client.beta.threads.messages.list(
            thread_id=thread_id,
        )
    )


def get_tool_outputs(run_id, thread_id):
    run = get_run(run_id, thread_id)
    outputs = []
    for action in run.required_action.submit_tool_outputs.tool_calls:
        action_id = action.id
        function = action.function
        print(f"Calling function: {function.name} with arg {function.arguments}")
        outputs.append(
            {
                "output": functions_map[function.name](json.loads(function.arguments)),
                "tool_call_id": action_id,
            }
        )
    return outputs


def submit_tool_outputs(run_id, thread_id):
    outputs = get_tool_outputs(run_id, thread_id)
    return client.beta.threads.runs.submit_tool_outputs(
        run_id=run_id,
        thread_id=thread_id,
        tool_outputs=outputs,
    )


functions_map = {
    "research_wikipedia": research_wikipedia,
    "research_duckduckgo": research_duckduckgo,
    "crawl_site": crawl_site,
}

functions = [
    {
        "type": "function",
        "function": {
            "name": "research_wikipedia",
            "description": "Research a information about keyword in Wikipedia.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "The keyword you will search for.",
                    }
                },
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "research_duckduckgo",
            "description": "List up sites about keyword in DuckDuckGo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "The keyword you will search for.",
                    }
                },
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "crawl_site",
            "description": "Crawling summary a site.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The url you will crawl for.",
                    },
                },
                "required": ["url"],
            },
        },
    },
]


st.set_page_config(page_title="ResearchGPT", page_icon="🌥️")
st.title("ResearchGPT")

with st.sidebar:
    st.link_button(
        label="https://github.com/ryugibo/nomadcoders-gpt/blob/main/pages/03_ResearchGPT.py",
        url="https://github.com/ryugibo/nomadcoders-gpt/blob/main/pages/03_ResearchGPT.py",
    )

    api_key = st.text_input("OPENAI API KEY")

    is_valid_api_key = check_openai_api_key(api_key) if api_key else False

message = st.chat_input("Enter a keyword for research", disabled=not is_valid_api_key)
if is_valid_api_key:
    if "assistant" not in st.session_state:
        client.api_key = api_key
        st.session_state["assistant"] = client.beta.assistants.create(
            name="Research Assistant",
            instructions="""You are a information collector.
            
            You find a information about keyword.

            You Wikipedia or DuckDuckGo.

            if you used DuckDuckGo, crawl informations from each websites.""",
            model="gpt-3.5-turbo",
            tools=functions,
        )
    if "thread" not in st.session_state:
        st.session_state["thread"] = client.beta.threads.create()

    print_message("I'm ready! Ask away", "assistant", False)
    paint_history()
    if message:
        print_message(message, "human")
        client.beta.threads.messages.create(
            thread_id=st.session_state["thread"].id,
            role="user",
            content=message,
        )

        run = client.beta.threads.runs.create(
            thread_id=st.session_state["thread"].id,
            assistant_id=st.session_state["assistant"].id,
        )
        with st.spinner("Wait assistant..."):
            while run.status != "completed":
                run = get_run(run.id, st.session_state["thread"].id)
                if run.status == "requires_action":
                    submit_tool_outputs(run.id, st.session_state["thread"].id)
                time.sleep(0.2)
        output = get_messages(st.session_state["thread"].id)[0]
        print_message("\n".join([c.text.value for c in output.content]), "assistant")
else:
    st.session_state["messages"] = []
    st.warning("Input open ai api key in sidebar")
