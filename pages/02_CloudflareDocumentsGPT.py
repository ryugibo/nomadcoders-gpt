import streamlit as st
from utils.openai import check_openai_api_key
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import SitemapLoader
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings


def parse_page(soup):
    text = soup.find("main", "DocsBody").get_text()
    return text


def load_website(api_key):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    loader = SitemapLoader(
        "https://developers.cloudflare.com/sitemap.xml",
        parsing_function=parse_page,
        filter_urls=[
            r"^(.*\/ai-gateway\/).*",
            r"^(.*\/vectorize\/).*",
            r"^(.*\/workers-ai\/).*",
        ],
    )
    loader.requests_per_second = 5
    docs = loader.load_and_split(text_splitter=splitter)
    vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings(api_key=api_key))
    return vectorstore.as_retriever()


st.set_page_config(page_title="CloudflareDocumentsGPT", page_icon="🌥️")
st.title("CloudflareDocumentsGPT")

with st.sidebar:
    st.link_button(
        label="https://github.com/ryugibo/nomadcoders-gpt/blob/main/pages/02_CloudflareDocumentsGPT.py",
        url="https://github.com/ryugibo/nomadcoders-gpt/blob/main/pages/02_CloudflareDocumentsGPT.py",
    )

    api_key = st.text_input("OPENAI API KEY")

if check_openai_api_key(api_key):
    retriever = load_website(api_key)
else:
    st.warning("Input open ai api key in sidebar")
