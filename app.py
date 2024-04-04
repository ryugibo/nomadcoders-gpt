import streamlit as st
import time
import openai
import os
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferMemory


class ChatCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.message = ""

    def on_llm_start(self, *args, **kawargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kawargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


def check_openai_api_key(api_key):
    openai.api_key = api_key
    try:
        openai.Model.list()
    except openai.error.AuthenticationError as e:
        return False
    else:
        return True


def embed_file(file, api_key):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(file_content)

    cache_path = f"./.cache/embeddings/{file.name}"
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    cache_dir = LocalFileStore(cache_path)

    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )

    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)

    embeddings = OpenAIEmbeddings(api_key=api_key)
    cached_embedding = CacheBackedEmbeddings.from_bytes_store(
        embeddings,
        cache_dir,
    )

    vectorstore = FAISS.from_documents(docs, cached_embedding)

    return vectorstore.as_retriever()


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


def format_docs(docs):
    return "\n\n".join([document.page_content for document in docs])


st.set_page_config(
    page_icon="🖨️",
    page_title="",
)

st.title("Hello")

with st.sidebar:
    api_key = st.text_input("OPENAI API KEY")

    is_valid_api_key = check_openai_api_key(api_key) if api_key else False

    file = st.file_uploader("file")

    st.link_button(
        label="https://github.com/ryugibo/nomadcoders-gpt",
        url="https://github.com/ryugibo/nomadcoders-gpt",
    )

settings_ok = is_valid_api_key and file

if settings_ok:
    memory = ConversationBufferMemory(return_messages=True, memory_key="history")
    llm = ChatOpenAI(
        temperature=1e-1,
        streaming=True,
        callbacks=[
            ChatCallbackHandler(),
        ],
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Answer questions using only the following context. If you don't know the answer just say you don't know, don't make it up:\n\n{context}",
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )
    retriever = embed_file(file, api_key)
    send_message("I'm ready! Ask away", "ai", False)
    paint_history()
    message = st.chat_input("Enter a question", disabled=not settings_ok)
    if message:
        send_message(message, "human")
        chain = (
            {
                "context": retriever,
                "question": RunnablePassthrough(),
                "history": RunnableLambda(
                    lambda _: memory.load_memory_variables({})["history"]
                ),
            }
            | prompt
            | llm
        )
        with st.chat_message("ai"):
            response = chain.invoke(message)
else:
    st.session_state["messages"] = []
    st.warning("Please, complete settings on sidebar.", icon="⚠️")

    with st.status("Wait for complete settings..", expanded=True):
        if is_valid_api_key:
            st.success("OPENAI API KEY OK", icon="🔑")
        elif api_key:
            st.warning(
                "OPEN API KEY is invalid. check https://platform.openai.com/api-keys",
                icon="⚠️",
            )
        else:
            st.warning("Please, enter a OPEN API KEY", icon="⚠️")

        if file:
            st.success("OK, File uploaded.", icon="📁")
        else:
            st.warning("Please, upload file.", icon="⚠️")

        while not is_valid_api_key or not file:
            time.sleep(0.2)
