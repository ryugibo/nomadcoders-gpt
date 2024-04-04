from enum import Enum
import streamlit as st
import time
import openai


def check_openai_api_key(api_key):
    openai.api_key = api_key
    try:
        openai.Model.list()
    except openai.error.AuthenticationError as e:
        return False
    else:
        return True


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
st.chat_input("Enter a question", disabled=not settings_ok)

if settings_ok:
    st.chat_message("human").write("hello")
    st.chat_message("ai").write("good day")
else:
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
