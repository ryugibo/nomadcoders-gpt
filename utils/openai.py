import openai


def check_openai_api_key(api_key):
    openai.api_key = api_key
    try:
        openai.Model.list()
    except openai.error.AuthenticationError as e:
        return False
    else:
        return True
