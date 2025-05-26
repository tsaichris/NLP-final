# %pip install -qU langchain-openai
from langchain_openai import ChatOpenAI
from langchain_openai import AzureChatOpenAI

# set logprobs=True when initializing the ChatOpenAI model
"""
logprobs
Whether to return log probabilities of the output tokens or not. 
If true, returns the log probabilities of each output token returned in the content of message
"""
# https://python.langchain.com/api_reference/openai/chat_models/langchain_openai.chat_models.azure.AzureChatOpenAI.html
llm = AzureChatOpenAI(
    azure_deployment="gpt-35-turbo",
    api_version="2024-05-01-preview",
    logprobs = True # set logprob 
    # other params...
)

# https://python.langchain.com/api_reference/openai/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html
llm = ChatOpenAI(model="gpt-4o-mini", 
                 logprobs=True,
                 # other params...
                 )
# or
logprobs_llm = llm.bind(logprobs=True)

msg = llm.invoke(("human", "how are you today"))

# content of message metadata
print(msg.response_metadata["logprobs"]["content"][:5])
""" Result
{
    "content": [
        {
            "token": "J",
            "bytes": [74],
            "logprob": -4.9617593e-06,
            "top_logprobs": [],
        },
        {
            "token": "'adore",
            "bytes": [39, 97, 100, 111, 114, 101],
            "logprob": -0.25202933,
            "top_logprobs": [],
        },
        {
            "token": " la",
            "bytes": [32, 108, 97],
            "logprob": -0.20141791,
            "top_logprobs": [],
        },
        {
            "token": " programmation",
            "bytes": [
                32,
                112,
                114,
                111,
                103,
                114,
                97,
                109,
                109,
                97,
                116,
                105,
                111,
                110,
            ],
            "logprob": -1.9361265e-07,
            "top_logprobs": [],
        },
        {
            "token": ".",
            "bytes": [46],
            "logprob": -1.2233183e-05,
            "top_logprobs": [],
        },
    ]
}


"""
# ref
# https://platform.openai.com/docs/api-reference/chat/create