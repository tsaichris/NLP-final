# %pip install -qU langchain-groq

from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    logprobs = True, # set 
    # other params...
)

messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
ai_msg = llm.invoke(messages)
""" Result

AIMessage(content='The English sentence "I love programming" can
be translated to French as "J'aime programmer". The word
"programming" is translated as "programmer" in French.',
response_metadata={'token_usage': {'completion_tokens': 38,
'prompt_tokens': 28, 'total_tokens': 66, 'completion_time':
0.057975474, 'prompt_time': 0.005366091, 'queue_time': None,
'total_time': 0.063341565}, 'model_name': 'llama-3.1-8b-instant',
'system_fingerprint': 'fp_c5f20b5bb1', 'finish_reason': 'stop',
'logprobs': None}, id='run-ecc71d70-e10c-4b69-8b8c-b8027d95d4b8-0')

"""

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that translates {input_language} to {output_language}.",
        ),
        ("human", "{input}"),
    ]
)

chain = prompt | llm
chain.invoke(
    {
        "input_language": "English",
        "output_language": "German",
        "input": "I love programming.",
    }
)