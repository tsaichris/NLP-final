# %pip install -qU langchain-google-vertexai

from langchain_google_vertexai import ChatVertexAI

# set logprobs=True 
# only avaliable for gemini-1.5-flash, gemini-2.0-flash-lite-001, gemini-2.0-flash-001
# TOP 1~5 for each token
# limit, 1 per day
# https://python.langchain.com/api_reference/google_vertexai/chat_models/langchain_google_vertexai.chat_models.ChatVertexAI.html
llm = ChatVertexAI(model="gemini-1.5-flash", logprobs=True)
msg = llm.invoke(("human", "how are you today"))

"""
logprobs
Optional: int

Returns the log probabilities of the top candidate tokens at each generation step. 
The model's chosen token might not be the same as the top candidate token at each step. Specify the number of candidates to return by using an integer value in the range of 1-5.

You must enable responseLogprobs to use this parameter. The daily limit for requests using logprobs is 1.

This is a preview feature.
"""
print(msg.response_metadata["logprobs_result"])

"""result:
[
    {'token': 'J', 'logprob': -1.549651415189146e-06, 'top_logprobs': []},
    {'token': "'", 'logprob': -1.549651415189146e-06, 'top_logprobs': []},
    {'token': 'adore', 'logprob': 0.0, 'top_logprobs': []},
    {'token': ' programmer', 'logprob': -1.1922384146600962e-07, 'top_logprobs': []},
    {'token': '.', 'logprob': -4.827636439586058e-05, 'top_logprobs': []},
    {'token': ' ', 'logprob': -0.018011733889579773, 'top_logprobs': []},
    {'token': '\n', 'logprob': -0.0008687592926435173, 'top_logprobs': []}
]


"""


# ref
# https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference#responseLogprobs
# https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/content-generation-parameters
