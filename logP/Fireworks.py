# %pip install -qU langchain-fireworks

from langchain_fireworks import ChatFireworks


# allow Top N candidate for tokens by top_logprobs = True
# model initialization, dont' need to set logprob
# https://python.langchain.com/api_reference/fireworks/chat_models/langchain_fireworks.chat_models.ChatFireworks.html
llm = ChatFireworks(model="accounts/fireworks/models/llama-v3p1-70b-instruct"
                    ,   temperature=0,
                    max_tokens=None,
                    timeout=None,
                    max_retries=2,
                    # other params...
                                    
                    )
msg = llm.invoke(("human", "how are you today"))

"""
The logprobs parameter determines how many token probabilities are returned. If set to N, it will return log (base e) probabilities for N+1 tokens: 
the chosen token plus the N most likely alternative tokens.

The log probabilities will be returned in a LogProbs object for each choice.

tokens contains each token of the chosen result.
token_ids contains the integer IDs of each token of the chosen result.
token_logprobs contains the logprobs of each chosen token.
top_logprobs will be a list whose length is the number of tokens of the output. Each element is a dictionary of size logprobs, 
from the most likely tokens at the given position to their respective log probabilities.

"""
print(msg.response_metadata["logprobs"])

# ref
# https://docs.fireworks.ai/guides/querying-text-models#logprobs