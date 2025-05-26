import getpass
import os

# API key
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass()


from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini").bind(logprobs=True)

msg = llm.invoke(("human", "how are you today"))

msg.response_metadata["logprobs"]["content"][:5]    

"""Response:

[{'token': 'I', 'bytes': [73], 'logprob': -0.26341408, 'top_logprobs': []},
 {'token': "'m",
  'bytes': [39, 109],
  'logprob': -0.48584133,
  'top_logprobs': []},
 {'token': ' just',
  'bytes': [32, 106, 117, 115, 116],
  'logprob': -0.23484154,
  'top_logprobs': []},
 {'token': ' a',
  'bytes': [32, 97],
  'logprob': -0.0018291725,
  'top_logprobs': []},
 {'token': ' computer',
  'bytes': [32, 99, 111, 109, 112, 117, 116, 101, 114],
  'logprob': -0.052299336,
  'top_logprobs': []}]

"""

ct = 0
full = None
for chunk in llm.stream(("human", "how are you today")):
    if ct < 5:
        full = chunk if full is None else full + chunk
        if "logprobs" in full.response_metadata:
            print(full.response_metadata["logprobs"]["content"])
    else:
        break
    ct += 1

"""Result:
[]
[{'token': 'I', 'bytes': [73], 'logprob': -0.26593843, 'top_logprobs': []}]
[{'token': 'I', 'bytes': [73], 'logprob': -0.26593843, 'top_logprobs': []}, {'token': "'m", 'bytes': [39, 109], 'logprob': -0.3238896, 'top_logprobs': []}]
[{'token': 'I', 'bytes': [73], 'logprob': -0.26593843, 'top_logprobs': []}, {'token': "'m", 'bytes': [39, 109], 'logprob': -0.3238896, 'top_logprobs': []}, {'token': ' just', 'bytes': [32, 106, 117, 115, 116], 'logprob': -0.23778509, 'top_logprobs': []}]
[{'token': 'I', 'bytes': [73], 'logprob': -0.26593843, 'top_logprobs': []}, {'token': "'m", 'bytes': [39, 109], 'logprob': -0.3238896, 'top_logprobs': []}, {'token': ' just', 'bytes': [32, 106, 117, 115, 116], 'logprob': -0.23778509, 'top_logprobs': []}, {'token': ' a', 'bytes': [32, 97], 'logprob': -0.0022134194, 'top_logprobs': []}]

"""