# %pip install -qU langchain-together
from langchain_together import ChatTogether
import math

# https://python.langchain.com/api_reference/together/chat_models/langchain_together.chat_models.ChatTogether.html
llm = ChatTogether(model="mistralai/Mixtral-8x7B-Instruct-v0.1", logprobs=True)

msg = llm.invoke(("human", "how are you today"))

print(msg.response_metadata["logprobs"])

# convert
def get_probability(logprob: float) -> float:
    return round(math.exp(logprob) * 100, 2)

print(get_probability(-0.39648438))
"""Response
{
  "id": "nrFCEVD-2j9zxn-934d8c409a0f43fd",
  "object": "chat.completion",
  "created": 1745413268,
  "model": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
  "choices": [
    {
      "index": 0,
      "logprobs": {
        "tokens": [
          "New",
          " York",
          " City",
          " is",
          " a",
          " vibrant",
          " and",
          " diverse",
          " destination",
          " with"
        ],
        "token_logprobs": [
          -0.39648438, -2.026558e-6, -0.3515625, -0.609375, -0.023803711,
          -0.53125, -0.03149414, -0.43359375, -0.38085938, -0.74609375
        ],
        "token_ids": [3648, 4356, 4409, 374, 264, 34076, 323, 17226, 9284, 449],
        "top_logprobs": [
          { "New": -0.39648438 },
          { " York": -2.026558e-6 },
          { " City": -0.3515625 },
          { " is": -0.609375 },
          { " a": -0.023803711 },
          { " vibrant": -0.53125 },
          { " and": -0.03149414 },
          { " diverse": -0.43359375 },
          { " destination": -0.38085938 },
          { " with": -0.74609375 }
        ]
      },
      "seed": 15158565520978651000,
      "finish_reason": "length",
      "message": {
        "role": "assistant",
        "content": "New York City is a vibrant and diverse destination with",
        "tool_calls": []
      }
    }
  ],
  "prompt": [],
  "usage": {
    "prompt_tokens": 48,
    "completion_tokens": 10,
    "total_tokens": 58,
    "cached_tokens": 0
  }
}

"""

# ref
# https://docs.together.ai/docs/logprobs