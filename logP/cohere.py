############ Not in LangChain ############

import cohere
co = cohere.ClientV2()
response = co.chat(
    model="command-a-03-2025",
    messages=[{"role": "user", "content": "hello world!"}],
    logprobs = True, # set this 
)
print(response)


""" response:

{
  "id": "c14c80c3-18eb-4519-9460-6c92edd8cfb4",
  "finish_reason": "COMPLETE",
  "message": {
    "role": "assistant",
    "content": [
      {
        "type": "text",
        "text": "LLMs stand for Large Language Models, which are a type of neural network model specialized in processing and generating human language. They are designed to understand and respond to natural language input and have become increasingly popular and valuable in recent years.\n\nLLMs are trained on vast amounts of text data, enabling them to learn patterns, grammar, and semantic meanings present in the language. These models can then be used for various natural language processing tasks, such as text generation, summarization, question answering, machine translation, sentiment analysis, and even some aspects of natural language understanding.\n\nSome well-known examples of LLMs include:\n\n1. GPT-3 (Generative Pre-trained Transformer 3) — An open-source LLM developed by OpenAI, capable of generating human-like text and performing various language tasks.\n\n2. BERT (Bidirectional Encoder Representations from Transformers) — A Google-developed LLM that is particularly good at understanding contextual relationships in text, and is widely used for natural language understanding tasks like sentiment analysis and named entity recognition.\n\n3. T5 (Text-to-Text Transfer Transformer) — Also from Google, T5 is a flexible LLM that frames all language tasks as text-to-text problems, where the model learns to generate output text based on input text prompts.\n\n4. RoBERTa (Robustly Optimized BERT Approach) — A variant of BERT that uses additional training techniques to improve performance.\n\n5. DeBERTa (Decoding-enhanced BERT with disentangled attention) — Another variant of BERT that introduces a new attention mechanism.\n\nLLMs have become increasingly powerful and larger in scale, improving the accuracy and sophistication of language tasks. They are also being used as a foundation for developing various applications, including chatbots, content recommendation systems, language translation services, and more. \n\nThe future of LLMs holds the potential for even more sophisticated language technologies, with ongoing research and development focused on enhancing their capabilities, improving efficiency, and exploring their applications in various domains."
      }
    ]
  },
  "usage": {
    "billed_units": {
      "input_tokens": 5,
      "output_tokens": 418
    },
    "tokens": {
      "input_tokens": 71,
      "output_tokens": 418
    }
  }
}

"""

# ref
#https://docs.cohere.com/reference/chat-stream