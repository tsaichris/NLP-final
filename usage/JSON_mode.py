from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-4o").with_structured_output(method="json_mode")
ai_msg = model.invoke("Return a JSON object with key 'random_ints' and a value of 10 random ints in [0-99]")
ai_msg
{'random_ints': [45, 67, 12, 34, 89, 23, 78, 56, 90, 11]}