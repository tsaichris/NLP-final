# https://python.langchain.com/docs/concepts/prompt_templates/
from langchain_core.prompts import ChatPromptTemplate

# language: The language to translate text into
# text: The text to translate

system_template = "Translate the following from English into {language}"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

prompt = prompt_template.invoke({"language": "Italian", "text": "hi!"})
print(prompt)
# ChatPromptValue(messages=
# [SystemMessage(content='Translate the following from English into Italian', additional_kwargs={}, response_metadata={}), 
# HumanMessage(content='hi!', additional_kwargs={}, response_metadata={})])

# access the message
prompt.to_messages()
#[SystemMessage(content='Translate the following from English into Italian', additional_kwargs={}, response_metadata={}),
# HumanMessage(content='hi!', additional_kwargs={}, response_metadata={})]

# invode to model
response = model.invoke(prompt)
print(response.content) # Ciao!
