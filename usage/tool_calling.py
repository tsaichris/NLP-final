#https://python.langchain.com/docs/concepts/tool_calling/

from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-4o", temperature=0)
# Bind responseformatter schema as a tool to the model
model_with_tools = model.bind_tools([ResponseFormatter])
# Invoke the model
ai_msg = model_with_tools.invoke("What is the powerhouse of the cell?")

# Get the tool call arguments
ai_msg.tool_calls[0]["args"]
{'answer': "The powerhouse of the cell is the mitochondrion. Mitochondria are organelles that generate most of the cell's supply of adenosine triphosphate (ATP), which is used as a source of chemical energy.",
 'followup_question': 'What is the function of ATP in the cell?'}
# Parse the dictionary into a pydantic object
pydantic_object = ResponseFormatter.model_validate(ai_msg.tool_calls[0]["args"])