# pip install -qU "langchain[modelName]"
"""
modelName_list
openai          => gpt-4o-mini, azure
anthropic       => claude-3-5-sonnet-latest, 
google-genai    => gemini-2.0-flash
google-vertexai => gemini-2.0-flash-001
aws             => anthropic.claude-3-5-sonnet-20240620-v1:0
groq            => llama3-8b-8192
cohere => command-r-plus
langchain-nvidia-ai-endpoints => meta/llama3-70b-instruct
fireworks => accounts/fireworks/models/llama-v3p1-70b-instruct
mistralai = > mistral-large-latest
together => mistralai/Mixtral-8x7B-Instruct-v0.1
langchain-ibm => ibm/granite-34b-code-instruct
databricks-langchain => databricks-meta-llama-3-1-70b-instruct
langchain-xai => grok-2
langchain-perplexity => llama-3.1-sonar-small-128k-online
"""
# Extra needed
# pip install -qU "langchain-nvidia-ai-endpoints" => meta/llama3-70b-instruct



# Ensure API key is set

# import model 
from langchain.chat_models import init_chat_model
model = init_chat_model("gpt-4o-mini", model_provider="openai")

"""
modelList:
- OpenAI

model = init_chat_model("gpt-4o-mini", model_provider="openai")

from langchain_openai import AzureChatOpenAI
model = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)


- anthropic
model = init_chat_model("claude-3-5-sonnet-latest", model_provider="anthropic")


- google-genai
model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
https://ai.google.dev/gemini-api/docs/models?hl=zh-tw


- google-vertexai (no api key)
model = init_chat_model("gemini-2.0-flash-001", model_provider="google_vertexai")


- aws (no api key)
model = init_chat_model("anthropic.claude-3-5-sonnet-20240620-v1:0", model_provider="bedrock_converse")


- groq
model = init_chat_model("llama3-8b-8192", model_provider="groq")


- cohere
model = init_chat_model("command-r-plus", model_provider="cohere")


- langchain-nvidia-ai-endpoints (NVIDIA)
model = init_chat_model("meta/llama3-70b-instruct", model_provider="nvidia")

from langchain_nvidia_ai_endpoints import ChatNVIDIA
model = ChatNVIDIA(model="meta/llama2-70b")
llm = ChatNVIDIA(model="mistralai/mixtral-8x7b-instruct-v0.1")
print(ChatNVIDIA.get_available_models())

- fireworks
model = init_chat_model("accounts/fireworks/models/llama-v3p1-70b-instruct", model_provider="fireworks")


- mistralai
model = init_chat_model("mistral-large-latest", model_provider="mistralai")
#https://docs.mistral.ai/getting-started/models/models_overview/


- together
from langchain.chat_models import init_chat_model
model = init_chat_model("mistralai/Mixtral-8x7B-Instruct-v0.1", model_provider="together")


- langchain-ibm
from langchain_ibm import ChatWatsonx
model = ChatWatsonx(
    model_id="ibm/granite-34b-code-instruct", 
    url="https://us-south.ml.cloud.ibm.com", 
    project_id="<WATSONX PROJECT_ID>"
)


- databricks-langchain
os.environ["DATABRICKS_HOST"] = "https://example.staging.cloud.databricks.com/serving-endpoints"
model = ChatDatabricks(endpoint="databricks-meta-llama-3-1-70b-instruct")


- langchain-xai
model = init_chat_model("grok-2", model_provider="xai")


- langchain-perplexity
model = init_chat_model("llama-3.1-sonar-small-128k-online", model_provider="perplexity")
"""
from langchain_nvidia_ai_endpoints import ChatNVIDIA





# usage example
from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage("Translate the following from English into Italian"),
    HumanMessage("hi!"),
]

# see "Message" in reference
model.invoke(messages)

"""
model.invoke("Hello")

model.invoke([{"role": "user", "content": "Hello"}])

model.invoke([HumanMessage("Hello")])
"""
