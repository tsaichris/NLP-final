import getpass
import os

try:
    # load environment variables from .env file (requires `python-dotenv`)
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Langsmith

# Model API
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = ""

if "ANTHROPIC_API_KEY" not in os.environ:
    os.environ["ANTHROPIC_API_KEY"] = ""

if "AZURE_OPENAI_API_KEY" not in os.environ:
    os.environ["AZURE_OPENAI_API_KEY"] = ""

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = ""

if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = ""

if "COHERE_API_KEY" not in os.environ:
    os.environ["COHERE_API_KEY"] = ""

if "NVIDIA_API_KEY" not in os.environ:
    os.environ["NVIDIA_API_KEY"] = ""

if "FIREWORKS_API_KEY" not in os.environ:
    os.environ["FIREWORKS_API_KEY"] = ""

if "MISTRAL_API_KEY" not in os.environ:
    os.environ["MISTRAL_API_KEY"] = ""

if "WATSONX_APIKEY" not in os.environ:
    os.environ["WATSONX_APIKEY"] = ""

if "DATABRICKS_TOKEN" not in os.environ:
    os.environ["DATABRICKS_TOKEN"] = ""

if "XAI_API_KEY" not in os.environ:
    os.environ["XAI_API_KEY"] = ""

if "PPLX_API_KEY" not in os.environ:
    os.environ["PPLX_API_KEY"] = ""





print(os.environ["LANGSMITH_TRACING"], os.environ["LANGSMITH_API_KEY"], os.environ["LANGSMITH_PROJECT"])