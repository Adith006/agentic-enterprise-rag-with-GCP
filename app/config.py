import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    PROJECT_ID="agentic-enterprise-rag"
    LOCATION="us-central1"
    GCP_DOC_AI_LOCATION="us"
    GCP_DOC_AI_PROCESSOR_ID="GCP_DOC_AI_PROCESSOR_ID"
    GCP_RAW_BUCKET="agentic-enterprise-rag-raw"
    GCP_PROCESSED_BUCKET="agentic-enterprise-rag-processed"
    VPC_CONNECTOR="vpc-test	"

    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip()

    QDRANT_URL = os.getenv("QDRANT_CLUSTER_ENDPOINT")

    QDRANT_COLLECTION = "enterprise_rag"

    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

    GROQ_MODEL = "openai/gpt-oss-120b"


    DB_USER = os.getenv("DB_USER","postgres")
    DB_PASS = os.getenv("DB_PASS")
    DB_NAME = os.getenv("DB_NAME","postgres")
    DB_CONNECTION_NAME = os.getenv("DB_CONNECTION_NAME")

    REDIS_HOST = os.getenv("REDIS_HOST","localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT",6739))

    LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING","true")
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
    LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT","enterprise-rag")
    LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT","https://api.smith.langchain.com")


os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGSMITH_TRACING","true")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] =  os.getenv("LANGSMITH_PROJECT","enterprise-rag") 
os.environ["LANGCHAIN_ENDPOINT"] =  os.getenv("LANGSMITH_ENDPOINT","https://api.smith.langchain.com")

settings = Settings()



  

  