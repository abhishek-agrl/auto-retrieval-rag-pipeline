import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Google Gemini Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

# Text splitter Configuration
TEXT_SPLITTER_CHUNK_SIZE=2500
TEXT_SPLITTER_CHUNK_OVERLAP=0

# Model configurations
GEMINI_MODEL = "gemma-4-26b-a4b-it"
EMBEDDING_MODEL = "gemini-embedding-2"

# Hyperparameters
LLM_TEMPERATURE = 0.25
VECTORSTORE_CONFIDENCE_THRESHOLD = 0.5
VECTORSTORE_DOCUMENT_LIMIT = 5