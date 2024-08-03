import os

import jwt
from cachetools import TTLCache
from dotenv import load_dotenv

load_dotenv()


class RAGConfig:
    API_KEY = os.environ.get("GEMINI_API_KEY")
    GENERATION_CONFIG = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8194,
        "response_mime_type": "text/plain",
    }
    MODEL_NAME = "gemini-1.5-flash"
    JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY")
    GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")
    JWT_ALGORITHM = "HS256"
    API_V1_PREFIX = "/api/v1"

    @staticmethod
    def generate_jwt_key(data: dict, expires_in: int = 3600) -> str:
        """Generates a JWT key."""
        return jwt.encode(
            {"data": data, "exp": expires_in},
            RAGConfig.JWT_SECRET_KEY,
            algorithm=RAGConfig.JWT_ALGORITHM,
        )


class CacheConfig:
    # --- Caching Configuration ---
    CACHE_DEFAULT_TIMEOUT = 300  # Cache timeout in seconds
    repo_metadata_cache = TTLCache(maxsize=100, ttl=CACHE_DEFAULT_TIMEOUT)
    file_content_cache = TTLCache(maxsize=100, ttl=CACHE_DEFAULT_TIMEOUT)
    CACHE_TYPE = "SimpleCache"
    CACHE_KEY_PREFIX = "rag_cache_"
