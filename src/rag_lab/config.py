from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://rag:rag@localhost:5432/rag"
    embedding_backend: str = "local"   # local | voyage | openai
    embedding_dim: int = 384
    chunk_size: int = 700              # target characters per chunk
    chunk_overlap: int = 100
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-6"


settings = Settings()
