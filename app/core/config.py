from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    app_name: str = "peachmenor"
    debug: bool = False
    version: str = "2.0.0"
    google_api_key: str = ""
    google_model: str = "gemini-2.0-flash"
    mistral_api_key: str = ""
    mistral_model: str = "pixtral-12b-2409"
    supabase_url: str = ""
    supabase_key: str = ""
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"



settings = Settings()
