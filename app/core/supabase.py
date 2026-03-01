from supabase import create_client, Client
from app.core.config import settings

_client: Client | None = None


def get_supabase() -> Client:
    global _client
    if _client is None:
        if not settings.supabase_url or not settings.supabase_key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set in .env")
        _client = create_client(settings.supabase_url, settings.supabase_key)
        _ensure_buckets(_client)
    return _client


def _ensure_buckets(client: Client) -> None:
    """Create storage buckets if they don't exist. Skips silently if permissions deny it."""
    try:
        existing = {b.name for b in client.storage.list_buckets()}
    except Exception:
        return
    for name in ("results", "generated"):
        if name not in existing:
            try:
                client.storage.create_bucket(name, options={"public": True})
            except Exception:
                pass  # bucket likely exists or anon key lacks create permission
