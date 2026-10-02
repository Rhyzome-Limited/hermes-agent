"""Google Vertex AI provider profile: Gemini via Google Cloud's OpenAI-compatible endpoint.

Auth is OAuth2 (service-account JSON or ADC), not a static key: ``agent/vertex_adapter.py``
mints ``(token, base_url)`` and the token is passed as ``api_key``. ``auth_type="vertex"``
keeps it out of the api_key provider path so a credentials-file path is never mistaken for a key.
"""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class VertexProfile(ProviderProfile):
    """Vertex AI — reuse Gemini's thinking_config translation for extra_body."""

    def build_extra_body(self, *, session_id: str | None = None, **context: Any) -> dict[str, Any]:
        """Emit ``extra_body.google.thinking_config`` like the ``gemini`` provider's OpenAI-compat subpath."""
        from agent.transports.chat_completions import _build_gemini_thinking_config, _snake_case_gemini_thinking_config

        raw = _build_gemini_thinking_config(context.get("model") or "", context.get("reasoning_config"))
        # TEMP (maria-demo): Vertex's OpenAI-compat stream sends Gemini thought summaries as plain
        # delta.content (3.5 Flash always, 3.8 rarely), so they leak into the reply; Hermes never
        # captures them on this path. Keep thinking (thinking_level) but don't ask for summaries.
        if raw:
            raw["includeThoughts"] = False
        thinking_config = _snake_case_gemini_thinking_config(raw) if raw else None
        return {"extra_body": {"google": {"thinking_config": thinking_config}}} if thinking_config else {}


vertex = VertexProfile(
    name="vertex", aliases=("google-vertex", "vertex-ai", "gcp-vertex", "vertexai"), api_mode="chat_completions",
    env_vars=(),  # OAuth2 via service account / ADC — not a static key env var
    base_url="https://aiplatform.googleapis.com",  # real base_url computed at runtime
    auth_type="vertex", default_aux_model="google/gemini-3.6-flash",
    supports_model_listing=False,  # no /models route on the OpenAI-compat endpoint; setup ships a curated list
)

register_provider(vertex)
