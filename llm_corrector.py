"""Optional LLM correction of offline machine translations.

This module is deliberately opt-in: importing it reads no configuration and no
credentials.  :func:`load_corrector` is the only entry point that touches the
user's provider config, and it is meant to be called by a caller that has
already decided to enable LLM post-editing.

The corrector sends the *source original* plus the *offline draft* to an
OpenAI-compatible ``/chat/completions`` endpoint and returns the corrected
translation.  Every failure is reported through :class:`LLMCorrectorError`
with an actionable message; request internals and secrets are never included
in an error or in ``repr``.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path.home() / ".omp" / "agent" / "models.yml"
"""Default location of the agent provider configuration."""

API_KEY_ENV = "MANGA_LLM_API_KEY"
BASE_URL_ENV = "MANGA_LLM_BASE_URL"
MODEL_ENV = "MANGA_LLM_MODEL"
FALLBACK_API_KEY_ENV = "OPENAI_API_KEY"

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_MAX_ERROR_DETAIL = 200
_DEFAULT_TIMEOUT = 60.0

_SYSTEM_PROMPT = (
    "You are a meticulous professional translator and editor. "
    "You receive an original text and a machine-translated draft of it. "
    "Rewrite the draft into natural, idiomatic {target} that preserves the "
    "original meaning exactly. Fix grammar, word order, terminology and "
    "register. Do not add, omit, explain, transliterate or censor content. "
    "Preserve paragraph and line breaks. Keep the length close to the draft. "
    "Reply with only the corrected {target} text and nothing else."
)


class LLMCorrectorError(RuntimeError):
    """Raised when the optional LLM corrector cannot be configured or reached."""


class LLMTLCorrector:
    """Post-edit an offline translation draft with an OpenAI-compatible model."""

    def __init__(self, base_url: str, api_key: str, model: str,
                 provider: str = "netra", timeout: float = _DEFAULT_TIMEOUT) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise LLMCorrectorError("an API key is required for the LLM corrector")
        if not isinstance(model, str) or not model.strip():
            raise LLMCorrectorError("a model name is required for the LLM corrector")
        self.base_url = _validate_base_url(base_url)
        self.endpoint = f"{self.base_url}/chat/completions"
        self.provider = str(provider)
        self.model = model.strip()
        self.timeout = float(timeout)
        self._api_key = api_key.strip()

    def __repr__(self) -> str:
        return (
            f"LLMTLCorrector(provider={self.provider!r}, model={self.model!r}, "
            f"base_url={self.base_url!r}, api_key='***')"
        )

    def correct(self, original: str, draft: str, source_lang: str,
                target_lang: str) -> str:
        """Return the corrected draft, preserving the original's meaning."""
        original = _require_text(original, "original")
        draft = _require_text(draft, "draft")
        source_lang = _require_text(source_lang, "source_lang")
        target_lang = _require_text(target_lang, "target_lang")

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system",
                 "content": _SYSTEM_PROMPT.format(target=target_lang)},
                {"role": "user", "content": _user_prompt(
                    original, draft, source_lang, target_lang)},
            ],
            "temperature": 0,
            "stream": False,
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
        )
        host = _safe_host(self.base_url)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            detail = _server_message(exc).replace(self._api_key, "[redacted]")
            suffix = f": {detail}" if detail else ""
            raise LLMCorrectorError(
                f"LLM endpoint {host} returned HTTP {exc.code}{suffix}") from None
        except urllib.error.URLError as exc:
            raise LLMCorrectorError(
                f"cannot reach LLM endpoint {host} ({_reason(exc)})") from None
        except TimeoutError:
            raise LLMCorrectorError(
                f"LLM endpoint {host} timed out after {self.timeout:g}s") from None
        except OSError as exc:
            raise LLMCorrectorError(
                f"cannot reach LLM endpoint {host} ({exc.strerror or 'network error'})"
            ) from None
        return _parse_reply(body, self._api_key)


def load_corrector(config_path: Path | None = None, provider: str = "netra",
                   model: str | None = None, api_key: str | None = None,
                   base_url: str | None = None,
                   timeout: float = _DEFAULT_TIMEOUT) -> LLMTLCorrector:
    """Build a corrector from provider config and/or explicit overrides.

    ``config_path`` defaults to :data:`DEFAULT_CONFIG_PATH`.  The configuration
    is only read when the provider, model or credentials are not fully supplied
    through arguments and environment variables, so callers that override
    everything never touch the user's config file.
    """
    provider = (provider or "").strip()
    if not provider:
        raise LLMCorrectorError("a provider name is required to configure the LLM corrector")

    model = (model or os.environ.get(MODEL_ENV) or "").strip() or None
    base_url = (base_url or os.environ.get(BASE_URL_ENV) or "").strip() or None
    api_key = (api_key or os.environ.get(API_KEY_ENV)
               or os.environ.get(f"{API_KEY_ENV}_{_env_suffix(provider)}")
               or os.environ.get(FALLBACK_API_KEY_ENV) or "").strip() or None

    if not (base_url and api_key and model):
        path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
        providers = _read_providers(path)
        if provider not in providers:
            available = ", ".join(sorted(providers)) or "none"
            raise LLMCorrectorError(
                f"provider {provider!r} is not defined in {path} (available: {available})")
        provider_cfg = providers[provider]
        if not isinstance(provider_cfg, dict):
            raise LLMCorrectorError(f"provider {provider!r} in {path} is not a mapping")
        base_url = base_url or _text(provider_cfg.get("baseUrl"))
        api_key = api_key or _text(provider_cfg.get("apiKey"))
        listed = _model_ids(provider_cfg)
        if model is None:
            if not listed:
                raise LLMCorrectorError(
                    f"provider {provider!r} in {path} lists no models; "
                    "pass model= explicitly")
            model = listed[0]
        elif listed and model not in listed:
            raise LLMCorrectorError(
                f"model {model!r} is not listed for provider {provider!r} in {path} "
                f"(available: {', '.join(listed)})")
        if not base_url:
            raise LLMCorrectorError(
                f"provider {provider!r} in {path} defines no baseUrl")

    if not api_key:
        raise LLMCorrectorError(
            "no API key available for the LLM corrector; set it in the provider "
            f"config, pass api_key=, or export {API_KEY_ENV}")
    if not base_url:
        raise LLMCorrectorError("no base URL available for the LLM corrector")
    return LLMTLCorrector(base_url=base_url, api_key=api_key, model=model,
                          provider=provider, timeout=timeout)


def _env_suffix(provider: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in provider.upper())


def _text(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _model_ids(provider_cfg: dict) -> list[str]:
    models = provider_cfg.get("models")
    if not isinstance(models, list):
        return []
    ids: list[str] = []
    for entry in models:
        if isinstance(entry, dict):
            model_id = _text(entry.get("id"))
        else:
            model_id = _text(entry)
        if model_id:
            ids.append(model_id)
    return ids


def _read_providers(path: Path) -> dict:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise LLMCorrectorError(
            f"LLM provider config not found at {path}; create it or pass "
            "config_path=") from None
    except OSError as exc:
        raise LLMCorrectorError(
            f"cannot read LLM provider config at {path}: {exc.strerror or exc}"
        ) from None
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        raise LLMCorrectorError(
            f"LLM provider config at {path} is not valid YAML") from None
    if not isinstance(data, dict):
        raise LLMCorrectorError(
            f"LLM provider config at {path} must be a mapping with a 'providers' key")
    providers = data.get("providers")
    if not isinstance(providers, dict):
        raise LLMCorrectorError(
            f"LLM provider config at {path} has no 'providers' mapping")
    return providers


def _validate_base_url(base_url: object) -> str:
    if not isinstance(base_url, str) or not base_url.strip():
        raise LLMCorrectorError("a base URL is required for the LLM corrector")
    url = base_url.strip().rstrip("/")
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise LLMCorrectorError(
            f"invalid LLM base URL {base_url!r}; expected an absolute http(s) URL")
    if parsed.scheme == "http" and (parsed.hostname or "") not in _LOOPBACK_HOSTS:
        raise LLMCorrectorError(
            f"refusing insecure LLM base URL {base_url!r}; use https "
            "(http is allowed only for loopback hosts)")
    return url


def _safe_host(base_url: str) -> str:
    parsed = urllib.parse.urlsplit(base_url)
    host = parsed.hostname or "endpoint"
    if parsed.port:
        host = f"{host}:{parsed.port}"
    return host


def _require_text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise LLMCorrectorError(f"{name} must be a string")
    if not value.strip():
        raise LLMCorrectorError(f"{name} must not be empty")
    return value


def _user_prompt(original: str, draft: str, source_lang: str,
                 target_lang: str) -> str:
    return (
        f"Source language: {source_lang}\n"
        f"Target language: {target_lang}\n\n"
        f"Original ({source_lang}):\n{original}\n\n"
        f"Draft translation ({target_lang}):\n{draft}\n\n"
        f"Corrected translation ({target_lang}):"
    )


def _server_message(exc: urllib.error.HTTPError) -> str:
    try:
        raw = exc.read(4096)
    except Exception:  # noqa: BLE001 - body access must never mask the status
        return ""
    try:
        payload = json.loads(raw.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = _text(error.get("message"))
            if message:
                return message[:_MAX_ERROR_DETAIL]
        message = _text(payload.get("message"))
        if message:
            return message[:_MAX_ERROR_DETAIL]
    return raw.decode("utf-8", errors="replace").strip()[:_MAX_ERROR_DETAIL]


def _reason(exc: urllib.error.URLError) -> str:
    reason = exc.reason
    if isinstance(reason, OSError) and reason.strerror:
        return reason.strerror
    text = str(reason).strip()
    return text[:_MAX_ERROR_DETAIL] or "network error"


def _parse_reply(body: bytes, api_key: str) -> str:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise LLMCorrectorError(
            "LLM endpoint returned a response that was not valid JSON") from None
    if not isinstance(payload, dict):
        raise LLMCorrectorError("LLM endpoint returned an unexpected response shape")
    error = payload.get("error")
    if isinstance(error, dict):
        message = _text(error.get("message")) or "unknown error"
        raise LLMCorrectorError(
            f"LLM endpoint reported an error: {message.replace(api_key, '[redacted]')[:_MAX_ERROR_DETAIL]}")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMCorrectorError("LLM response contained no choices")
    first = choices[0]
    message = first.get("message") if isinstance(first, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    text = _content_text(content)
    if not text or not text.strip():
        raise LLMCorrectorError("LLM returned an empty correction")
    return text.strip()


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                piece = part.get("text", part.get("content"))
                if isinstance(piece, str):
                    parts.append(piece)
        return "".join(parts)
    return ""
