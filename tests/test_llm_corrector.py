"""Behavioural tests for the optional LLM post-editor.

The tests exercise real HTTP through a loopback server and load a fixture YAML
config that contains only fake credentials.  No test touches the user's real
``~/.omp/agent/models.yml``.
"""

import json
import os
import socket
import tempfile
import threading
import unittest
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from llm_corrector import (
    API_KEY_ENV,
    BASE_URL_ENV,
    FALLBACK_API_KEY_ENV,
    MODEL_ENV,
    LLMCorrectorError,
    LLMTLCorrector,
    load_corrector,
)


class _LLMServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.requests = []
        self.status = 200
        self.payload = {}
        self.raw = None

    @property
    def port(self):
        return self.server_address[1]


class _LLMHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw_body = self.rfile.read(length) if length else b""
        try:
            parsed = json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            parsed = None
        self.server.requests.append({
            "path": self.path,
            "headers": dict(self.headers.items()),
            "body": raw_body,
            "json": parsed,
        })
        raw = self.server.raw
        if raw is None:
            raw = json.dumps(self.server.payload).encode("utf-8")
        self.send_response(self.server.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):  # keep test output clean
        pass


@contextmanager
def running_server():
    server = _LLMServer(("127.0.0.1", 0), _LLMHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _CorrectorTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self._saved_env = {}
        for name in list(os.environ):
            if name in {API_KEY_ENV, BASE_URL_ENV, MODEL_ENV, FALLBACK_API_KEY_ENV} \
                    or name.startswith(f"{API_KEY_ENV}_"):
                self._saved_env[name] = os.environ.pop(name)
        self.addCleanup(self._restore_env)

    def _restore_env(self):
        for name in list(os.environ):
            if name in {API_KEY_ENV, BASE_URL_ENV, MODEL_ENV, FALLBACK_API_KEY_ENV} \
                    or name.startswith(f"{API_KEY_ENV}_"):
                del os.environ[name]
        os.environ.update(self._saved_env)

    def write_config(self, name="models.yml", *, base_url, api_key="test-key",
                     models=("model-one", "model-two"), provider="netra"):
        lines = ["providers:", f"  {provider}:"]
        if base_url is not None:
            lines.append(f"    baseUrl: {base_url}")
        if api_key is not None:
            lines.append(f"    apiKey: {api_key}")
        lines.append("    models:")
        for model in models:
            lines.append(f"      - id: {model}")
        path = Path(self.tmp.name) / name
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def loopback_config(self, server, **kwargs):
        kwargs.setdefault("base_url", f"http://127.0.0.1:{server.port}/v1")
        return self.write_config(**kwargs)


class SuccessTests(_CorrectorTestCase):
    def test_posts_source_and_draft_to_chat_completions(self):
        with running_server() as server:
            server.payload = {"choices": [{"message": {"content": "  Corrected text.  "}}]}
            config = self.loopback_config(server)
            corrector = load_corrector(config_path=config)
            result = corrector.correct("こんにちは", "Hello draft", "ja", "en")

        self.assertEqual(result, "Corrected text.")
        self.assertEqual(len(server.requests), 1)
        request = server.requests[0]
        self.assertEqual(request["path"], "/v1/chat/completions")
        self.assertEqual(request["headers"].get("Authorization"), "Bearer test-key")
        self.assertEqual(request["headers"].get("Content-Type"), "application/json")
        body = request["json"]
        self.assertEqual(body["model"], "model-one")
        self.assertFalse(body["stream"])
        system, user = body["messages"]
        self.assertEqual(system["role"], "system")
        self.assertIn("en", system["content"])
        self.assertEqual(user["role"], "user")
        self.assertIn("こんにちは", user["content"])
        self.assertIn("Hello draft", user["content"])
        self.assertIn("ja", user["content"])
        self.assertIn("en", user["content"])

    def test_defaults_to_first_listed_model(self):
        with running_server() as server:
            server.payload = {"choices": [{"message": {"content": "ok"}}]}
            config = self.loopback_config(server)
            load_corrector(config_path=config).correct("a", "b", "ja", "en")
        self.assertEqual(server.requests[0]["json"]["model"], "model-one")

    def test_named_model_selection(self):
        with running_server() as server:
            server.payload = {"choices": [{"message": {"content": "ok"}}]}
            config = self.loopback_config(server)
            corrector = load_corrector(config_path=config, model="model-two")
            self.assertEqual(corrector.model, "model-two")
            corrector.correct("a", "b", "ja", "en")
        self.assertEqual(server.requests[0]["json"]["model"], "model-two")

    def test_model_environment_override_wins_over_default(self):
        with running_server() as server:
            server.payload = {"choices": [{"message": {"content": "ok"}}]}
            config = self.loopback_config(server)
            os.environ[MODEL_ENV] = "model-two"
            load_corrector(config_path=config).correct("a", "b", "ja", "en")
        self.assertEqual(server.requests[0]["json"]["model"], "model-two")

    def test_plain_string_model_entries_supported(self):
        with running_server() as server:
            server.payload = {"choices": [{"message": {"content": "ok"}}]}
            path = Path(self.tmp.name) / "strings.yml"
            path.write_text(
                "providers:\n"
                "  netra:\n"
                f"    baseUrl: http://127.0.0.1:{server.port}/v1\n"
                "    apiKey: test-key\n"
                "    models:\n"
                "      - plain-one\n"
                "      - plain-two\n",
                encoding="utf-8",
            )
            corrector = load_corrector(config_path=path)
            self.assertEqual(corrector.model, "plain-one")
            corrector.correct("a", "b", "ja", "en")
            self.assertEqual(server.requests[0]["json"]["model"], "plain-one")

    def test_content_parts_are_joined(self):
        with running_server() as server:
            server.payload = {"choices": [{"message": {"content": [
                {"type": "text", "text": "part one "},
                {"type": "text", "text": "part two"},
            ]}}]}
            config = self.loopback_config(server)
            result = load_corrector(config_path=config).correct("a", "b", "ja", "en")
        self.assertEqual(result, "part one part two")

    def test_api_key_from_environment(self):
        with running_server() as server:
            server.payload = {"choices": [{"message": {"content": "ok"}}]}
            config = self.loopback_config(server, api_key=None)
            os.environ[API_KEY_ENV] = "env-key"
            load_corrector(config_path=config).correct("a", "b", "ja", "en")
        self.assertEqual(server.requests[0]["headers"].get("Authorization"), "Bearer env-key")

    def test_provider_specific_environment_key(self):
        with running_server() as server:
            server.payload = {"choices": [{"message": {"content": "ok"}}]}
            config = self.loopback_config(server, api_key=None)
            os.environ[f"{API_KEY_ENV}_NETRA"] = "netra-key"
            load_corrector(config_path=config).correct("a", "b", "ja", "en")
        self.assertEqual(server.requests[0]["headers"].get("Authorization"), "Bearer netra-key")

    def test_explicit_overrides_skip_config_entirely(self):
        missing = Path(self.tmp.name) / "does-not-exist.yml"
        with running_server() as server:
            server.payload = {"choices": [{"message": {"content": "ok"}}]}
            corrector = load_corrector(
                config_path=missing,
                base_url=f"http://127.0.0.1:{server.port}/v1",
                api_key="override-key",
                model="override-model",
            )
            self.assertEqual(corrector.correct("a", "b", "ja", "en"), "ok")
        self.assertEqual(server.requests[0]["json"]["model"], "override-model")
        self.assertEqual(server.requests[0]["headers"].get("Authorization"),
                         "Bearer override-key")

    def test_repr_does_not_leak_api_key(self):
        corrector = LLMTLCorrector(base_url="https://api.example.com/v1",
                                   api_key="super-secret", model="m")
        self.assertNotIn("super-secret", repr(corrector))
        self.assertIn("***", repr(corrector))


class ConfigurationErrorTests(_CorrectorTestCase):
    def test_missing_config_file(self):
        missing = Path(self.tmp.name) / "missing.yml"
        with self.assertRaisesRegex(LLMCorrectorError, "not found"):
            load_corrector(config_path=missing)

    def test_unknown_provider_lists_available(self):
        config = self.write_config(base_url="https://api.example.com/v1")
        with self.assertRaisesRegex(LLMCorrectorError, "available: netra"):
            load_corrector(config_path=config, provider="ghost")

    def test_missing_providers_mapping(self):
        path = Path(self.tmp.name) / "bad.yml"
        path.write_text("something: else\n", encoding="utf-8")
        with self.assertRaisesRegex(LLMCorrectorError, "providers"):
            load_corrector(config_path=path)

    def test_invalid_yaml(self):
        path = Path(self.tmp.name) / "broken.yml"
        path.write_text("providers: [unclosed\n", encoding="utf-8")
        with self.assertRaisesRegex(LLMCorrectorError, "not valid YAML"):
            load_corrector(config_path=path)

    def test_unknown_named_model_lists_available(self):
        config = self.write_config(base_url="https://api.example.com/v1")
        with self.assertRaisesRegex(LLMCorrectorError, "model-one"):
            load_corrector(config_path=config, model="ghost")

    def test_missing_api_key(self):
        config = self.write_config(base_url="https://api.example.com/v1",
                                   api_key=None)
        with self.assertRaisesRegex(LLMCorrectorError, "API key"):
            load_corrector(config_path=config)

    def test_missing_base_url(self):
        config = self.write_config(base_url=None)
        with self.assertRaisesRegex(LLMCorrectorError, "baseUrl"):
            load_corrector(config_path=config)

    def test_empty_model_list(self):
        path = Path(self.tmp.name) / "empty.yml"
        path.write_text(
            "providers:\n"
            "  netra:\n"
            "    baseUrl: https://api.example.com/v1\n"
            "    apiKey: test-key\n"
            "    models: []\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(LLMCorrectorError, "no models"):
            load_corrector(config_path=path)

    def test_remote_plain_http_base_url_rejected(self):
        with self.assertRaisesRegex(LLMCorrectorError, "refusing insecure"):
            load_corrector(base_url="http://api.example.com/v1",
                           api_key="k", model="m")

    def test_relative_base_url_rejected(self):
        with self.assertRaisesRegex(LLMCorrectorError, "invalid LLM base URL"):
            load_corrector(base_url="api.example.com/v1", api_key="k", model="m")

    def test_direct_constructor_requires_key_and_model(self):
        with self.assertRaises(LLMCorrectorError):
            LLMTLCorrector(base_url="https://api.example.com/v1", api_key="", model="m")
        with self.assertRaises(LLMCorrectorError):
            LLMTLCorrector(base_url="https://api.example.com/v1", api_key="k", model="  ")

    def test_empty_inputs_rejected_without_request(self):
        with running_server() as server:
            config = self.loopback_config(server)
            corrector = load_corrector(config_path=config)
            with self.assertRaisesRegex(LLMCorrectorError, "original"):
                corrector.correct("   ", "draft", "ja", "en")
            with self.assertRaisesRegex(LLMCorrectorError, "draft"):
                corrector.correct("original", "", "ja", "en")
            self.assertEqual(server.requests, [])


class EndpointFailureTests(_CorrectorTestCase):
    def test_http_error_reports_status_without_secret(self):
        with running_server() as server:
            server.status = 500
            server.payload = {"error": {"message": "upstream exploded"}}
            config = self.loopback_config(server)
            corrector = load_corrector(config_path=config)
            with self.assertRaises(LLMCorrectorError) as caught:
                corrector.correct("a", "b", "ja", "en")
        message = str(caught.exception)
        self.assertIn("HTTP 500", message)
        self.assertIn("upstream exploded", message)
        self.assertNotIn("test-key", message)
        self.assertIsNone(caught.exception.__cause__)

    def test_server_echoing_credential_cannot_expose_it(self):
        with running_server() as server:
            config = self.loopback_config(server)
            corrector = load_corrector(config_path=config)
            for status in (401, 200):
                server.status = status
                server.payload = {"error": {"message": "invalid credential test-key"}}
                with self.assertRaises(LLMCorrectorError) as caught:
                    corrector.correct("a", "b", "ja", "en")
                self.assertNotIn("test-key", str(caught.exception))
                self.assertIn("invalid credential", str(caught.exception))

    def test_error_object_in_success_body(self):
        with running_server() as server:
            server.payload = {"error": {"message": "rate limited"}}
            config = self.loopback_config(server)
            corrector = load_corrector(config_path=config)
            with self.assertRaisesRegex(LLMCorrectorError, "rate limited"):
                corrector.correct("a", "b", "ja", "en")

    def test_malformed_json_response(self):
        with running_server() as server:
            server.raw = b"<html>nope</html>"
            config = self.loopback_config(server)
            corrector = load_corrector(config_path=config)
            with self.assertRaisesRegex(LLMCorrectorError, "not valid JSON"):
                corrector.correct("a", "b", "ja", "en")

    def test_missing_choices(self):
        with running_server() as server:
            server.payload = {}
            config = self.loopback_config(server)
            corrector = load_corrector(config_path=config)
            with self.assertRaisesRegex(LLMCorrectorError, "no choices"):
                corrector.correct("a", "b", "ja", "en")

    def test_empty_and_blank_replies_rejected(self):
        for content in ("", "   "):
            with self.subTest(content=content), running_server() as server:
                server.payload = {"choices": [{"message": {"content": content}}]}
                config = self.loopback_config(server)
                corrector = load_corrector(config_path=config)
                with self.assertRaisesRegex(LLMCorrectorError, "empty correction"):
                    corrector.correct("a", "b", "ja", "en")

    def test_unreachable_endpoint(self):
        port = _free_port()
        corrector = LLMTLCorrector(base_url=f"http://127.0.0.1:{port}/v1",
                                   api_key="test-key", model="m")
        with self.assertRaisesRegex(LLMCorrectorError, "cannot reach LLM endpoint"):
            corrector.correct("a", "b", "ja", "en")


if __name__ == "__main__":
    unittest.main()
