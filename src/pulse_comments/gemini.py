import json
import os
import random
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .schema import (
    INSTRUCTIONS, INSTRUCTION_HASH, INSTRUCTION_VERSION, DuplicateJSONField, LabelValidationError,
    api_schema, json_loads, validate_label,
)


MODEL_ID = "gemini-3.5-flash-lite"
ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_ID}:generateContent"
TRANSIENT = {408, 429, 500, 502, 503, 504}
MAX_ATTEMPTS = 4
MAX_FORMAT_ATTEMPTS = 3
MAX_WAIT_SECONDS = 60
BLOCK_REASONS = {
    "SAFETY", "OTHER", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "RECITATION", "IMAGE_SAFETY", "MODEL_ARMOR",
}
OTHER_FINISH_REASONS = {
    "FINISH_REASON_UNSPECIFIED", "LANGUAGE", "MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL",
    "TOO_MANY_TOOL_CALLS", "IMAGE_OTHER", "NO_IMAGE",
}


class GeminiError(RuntimeError):
    pass


class InvalidGeminiResponse(GeminiError):
    def __init__(self, code, detail, *, retryable=True):
        super().__init__(f"{code}: {detail}")
        self.retryable = retryable
        self.diagnostics = [{"code": code, "detail": detail}]


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def retry_after(value):
    if not value:
        return 0
    try:
        if value.isdecimal():
            return int(value)
        return max(0, (parsedate_to_datetime(value).astimezone(UTC) - datetime.now(UTC)).total_seconds())
    except (TypeError, ValueError, OverflowError):
        return 0


def make_request(key, text, feedback=None):
    system_prompt = INSTRUCTIONS
    if feedback is not None:
        system_prompt += (
            "\nPrevious response did not pass local validation. Correct this issue and return only "
            f"one JSON object with the same schema. {feedback}"
        )
    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": json.dumps({"comment": text}, ensure_ascii=False)}]}],
        "generationConfig": {
            "maxOutputTokens": 2048,
            "responseFormat": {"text": {"mimeType": "APPLICATION_JSON", "schema": api_schema()}},
        },
    }
    return Request(ENDPOINT, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                   headers={"Content-Type": "application/json", "x-goog-api-key": key}, method="POST")


def decode_json(value, stage):
    try:
        return json_loads(value)
    except json.JSONDecodeError as exc:
        raise InvalidGeminiResponse(
            f"{stage}_json", f"Invalid JSON on line {exc.lineno}, column {exc.colno}. Return exactly one JSON object."
        ) from None
    except UnicodeError:
        raise InvalidGeminiResponse(f"{stage}_json", "Invalid text encoding") from None
    except DuplicateJSONField:
        raise InvalidGeminiResponse(f"{stage}_json", "JSON contains a repeated field") from None
    except ValueError:
        raise InvalidGeminiResponse(f"{stage}_json", "JSON number cannot be parsed") from None


def block_error(reason):
    safe_reason = reason if isinstance(reason, str) and reason in BLOCK_REASONS else "unspecified"
    return InvalidGeminiResponse("provider_block", f"Gemini blocked generation ({safe_reason})", retryable=False)


def parse_response(response, text):
    if not isinstance(response, dict):
        raise InvalidGeminiResponse("provider_envelope", "Expected a response object")
    feedback = response.get("promptFeedback")
    if isinstance(feedback, dict):
        reason = feedback.get("blockReason")
        if reason and reason != "BLOCK_REASON_UNSPECIFIED":
            raise block_error(reason)
    candidates = response.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 1 or not isinstance(candidates[0], dict):
        raise InvalidGeminiResponse("provider_candidate", "Expected exactly one candidate")
    candidate = candidates[0]
    reason = candidate.get("finishReason")
    if reason == "MAX_TOKENS":
        raise InvalidGeminiResponse("output_truncated", "MAX_TOKENS before completion. Return a compact JSON object.")
    if isinstance(reason, str) and reason in BLOCK_REASONS:
        raise block_error(reason)
    if reason != "STOP":
        safe_reason = reason if isinstance(reason, str) and reason in OTHER_FINISH_REASONS else "missing or unrecognized"
        raise InvalidGeminiResponse("provider_finish", f"Unexpected finish reason ({safe_reason})", retryable=False)
    content = candidate.get("content")
    parts = content.get("parts") if isinstance(content, dict) else None
    if not isinstance(parts, list) or any(not isinstance(part, dict) for part in parts):
        raise InvalidGeminiResponse("candidate_text", "Expected a list of candidate parts with JSON text")
    fragments = [part["text"] for part in parts if "text" in part and not part.get("thought")]
    if any(not isinstance(fragment, str) for fragment in fragments) or not "".join(fragments).strip():
        raise InvalidGeminiResponse("candidate_text", "Candidate contains no usable JSON text")
    try:
        label = validate_label(decode_json("".join(fragments), "label"), text)
    except LabelValidationError as exc:
        raise InvalidGeminiResponse(exc.code, str(exc)) from None
    model_version = response.get("modelVersion")
    if not isinstance(model_version, str) or not model_version.strip():
        raise InvalidGeminiResponse("missing_model_version", "API response did not identify the model version", retryable=False)
    return {
        "provider": "google-gemini-developer-api", "model": MODEL_ID, "model_version": model_version,
        "instruction_version": INSTRUCTION_VERSION, "instruction_sha256": INSTRUCTION_HASH,
        "processed_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "response_id": response.get("responseId"), "usage_metadata": response.get("usageMetadata", {}),
        "label": label,
    }


class GeminiClient:
    def __init__(self, requests_per_minute, *, opener=None, sleep=time.sleep, clock=time.monotonic):
        self.key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not self.key:
            raise GeminiError("Set GEMINI_API_KEY in the process environment")
        if not 1 <= requests_per_minute <= 60:
            raise GeminiError("Choose a pace between 1 and 60 requests per minute")
        self.interval = 60 / requests_per_minute
        self.opener = opener or build_opener(NoRedirect())
        self.sleep, self.clock = sleep, clock
        self.next_request = 0

    def classify(self, text):
        diagnostics = []
        feedback = None
        for attempt in range(1, MAX_FORMAT_ATTEMPTS + 1):
            request = make_request(self.key, text, feedback)
            try:
                annotation = parse_response(decode_json(self._send(request), "provider"), text)
            except InvalidGeminiResponse as error:
                diagnostics.extend(item | {"attempt": attempt} for item in error.diagnostics)
                if not error.retryable or attempt == MAX_FORMAT_ATTEMPTS:
                    error.diagnostics = diagnostics
                    raise
                feedback = str(error)
            except GeminiError as error:
                if diagnostics:
                    summary = "; ".join(f"{item['code']}: {item['detail']}" for item in diagnostics)
                    raise GeminiError(f"{error} Earlier format diagnostics: {summary}.") from None
                raise
            else:
                if diagnostics:
                    annotation["format_diagnostics"] = diagnostics
                return annotation

    def _send(self, request):
        for attempt in range(MAX_ATTEMPTS):
            wait = self.next_request - self.clock()
            if wait > 0:
                self.sleep(wait)
            self.next_request = self.clock() + self.interval
            service_wait = 0
            try:
                with self.opener.open(request, timeout=45) as response:
                    return response.read()
            except HTTPError as error:
                status = error.code
                service_wait = retry_after(error.headers.get("Retry-After"))
                error.close()
                if status not in TRANSIENT:
                    raise GeminiError(f"Gemini HTTP {status}. Check project access, region and API configuration.") from None
                if status == 429 and (attempt == MAX_ATTEMPTS - 1 or service_wait > MAX_WAIT_SECONDS):
                    raise GeminiError("Gemini quota exceeded. Check RPM, TPM and RPD in AI Studio and resume later.") from None
                failure = f"Gemini temporary HTTP {status}"
            except (URLError, TimeoutError, ConnectionError):
                failure = "Temporary connection error to Gemini"
            if attempt == MAX_ATTEMPTS - 1 or service_wait > MAX_WAIT_SECONDS:
                raise GeminiError(f"{failure}. Saved results can be resumed.") from None
            self.sleep(max(service_wait, 2**attempt + random.uniform(0, 0.5)))

        raise GeminiError("Gemini retry attempts exhausted")
