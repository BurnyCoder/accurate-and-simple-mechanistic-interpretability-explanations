"""delphi's OpenRouter client plus full prompt/response logging and provider pinning.

Global context: scoring and explaining both go through delphi's own `OpenRouter` client
(delphi/clients/openrouter.py). This subclass only adds what reproducibility needs:
- every prompt and response is logged in full, tagged with the (latent, condition) job;
- fixed request fields (provider routing, seed) are merged into every request body;
- the provider that served each call and its token usage are logged, for cost reporting.
"""

import contextvars
import json
import logging

from delphi.clients import OpenRouter

from . import config

llm_log = logging.getLogger("concise_interp.llm")
# Set per job in scoring/explaining; asyncio tasks copy the context, so delphi's inner
# per-batch tasks inherit the tag and every log line says which job it belongs to.
job_tag: contextvars.ContextVar[str] = contextvars.ContextVar("job_tag", default="-")


class _MergeIntoBody:
    """Wraps httpx.AsyncClient so `post(json=...)` also carries fixed extra fields.
    delphi builds the request body inside `generate`, so this is the narrowest hook."""

    def __init__(self, inner, extra: dict):
        self.inner, self.extra = inner, extra

    async def post(self, url, json, **kwargs):
        return await self.inner.post(url, json={**json, **self.extra}, **kwargs)


class LoggedOpenRouter(OpenRouter):
    def __init__(self, model: str, provider: dict | None = None, seed: int | None = 0, **kwargs):
        super().__init__(model, api_key=config.OPENROUTER_API_KEY, base_url=config.OPENROUTER_URL, **kwargs)
        # OpenRouter provider routing: https://openrouter.ai/docs/features/provider-routing
        extra = {k: v for k, v in {"provider": provider, "seed": seed}.items() if v is not None}
        self.client = _MergeIntoBody(self.client, extra)

    def postprocess(self, response):
        body = response.json()
        if "choices" not in body:  # OpenRouter error payload; delphi's retry loop handles it
            llm_log.warning("ERROR-BODY [%s] %s", job_tag.get(), json.dumps(body))
        else:
            llm_log.info("META [%s] provider=%s usage=%s", job_tag.get(),
                         body.get("provider"), json.dumps(body.get("usage")))
        return super().postprocess(response)

    async def generate(self, prompt, **kwargs):
        tag = job_tag.get()
        llm_log.info("PROMPT [%s] %s", tag, json.dumps(prompt, ensure_ascii=False))
        try:
            response = await super().generate(prompt, **kwargs)
        except Exception as e:
            llm_log.error("FAILED [%s] %r", tag, e)
            raise
        llm_log.info("RESPONSE [%s] %s", tag, response.text)
        return response
