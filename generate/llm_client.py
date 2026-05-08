import json
import re
import time
import logging
import asyncio
import httpx
from tenacity import (
    retry, stop_after_attempt, wait_exponential, retry_if_exception_type
)
from .models import LLMConfig, ProxyConfig

logger = logging.getLogger(__name__)


def _build_proxy_url(proxy: ProxyConfig) -> str:
    auth = ""
    if proxy.username:
        auth = f"{proxy.username}:{proxy.password}@" if proxy.password else f"{proxy.username}@"
    return f"{proxy.protocol}://{auth}{proxy.host}:{proxy.port}"


class LLMClient:
    def __init__(self, llm_config: LLMConfig, proxy_config: ProxyConfig):
        self.config = llm_config
        self.proxy = proxy_config
        self.semaphore = asyncio.Semaphore(llm_config.max_concurrency)
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is not None:
            return self._client

        kwargs = {
            "timeout": httpx.Timeout(self.config.timeout_seconds),
            "trust_env": False,  # Don't use HTTP_PROXY/HTTPS_PROXY env vars — we control proxy via config
        }

        if self.proxy.enabled:
            if self.proxy.protocol == "socks5":
                try:
                    from httpx_socks import AsyncProxyTransport
                    proxy_url = _build_proxy_url(self.proxy)
                    transport = AsyncProxyTransport.from_url(proxy_url)
                    kwargs["transport"] = transport
                except ImportError:
                    logger.warning("httpx-socks not installed, falling back to direct connection")
            else:
                proxy_url = _build_proxy_url(self.proxy)
                kwargs["proxy"] = proxy_url

        self._client = httpx.AsyncClient(**kwargs)
        return self._client

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type((
            httpx.TimeoutException, httpx.HTTPStatusError, httpx.ConnectError,
            httpx.RemoteProtocolError, httpx.ReadError,
        )),
        reraise=True,
    )
    async def generate(self, messages: list[dict], temperature: float = 0.9) -> dict:
        async with self.semaphore:
            client = await self._get_client()
            response = await client.post(
                f"{self.config.base_url.rstrip('/')}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.config.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": 16384,
                },
            )

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After", "5")
                wait = int(retry_after) if retry_after.isdigit() else 5
                logger.warning(f"Rate limited (429), waiting {wait}s before retry")
                await asyncio.sleep(wait)
                raise httpx.HTTPStatusError(
                    f"429 Too Many Requests", request=response.request, response=response
                )

            response.raise_for_status()
            return response.json()


def parse_llm_response(response_text: str) -> list[dict]:
    """Parse LLM response text into a list of JSON objects.

    Handles common malformations: markdown code fences, list markers,
    trailing commas, truncated output.
    """
    text = response_text.strip()

    # Remove markdown code fences if present
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*```$', '', text, flags=re.MULTILINE)

    samples = []
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue

        # Strip list markers like "1. " or "- "
        line = re.sub(r'^[\d]+\.[\s]*|^-\s*', '', line)

        try:
            obj = json.loads(line)
            samples.append(obj)
        except json.JSONDecodeError:
            # Try to find a JSON object within the line
            match = re.search(r'\{.*\}', line, re.DOTALL)
            if match:
                try:
                    obj = json.loads(match.group())
                    samples.append(obj)
                except json.JSONDecodeError:
                    logger.warning(f"Skipping malformed line: {line[:100]}...")
            else:
                logger.warning(f"Skipping non-JSON line: {line[:100]}...")

    return samples


async def generate_and_parse(
    client: LLMClient, request: dict, temperature: float = 0.9
) -> list[dict]:
    """Send a generation request and parse the response into samples."""
    messages = request["messages"]
    try:
        response = await client.generate(messages, temperature)
        content = response["choices"][0]["message"]["content"]
        tokens_used = response.get("usage", {}).get("total_tokens", 0)
        samples = parse_llm_response(content)
        logger.info(f"Generated {len(samples)} samples, used {tokens_used} tokens")
        return samples
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return []
