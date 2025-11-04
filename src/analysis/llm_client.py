"""LLM client abstraction for OpenAI, Anthropic, and future Bedrock."""
from dataclasses import dataclass
from typing import Dict, Any, Optional
import json
import os
import time
from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMResponse:
    """Structured LLM response."""
    content: Dict[str, Any]
    model: str
    tokens_used: int
    latency_ms: float


class LLMClient:
    """Abstract base class for LLM clients."""
    
    def call_structured(self, prompt: str, schema: Dict[str, Any], timeout: Optional[float] = None) -> LLMResponse:
        """
        Call LLM with structured output.
        
        Args:
            prompt: Input prompt
            schema: JSON schema for structured output
            timeout: Optional timeout in seconds
            
        Returns:
            LLMResponse with parsed content
        """
        raise NotImplementedError


class OpenAILLMClient(LLMClient):
    """OpenAI API client."""
    
    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        try:
            import openai
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")
        
        self.client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
    
    def call_structured(self, prompt: str, schema: Dict[str, Any], timeout: Optional[float] = None) -> LLMResponse:
        """Call OpenAI with structured output."""
        start_time = time.time()
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            timeout=timeout,
        )
        
        latency_ms = (time.time() - start_time) * 1000
        content = json.loads(response.choices[0].message.content)
        
        return LLMResponse(
            content=content,
            model=self.model,
            tokens_used=response.usage.total_tokens if response.usage else 0,
            latency_ms=latency_ms,
        )


class AnthropicLLMClient(LLMClient):
    """Anthropic API client."""
    
    def __init__(self, model: str = "claude-3-haiku-20240307", api_key: Optional[str] = None):
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package required. Install with: pip install anthropic")
        
        self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        self.model = model
    
    def call_structured(self, prompt: str, schema: Dict[str, Any], timeout: Optional[float] = None) -> LLMResponse:
        """Call Anthropic with structured output."""
        start_time = time.time()
        
        # Anthropic doesn't have native JSON mode, so we'll parse manually
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        
        latency_ms = (time.time() - start_time) * 1000
        content_text = response.content[0].text
        
        # Try to extract JSON from response
        try:
            # Look for JSON in the response
            import re
            json_match = re.search(r'\{.*\}', content_text, re.DOTALL)
            if json_match:
                content = json.loads(json_match.group())
            else:
                content = {"error": "No JSON found in response", "raw": content_text}
        except json.JSONDecodeError:
            content = {"error": "Failed to parse JSON", "raw": content_text}
        
        return LLMResponse(
            content=content,
            model=self.model,
            tokens_used=response.usage.input_tokens + response.usage.output_tokens,
            latency_ms=latency_ms,
        )


class GrokLLMClient(LLMClient):
    """Grok (xAI) API client."""
    
    def __init__(self, model: str = "grok-beta", api_key: Optional[str] = None):
        try:
            import requests
        except ImportError:
            raise ImportError("requests package required. Install with: pip install requests")
        
        self.api_key = api_key or os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY")
        if not self.api_key:
            raise RuntimeError("Missing XAI_API_KEY or GROK_API_KEY environment variable")
        
        self.model = model
        self.base_url = "https://api.x.ai/v1"
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })
    
    def call_structured(self, prompt: str, schema: Dict[str, Any], timeout: Optional[float] = None) -> LLMResponse:
        """Call Grok API with structured output."""
        start_time = time.time()
        
        # Grok API format (similar to OpenAI)
        try:
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"},
                },
                timeout=timeout or 30
            )
            response.raise_for_status()
            data = response.json()
            
            latency_ms = (time.time() - start_time) * 1000
            content = json.loads(data["choices"][0]["message"]["content"])
            
            return LLMResponse(
                content=content,
                model=self.model,
                tokens_used=data.get("usage", {}).get("total_tokens", 0),
                latency_ms=latency_ms,
            )
        except Exception as e:
            # Graceful degradation
            latency_ms = (time.time() - start_time) * 1000
            return LLMResponse(
                content={"error": str(e), "raw": ""},
                model=self.model,
                tokens_used=0,
                latency_ms=latency_ms,
            )


def create_llm_client(provider: str = "openai", model: Optional[str] = None) -> LLMClient:
    """Factory function to create LLM client."""
    if provider.lower() == "openai":
        return OpenAILLMClient(model=model or "gpt-4o-mini")
    elif provider.lower() == "anthropic":
        return AnthropicLLMClient(model=model or "claude-3-haiku-20240307")
    elif provider.lower() == "grok" or provider.lower() == "xai":
        return GrokLLMClient(model=model or "grok-beta")
    else:
        raise ValueError(f"Unknown provider: {provider}. Supported: openai, anthropic, grok")

