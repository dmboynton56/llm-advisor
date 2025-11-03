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


def create_llm_client(provider: str = "openai", model: Optional[str] = None) -> LLMClient:
    """Factory function to create LLM client."""
    if provider.lower() == "openai":
        return OpenAILLMClient(model=model or "gpt-4o-mini")
    elif provider.lower() == "anthropic":
        return AnthropicLLMClient(model=model or "claude-3-haiku-20240307")
    else:
        raise ValueError(f"Unknown provider: {provider}")

