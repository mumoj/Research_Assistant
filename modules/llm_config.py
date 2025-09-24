"""LLM configuration module for different providers."""
import os
from typing import Optional, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import BaseLanguageModel


class LLMConfig:
    """Configuration class for different LLM providers."""
    
    @staticmethod
    def get_primary_llm() -> BaseLanguageModel:
        """Get the primary LLM (Gemini) for answer generation."""
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
        
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest",
            temperature=0.2,
            max_tokens=1500,
            google_api_key=gemini_api_key
        )
    
    @staticmethod
    def get_validator_llm(
        provider: Optional[str] = None,
        model: Optional[str] = None
    ) -> BaseLanguageModel:
        """Get the validator LLM based on configuration."""
        provider = provider or os.getenv("VALIDATOR_PROVIDER", "openai")
        
        if provider.lower() == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")
            
            model_name = model or os.getenv("VALIDATOR_MODEL", "gpt-4o-mini")
            return ChatOpenAI(
                model=model_name,
                temperature=0.0,  # Deterministic for validation
                api_key=api_key
            )
        
        elif provider.lower() == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment")
            
            model_name = model or os.getenv("VALIDATOR_MODEL", "claude-haiku")
            return ChatAnthropic(
                model=model_name,
                temperature=0.0,
                anthropic_api_key=api_key
            )
        
        elif provider.lower() == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in environment")
            
            model_name = model or os.getenv("VALIDATOR_MODEL", "gemini-pro")
            return ChatGoogleGenerativeAI(
                model=model_name,
                temperature=0.0,
                google_api_key=api_key
            )
        
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    @staticmethod
    def get_available_providers() -> Dict[str, list]:
        """Get list of available providers and their models."""
        return {
            "openai": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
            "anthropic": ["claude-haiku", "claude-sonnet", "claude-opus"],
            "gemini": ["gemini-pro", "gemini-1.5-flash", "gemini-1.5-pro"]
        }
    
    @staticmethod
    def validate_config() -> Dict[str, bool]:
        """Validate API key configuration."""
        return {
            "gemini": bool(os.getenv("GEMINI_API_KEY")),
            "openai": bool(os.getenv("OPENAI_API_KEY")),
            "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
            "youtube": bool(os.getenv("YOUTUBE_API_KEY")),
            "serpapi": bool(os.getenv("SERPAPI_KEY"))
        }