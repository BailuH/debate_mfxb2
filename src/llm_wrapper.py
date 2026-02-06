"""LLM call wrapper with retry logic for rate limit errors."""
import asyncio
import logging
from typing import Any, Callable
from openai import RateLimitError

logger = logging.getLogger(__name__)


class LLMWrapper:
    """Wrapper for LLM calls with retry logic for rate limiting."""

    def __init__(
        self,
        max_retries: int = 5,
        initial_delay: float = 2.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
    ):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base

    async def ainvoke_with_retry(
        self,
        llm_call: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute LLM call with exponential backoff retry for rate limit errors.

        Args:
            llm_call: The async function to call (e.g., defendant.ainvoke)
            *args: Positional arguments for the LLM call
            **kwargs: Keyword arguments for the LLM call

        Returns:
            The result of the LLM call

        Raises:
            Exception: If all retries are exhausted
        """
        delay = self.initial_delay

        for attempt in range(self.max_retries):
            try:
                return await llm_call(*args, **kwargs)
            except RateLimitError as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Rate limit: Max retries ({self.max_retries}) exceeded")
                    raise

                # Calculate delay with exponential backoff
                wait_time = min(delay, self.max_delay)
                logger.warning(
                    f"Rate limit hit (attempt {attempt + 1}/{self.max_retries}). "
                    f"Retrying in {wait_time:.1f}s..."
                )

                await asyncio.sleep(wait_time)
                delay *= self.exponential_base

            except Exception as e:
                # Non-rate-limit errors: don't retry
                logger.error(f"Non-retryable error in LLM call: {e}")
                raise


# Global wrapper instance
llm_wrapper = LLMWrapper(
    max_retries=5,
    initial_delay=2.0,  # Start with 2 seconds
    max_delay=60.0,     # Max 60 seconds
    exponential_base=2.0,
)
