import time
import os
from dotenv import load_dotenv
import anthropic
import openai

load_dotenv()

# Cost per million tokens (input, output) in USD
COST_TABLE = {
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-sonnet-4-6":         (3.00, 15.00),
    "gpt-4o-mini":               (0.15,  0.60),
    "gpt-4o":                    (2.50, 10.00),
}

DEFAULT_MODELS = {
    "anthropic": "claude-haiku-4-5-20251001",
    "openai":    "gpt-4o-mini",
}

VALID_PROVIDERS = set(DEFAULT_MODELS)

MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0  # seconds


def _calc_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    if model not in COST_TABLE:
        return 0.0
    input_rate, output_rate = COST_TABLE[model]
    return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000


class LLMClient:
    def __init__(self, provider: str = "anthropic"):
        if provider not in VALID_PROVIDERS:
            raise ValueError(f"Invalid provider '{provider}'. Options: {sorted(VALID_PROVIDERS)}")

        self.provider = provider

        if provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise EnvironmentError("ANTHROPIC_API_KEY not found in environment.")
            self.client = anthropic.Anthropic(api_key=api_key)
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError("OPENAI_API_KEY not found in environment.")
            self.client = openai.OpenAI(api_key=api_key)

    def ask(
        self,
        question: str,
        context: str,
        system_prompt: str,
        model: str = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> dict:
        """Send a question with document context to the LLM.

        Returns:
            dict with keys: answer, input_tokens, output_tokens, cost_usd
        """
        model = model or DEFAULT_MODELS[self.provider]
        user_message = f"Document context:\n{context}\n\nQuestion: {question}"

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                print(f"[LLMClient] Attempt {attempt}/{MAX_RETRIES} — provider={self.provider}, model={model}")
                return self._call(system_prompt, user_message, model, temperature, max_tokens)

            except (anthropic.RateLimitError, openai.RateLimitError) as e:
                if attempt == MAX_RETRIES:
                    raise
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                print(f"[LLMClient] RateLimitError — retrying in {delay:.1f}s ({e})")
                time.sleep(delay)

            except (anthropic.APIConnectionError, openai.APIConnectionError) as e:
                if attempt == MAX_RETRIES:
                    raise
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                print(f"[LLMClient] APIConnectionError — retrying in {delay:.1f}s ({e})")
                time.sleep(delay)

            except (anthropic.AuthenticationError, openai.AuthenticationError):
                print("[LLMClient] AuthenticationError — invalid API key, aborting.")
                raise

        # Should never reach here
        raise RuntimeError("_ask: retry loop exhausted unexpectedly")

    def _call(
        self,
        system_prompt: str,
        user_message: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> dict:
        if self.provider == "anthropic":
            response = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            answer = next(b.text for b in response.content if b.type == "text")

        else:  # openai
            response = self.client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            )
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            answer = response.choices[0].message.content

        return {
            "answer": answer,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": _calc_cost(model, input_tokens, output_tokens),
        }


if __name__ == "__main__":
    SAMPLE_CONTEXT = (
        "Article 3: The employee will receive a monthly salary of $5,000 USD.\n"
        "Article 7: Vacation time is 15 business days per year worked.\n"
        "Article 12: The contract has a duration of 12 months from the date of signing."
    )
    SAMPLE_SYSTEM = (
        "You are an assistant specialized in analyzing legal and employment documents. "
        "Answer only based on the provided context. "
        "If the information is not in the context, state that clearly."
    )
    SAMPLE_QUESTION = "How many vacation days does the employee have?"

    print("=== LLMClient Demo ===\n")

    provider = "openai"
    print(f"Provider: {provider}")
    try:
        client = LLMClient(provider=provider)
        result = client.ask(
            question=SAMPLE_QUESTION,
            context=SAMPLE_CONTEXT,
            system_prompt=SAMPLE_SYSTEM,
        )
        print(f"Answer : {result['answer']}")
        print(f"Tokens : {result['input_tokens']} in / {result['output_tokens']} out")
        print(f"Cost   : ${result['cost_usd']:.6f} USD")
    except EnvironmentError as e:
        print(f"Skipping {provider}: {e}")
