"""
LLM client wrapping OpenAI (default) 
"""
import threading


# gpt-4o-mini pricing, USD per 1M tokens (input, output).
GPT4O_MINI_RATE = (0.15, 0.60)


class LLMClient:
    """Minimal chat client. Routes by model-name prefix.

    `model.lower().startswith("claude")` -> Anthropic, else OpenAI.
    """

    def __init__(self, model: str = "gpt-4o-mini",
                 input_rate_per_m: float = GPT4O_MINI_RATE[0],
                 output_rate_per_m: float = GPT4O_MINI_RATE[1]):
        self.model = model
        self.input_rate_per_m = input_rate_per_m
        self.output_rate_per_m = output_rate_per_m
        self._lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self._openai = None
        self._anthropic = None

    # -- lazy SDK clients --------------------------------------------------
    def _openai_client(self):
        if self._openai is None:
            from openai import OpenAI
            self._openai = OpenAI()
        return self._openai

    def _anthropic_client(self):
        if self._anthropic is None:
            import anthropic
            self._anthropic = anthropic.Anthropic()
        return self._anthropic

    def _account(self, prompt_tokens: int, completion_tokens: int):
        with self._lock:
            self.prompt_tokens += int(prompt_tokens or 0)
            self.completion_tokens += int(completion_tokens or 0)

    # -- chat --------------------------------------------------------------
    def chat(self, messages, temperature: float = 0.0,
             max_tokens: int = 512) -> str:
        """Run one chat completion and return the assistant text.

        Anthropic requires a separate `system` argument: if `messages[0]` is a
        system message it is split out automatically.
        """
        if self.model.lower().startswith("claude"):
            return self._chat_anthropic(messages, temperature, max_tokens)
        return self._chat_openai(messages, temperature, max_tokens)

    def _chat_openai(self, messages, temperature, max_tokens) -> str:
        client = self._openai_client()
        m = self.model.lower()
        reasoning = m.startswith("gpt-5") or m.startswith("o1") or m.startswith("o3") or m.startswith("o4")
        if reasoning:
            # gpt-5 / o-series: omit temperature; reasoning tokens need a large budget.
            resp = client.chat.completions.create(
                model=self.model, messages=messages,
                max_completion_tokens=max(max_tokens, 8000))
        else:
            resp = client.chat.completions.create(
                model=self.model, messages=messages,
                temperature=temperature, max_completion_tokens=max_tokens)
        usage = getattr(resp, "usage", None)
        if usage is not None:
            self._account(getattr(usage, "prompt_tokens", 0),
                          getattr(usage, "completion_tokens", 0))
        return (resp.choices[0].message.content or "").strip()

    def _chat_anthropic(self, messages, temperature, max_tokens) -> str:
        client = self._anthropic_client()
        system = None
        chat_messages = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                chat_messages.append({"role": m["role"], "content": m["content"]})
        kwargs = dict(model=self.model, max_tokens=max_tokens,
                      temperature=temperature, messages=chat_messages)
        if system is not None:
            kwargs["system"] = system
        resp = client.messages.create(**kwargs)
        usage = getattr(resp, "usage", None)
        if usage is not None:
            self._account(getattr(usage, "input_tokens", 0),
                          getattr(usage, "output_tokens", 0))
        return resp.content[0].text if resp.content else ""

    # -- accounting --------------------------------------------------------
    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def estimated_cost_usd(self) -> float:
        return (self.prompt_tokens / 1e6 * self.input_rate_per_m
                + self.completion_tokens / 1e6 * self.output_rate_per_m)

    def usage_summary(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "input_rate_per_1m": self.input_rate_per_m,
            "output_rate_per_1m": self.output_rate_per_m,
        }
