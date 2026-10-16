"""LLM-based annotation using vLLM, transformers, or a hosted API."""

import json
import logging
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from .prompts import get_annotation_prompt, SYSTEM_PROMPT
from .source_loader import DocumentSource


@dataclass
class AnnotationResult:
    """Result of LLM annotation."""

    text: str
    entities: List[str] = field(default_factory=list)
    entity_spans: List[List[int]] = field(default_factory=list)
    concepts: List[List[str]] = field(default_factory=list)
    relations: List[List] = field(default_factory=list)
    entity_types: List[str] = field(default_factory=list)
    should_respond: int = 0
    response: str = ""
    success: bool = True
    error: Optional[str] = None
    raw_output: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for output."""
        return {
            "text": self.text,
            "entities": self.entities,
            "entity_spans": self.entity_spans,
            "concepts": self.concepts,
            "relations": self.relations,
            "entity_types": self.entity_types,
            "should_respond": self.should_respond,
            "response": self.response,
        }


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""

    @abstractmethod
    def generate(self, prompts: List[str]) -> List[str]:
        """Generate responses for a batch of prompts."""
        pass

    @abstractmethod
    def generate_single(self, prompt: str) -> str:
        """Generate response for a single prompt."""
        pass

    def generate_chat(
        self,
        prompt: str,
        system: Optional[str] = None,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a response to one user message.

        Backends with a tokenizer chat template wrap the message in that
        template. The default sends the plain prompt. API backends use
        ``json_schema`` for structured output. Local backends ignore it.
        """
        return self.generate_single(prompt)

    def generate_chat_batch(
        self,
        prompts: List[str],
        system: Optional[str] = None,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Generate responses to many user messages, in input order.

        The default calls :meth:`generate_chat` once per prompt. Backends
        override it to batch (vLLM) or to run requests in parallel (API).
        """
        return [self.generate_chat(p, system, json_schema) for p in prompts]


def apply_chat_template(tokenizer, prompt: str, system: Optional[str] = None) -> str:
    """Wrap a user message in the tokenizer chat template.

    Base models have no chat template. For them the system text (if any)
    goes before the prompt, and the prompt is otherwise unchanged.
    """
    if tokenizer is None or not getattr(tokenizer, "chat_template", None):
        return f"{system}\n\n{prompt}" if system else prompt
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


class VLLMBackend(LLMBackend):
    """vLLM backend for high-throughput inference."""

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        quantization: Optional[str] = None,
        tensor_parallel_size: int = 1,
        max_tokens: int = 2048,
        temperature: float = 0.1,
        gpu_memory_utilization: float = 0.9,
    ):
        """
        Initialize vLLM backend.

        Args:
            model_name: Model name or path
            quantization: Quantization method ('awq', 'gptq', None)
            tensor_parallel_size: Number of GPUs for tensor parallelism
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            gpu_memory_utilization: GPU memory fraction to use. Use a lower
                value (about 0.6) when another model shares the GPU.
        """
        self.model_name = model_name
        self.quantization = quantization
        self.max_tokens = max_tokens
        self.temperature = temperature

        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError("Please install vLLM: pip install vllm")

        # Initialize vLLM
        self.llm = LLM(
            model=model_name,
            quantization=quantization,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
        )

        # No stop strings: vLLM strips a matched stop string from the output,
        # and stops such as "}\n\n" cut the closing brace from the JSON.
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self._tokenizer = None

    def generate(self, prompts: List[str]) -> List[str]:
        """Generate responses for batch of prompts."""
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [output.outputs[0].text for output in outputs]

    def generate_single(self, prompt: str) -> str:
        """Generate response for single prompt."""
        return self.generate([prompt])[0]

    def generate_chat(
        self,
        prompt: str,
        system: Optional[str] = None,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a response with the model chat template applied."""
        return self.generate_chat_batch([prompt], system, json_schema)[0]

    def generate_chat_batch(
        self,
        prompts: List[str],
        system: Optional[str] = None,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Apply the chat template to each prompt, then make one vLLM call."""
        if self._tokenizer is None:
            self._tokenizer = self.llm.get_tokenizer()
        return self.generate(
            [apply_chat_template(self._tokenizer, p, system) for p in prompts]
        )


class TransformersBackend(LLMBackend):
    """Transformers backend for local inference."""

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        device: str = "cuda",
        max_tokens: int = 2048,
        temperature: float = 0.1,
        load_in_4bit: bool = False,
    ):
        """
        Initialize transformers backend.

        Args:
            model_name: Model name or path
            device: Device to run on
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            load_in_4bit: Use 4-bit quantization. This needs bitsandbytes,
                which is not a project dependency, so the default is False.
        """
        self.model_name = model_name
        self.device = device
        self.max_tokens = max_tokens
        self.temperature = temperature

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig

                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                )
            except ImportError:
                print("bitsandbytes not available, loading without quantization")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                )
        else:
            import torch
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )

        self.model.eval()

    def generate(self, prompts: List[str]) -> List[str]:
        """Generate responses for batch of prompts."""
        import torch

        results = []
        for prompt in prompts:
            results.append(self.generate_single(prompt))
        return results

    def generate_single(self, prompt: str) -> str:
        """Generate response for single prompt."""
        import torch

        # A chat-templated prompt already starts with the BOS token
        bos = self.tokenizer.bos_token
        add_special = not (bos and prompt.startswith(bos))
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
            add_special_tokens=add_special,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return generated

    def generate_chat(
        self,
        prompt: str,
        system: Optional[str] = None,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a response with the model chat template applied."""
        return self.generate_single(apply_chat_template(self.tokenizer, prompt, system))


# HTTP status codes worth a retry: rate limit, server errors, overload
RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504, 529})

logger = logging.getLogger(__name__)


def is_retryable_error(exc: BaseException) -> bool:
    """True for rate limit, overload, 5xx, timeout, and connection errors.

    The check uses duck typing, so it works for both SDKs and for test
    doubles without an SDK import.
    """
    if getattr(exc, "status_code", None) in RETRYABLE_STATUS:
        return True
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    # SDK timeout and connection errors carry no status code
    names = {cls.__name__ for cls in type(exc).__mro__}
    return bool(names & {"APITimeoutError", "APIConnectionError"})


def retryable_error_logged(backend, error) -> bool:
    """True after the first non-retryable error was logged."""
    if is_retryable_error(error):
        return True
    # Worker threads share the backend; the stats lock makes this one-time
    with backend._stats_lock:
        if getattr(backend, "_rejection_logged", False):
            return True
        backend._rejection_logged = True
        return False


class APIBackend(LLMBackend):
    """Shared logic for hosted API backends.

    Subclasses implement :meth:`_complete` (one request) and
    :meth:`_make_client`. This class adds a retry loop with exponential
    backoff and jitter, and a thread pool for batches. The API key comes
    only from an environment variable. It is never stored on the backend.
    """

    provider = "api"
    api_key_env = ""
    default_model = ""

    def __init__(
        self,
        model_name: Optional[str] = None,
        max_tokens: int = 1024,
        concurrency: int = 8,
        max_retries: int = 6,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        timeout: float = 120.0,
        json_schema: Optional[Dict[str, Any]] = None,
        effort: Optional[str] = None,
        client: Any = None,
        sleep=time.sleep,
    ):
        """
        Args:
            model_name: Provider model ID (None = provider default)
            max_tokens: Maximum output tokens per request
            concurrency: Number of requests in flight for a batch
            max_retries: Retries after the first try, for retryable errors
            base_delay: First backoff delay in seconds
            max_delay: Upper limit for one backoff delay in seconds
            timeout: Request timeout in seconds
            json_schema: Default JSON schema for structured output
            effort: Provider effort or reasoning level (None = provider default)
            client: Pre-built SDK client (dependency injection for tests)
            sleep: Sleep function (tests pass a no-op)
        """
        self.model_name = model_name or self.default_model
        self.max_tokens = max_tokens
        self.concurrency = max(1, concurrency)
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.timeout = timeout
        self.json_schema = json_schema
        self.effort = None if effort in (None, "default") else effort
        self._sleep = sleep
        self._stats = {"requests": 0, "retries": 0, "failures": 0}
        self._stats_lock = threading.Lock()
        if client is not None:
            self.client = client
        else:
            api_key = os.environ.get(self.api_key_env)
            if not api_key:
                raise RuntimeError(
                    f"Set {self.api_key_env} to use the {self.provider} backend"
                )
            # The SDK retries are off: this class owns the retry loop
            self.client = self._make_client(api_key)

    def __repr__(self) -> str:
        # No client and no key in the repr
        return (
            f"{type(self).__name__}(model_name={self.model_name!r}, "
            f"concurrency={self.concurrency})"
        )

    def _count(self, key: str) -> None:
        with self._stats_lock:
            self._stats[key] += 1

    def _make_client(self, api_key: str) -> Any:
        raise NotImplementedError

    def _complete(
        self, prompt: str, system: Optional[str], json_schema: Optional[Dict[str, Any]]
    ) -> str:
        raise NotImplementedError

    def _backoff(self, attempt: int) -> float:
        """Exponential backoff with jitter in the upper half of the window."""
        cap = min(self.max_delay, self.base_delay * (2 ** attempt))
        return random.uniform(cap / 2, cap)

    def _complete_with_retry(
        self, prompt: str, system: Optional[str], json_schema: Optional[Dict[str, Any]]
    ) -> str:
        """Run one request. Retry retryable errors. Return "" on failure.

        An empty string makes the caller's JSON parse fail, so the item
        counts as a failure without stopping the batch.
        """
        for attempt in range(self.max_retries + 1):
            self._count("requests")
            try:
                return self._complete(prompt, system, json_schema)
            except Exception as e:
                if is_retryable_error(e) and attempt < self.max_retries:
                    self._count("retries")
                    self._sleep(self._backoff(attempt))
                    continue
                self._count("failures")
                # Log the error class and status only. The message can echo
                # request data, and the key must never reach the logs.
                logger.warning(
                    "%s request failed: %s (status %s)",
                    self.provider,
                    type(e).__name__,
                    getattr(e, "status_code", None),
                )
                if not retryable_error_logged(self, e):
                    # One time: the first rejection text (e.g. a bad model
                    # name or schema). This is the API's validation message.
                    # The key is sent in a header, not echoed in it.
                    msg = str(getattr(e, "message", None) or e)[:300]
                    logger.warning("%s first rejection: %s", self.provider, msg)
                    print(f"{self.provider} request rejected: {msg}")
                return ""
        return ""

    def generate(self, prompts: List[str]) -> List[str]:
        """Send each prompt as one user message. Use the default schema."""
        return self.generate_chat_batch(prompts)

    def generate_single(self, prompt: str) -> str:
        return self.generate_chat(prompt)

    def generate_chat(
        self,
        prompt: str,
        system: Optional[str] = None,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> str:
        return self._complete_with_retry(prompt, system, json_schema or self.json_schema)

    def generate_chat_batch(
        self,
        prompts: List[str],
        system: Optional[str] = None,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Run the requests in a thread pool. Results keep input order."""
        if not prompts:
            return []
        schema = json_schema or self.json_schema
        workers = min(self.concurrency, len(prompts))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(
                pool.map(lambda p: self._complete_with_retry(p, system, schema), prompts)
            )

    def get_statistics(self) -> Dict[str, int]:
        with self._stats_lock:
            return dict(self._stats)


class AnthropicBackend(APIBackend):
    """Claude Messages API backend (official ``anthropic`` SDK).

    The request sends only parameters that every current Claude model
    accepts: model, max_tokens, messages, system, and
    ``output_config.format``. Newer models reject a non-default
    temperature, and Haiku 4.5 rejects ``effort``, so ``effort`` is sent
    only when the caller sets it.
    """

    provider = "anthropic"
    api_key_env = "ANTHROPIC_API_KEY"
    # QA generation is short, grounded, high-volume work. Haiku 4.5 is the
    # lowest-cost current Claude model and supports structured outputs.
    # Use --llm claude-sonnet-5-5 or claude-opus-5-5 for harder text.
    default_model = "claude-haiku-4-5"

    def _make_client(self, api_key: str) -> Any:
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "The anthropic backend needs the SDK: pip install anthropic"
            ) from e
        return anthropic.Anthropic(api_key=api_key, max_retries=0, timeout=self.timeout)

    def _complete(
        self, prompt: str, system: Optional[str], json_schema: Optional[Dict[str, Any]]
    ) -> str:
        kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        output_config: Dict[str, Any] = {}
        if json_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": json_schema}
        if self.effort:
            output_config["effort"] = self.effort
        if output_config:
            kwargs["output_config"] = output_config
        response = self.client.messages.create(**kwargs)
        # A refusal or a cut-off output does not match the schema
        if response.stop_reason in ("refusal", "max_tokens"):
            raise ValueError(f"stop_reason={response.stop_reason}")
        return "".join(b.text for b in response.content if b.type == "text")


class OpenAIBackend(APIBackend):
    """OpenAI Chat Completions backend (official ``openai`` SDK).

    Current OpenAI models are reasoning models. They take
    ``max_completion_tokens`` and ``reasoning_effort`` and do not take a
    custom temperature, so no temperature is sent.
    """

    provider = "openai"
    api_key_env = "OPENAI_API_KEY"
    # gpt-6-luna is the lowest-cost current GPT-6 model and OpenAI lists it
    # for focused, high-volume tasks. It supports structured outputs.
    default_model = "gpt-6-luna"
    # Low reasoning keeps the cost of hidden reasoning tokens small
    default_effort = "low"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.effort is None:
            self.effort = self.default_effort

    def _make_client(self, api_key: str) -> Any:
        try:
            import openai
        except ImportError as e:
            raise ImportError(
                "The openai backend needs the SDK: pip install openai"
            ) from e
        return openai.OpenAI(api_key=api_key, max_retries=0, timeout=self.timeout)

    def _complete(
        self, prompt: str, system: Optional[str], json_schema: Optional[Dict[str, Any]]
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_completion_tokens": self.max_tokens,
        }
        if self.effort:
            kwargs["reasoning_effort"] = self.effort
        if json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": json_schema, "strict": True},
            }
        else:
            kwargs["response_format"] = {"type": "json_object"}
        response = self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise ValueError("finish_reason=length")
        if getattr(choice.message, "refusal", None):
            raise ValueError("refusal")
        return choice.message.content or ""


API_BACKENDS = {"anthropic": AnthropicBackend, "openai": OpenAIBackend}


class MockBackend(LLMBackend):
    """Mock backend for testing without GPU."""

    def __init__(self):
        """Initialize mock backend."""
        pass

    def generate(self, prompts: List[str]) -> List[str]:
        """Generate mock responses."""
        return [self._mock_response(p) for p in prompts]

    def generate_single(self, prompt: str) -> str:
        """Generate mock response."""
        return self._mock_response(prompt)

    def _mock_response(self, prompt: str) -> str:
        """Generate a mock response for testing."""
        # Extract some text from the prompt to create plausible entities
        text_match = re.search(r"TEXT:\s*(.+?)(?:\n\n|Provide)", prompt, re.DOTALL)
        if text_match:
            text = text_match.group(1).strip()
            # Find capitalized words as mock entities
            words = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
            entities = list(set(words[:5]))

            spans = []
            for ent in entities:
                start = text.find(ent)
                if start >= 0:
                    spans.append([start, start + len(ent)])
                else:
                    spans.append([0, len(ent)])

            concepts = [["object"] for _ in entities]
            relations = []
            if len(entities) >= 2:
                relations = [[0, 1, "related_to"]]

            return json.dumps({
                "entities": entities,
                "entity_spans": spans,
                "concepts": concepts,
                "relations": relations,
                "should_respond": 1,
                "response": f"This text discusses {', '.join(entities[:3])}.",
            })

        return json.dumps({
            "entities": [],
            "entity_spans": [],
            "concepts": [],
            "relations": [],
            "should_respond": 0,
            "response": "",
        })


class LLMAnnotator:
    """Annotate documents using an LLM.

    Extracts:
    - Entities with character spans
    - Concepts for each entity
    - Relations between entities
    - Optional QA pairs
    """

    def __init__(
        self,
        model_name: Optional[str] = "meta-llama/Llama-3.1-8B-Instruct",
        backend: str = "vllm",
        quantization: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.1,
        max_retries: int = 3,
        should_respond_ratio: float = 0.7,
        gpu_memory_utilization: float = 0.9,
        api_concurrency: int = 8,
        api_effort: Optional[str] = None,
        json_schema: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize LLM annotator.

        Args:
            model_name: Model name or path
            backend: Backend to use ('vllm', 'transformers', 'openai',
                'anthropic', 'api', 'mock'). 'api' picks 'anthropic' for
                model names that start with 'claude', else 'openai'.
            quantization: Quantization method
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            max_retries: Maximum retries for JSON parsing
            should_respond_ratio: Ratio of samples to generate QA for
            gpu_memory_utilization: GPU memory fraction for vLLM
            api_concurrency: Requests in flight per batch (API backends)
            api_effort: Effort or reasoning level (API backends; None =
                provider default)
            json_schema: Default structured-output schema (API backends)
        """
        if backend == "api":
            is_claude = (model_name or "").startswith("claude")
            backend = "anthropic" if is_claude else "openai"
        self.backend_name = backend
        self.max_retries = max_retries
        self.should_respond_ratio = should_respond_ratio

        # Initialize backend
        if backend in API_BACKENDS:
            self.backend = API_BACKENDS[backend](
                model_name=model_name,
                max_tokens=max_tokens,
                concurrency=api_concurrency,
                json_schema=json_schema,
                effort=api_effort,
            )
            model_name = self.backend.model_name
        elif backend == "vllm":
            model_name = model_name or "meta-llama/Llama-3.1-8B-Instruct"
            self.backend = VLLMBackend(
                model_name=model_name,
                quantization=quantization,
                max_tokens=max_tokens,
                temperature=temperature,
                gpu_memory_utilization=gpu_memory_utilization,
            )
        elif backend == "transformers":
            model_name = model_name or "meta-llama/Llama-3.1-8B-Instruct"
            self.backend = TransformersBackend(
                model_name=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        elif backend == "mock":
            self.backend = MockBackend()
        else:
            raise ValueError(f"Unknown backend: {backend}")
        self.model_name = model_name

        # Statistics
        self._stats = {
            "total": 0,
            "success": 0,
            "json_errors": 0,
            "retries": 0,
        }

    def _build_prompt(self, text: str, include_response: bool) -> str:
        """Build the user message. The backend adds the system prompt.

        Each backend applies its own chat format in ``generate_chat``:
        the tokenizer chat template (vLLM, transformers) or the API
        message roles (OpenAI, Anthropic).
        """
        return get_annotation_prompt(text, include_response)

    def _parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from LLM response with recovery."""
        # Try direct parsing
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from response
        # Look for JSON object
        json_match = re.search(r"\{[\s\S]*\}", response)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Try to fix common issues
        cleaned = response.strip()

        # Remove markdown code blocks
        cleaned = re.sub(r"```json\s*", "", cleaned)
        cleaned = re.sub(r"```\s*$", "", cleaned)

        # Fix trailing commas
        cleaned = re.sub(r",\s*}", "}", cleaned)
        cleaned = re.sub(r",\s*]", "]", cleaned)

        # Try parsing again
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Final attempt: find balanced braces
        depth = 0
        start = None
        for i, c in enumerate(response):
            if c == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        return json.loads(response[start : i + 1])
                    except json.JSONDecodeError:
                        pass

        return None

    def annotate(
        self, doc: DocumentSource, include_response: Optional[bool] = None
    ) -> AnnotationResult:
        """
        Annotate a single document.

        Args:
            doc: Document to annotate
            include_response: Whether to generate QA (None = use ratio)

        Returns:
            AnnotationResult with extracted information
        """
        import random

        self._stats["total"] += 1

        # Determine whether to include response
        if include_response is None:
            include_response = random.random() < self.should_respond_ratio

        # Build prompt
        prompt = self._build_prompt(doc.text, include_response)

        # Try to get valid JSON
        for attempt in range(self.max_retries):
            try:
                response = self.backend.generate_chat(prompt, system=SYSTEM_PROMPT)
                parsed = self._parse_json_response(response)

                if parsed is not None:
                    self._stats["success"] += 1
                    return AnnotationResult(
                        text=doc.text,
                        entities=parsed.get("entities", []),
                        entity_spans=parsed.get("entity_spans", []),
                        concepts=parsed.get("concepts", []),
                        relations=parsed.get("relations", []),
                        should_respond=parsed.get("should_respond", 0 if not include_response else 1),
                        response=parsed.get("response", ""),
                        success=True,
                        raw_output=response,
                    )
                else:
                    self._stats["json_errors"] += 1
                    if attempt < self.max_retries - 1:
                        self._stats["retries"] += 1
                        continue

            except Exception as e:
                if attempt < self.max_retries - 1:
                    self._stats["retries"] += 1
                    continue
                return AnnotationResult(
                    text=doc.text,
                    success=False,
                    error=str(e),
                )

        # All retries failed
        return AnnotationResult(
            text=doc.text,
            success=False,
            error="Failed to parse JSON after retries",
            raw_output=response if "response" in locals() else None,
        )

    def annotate_batch(
        self, docs: List[DocumentSource], include_response: Optional[bool] = None
    ) -> List[AnnotationResult]:
        """
        Annotate a batch of documents.

        Args:
            docs: List of documents to annotate
            include_response: Whether to generate QA (None = use ratio per doc)

        Returns:
            List of AnnotationResult objects
        """
        import random

        # Build prompts
        prompts = []
        include_flags = []
        for doc in docs:
            if include_response is None:
                flag = random.random() < self.should_respond_ratio
            else:
                flag = include_response
            include_flags.append(flag)
            prompts.append(self._build_prompt(doc.text, flag))

        # Generate responses
        try:
            responses = self.backend.generate_chat_batch(prompts, system=SYSTEM_PROMPT)
        except Exception as e:
            # Fall back to individual generation
            return [self.annotate(doc, inc) for doc, inc in zip(docs, include_flags)]

        # Parse responses
        results = []
        for doc, response, include_flag in zip(docs, responses, include_flags):
            self._stats["total"] += 1
            parsed = self._parse_json_response(response)

            if parsed is not None:
                self._stats["success"] += 1
                results.append(
                    AnnotationResult(
                        text=doc.text,
                        entities=parsed.get("entities", []),
                        entity_spans=parsed.get("entity_spans", []),
                        concepts=parsed.get("concepts", []),
                        relations=parsed.get("relations", []),
                        should_respond=parsed.get("should_respond", 0 if not include_flag else 1),
                        response=parsed.get("response", ""),
                        success=True,
                        raw_output=response,
                    )
                )
            else:
                self._stats["json_errors"] += 1
                # Try single annotation with retries
                results.append(self.annotate(doc, include_flag))

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get annotation statistics."""
        total = self._stats["total"]
        return {
            **self._stats,
            "success_rate": self._stats["success"] / total if total > 0 else 0,
        }
