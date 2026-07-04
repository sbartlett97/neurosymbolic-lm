"""LLM-based annotation using vLLM or transformers."""

import json
import re
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


class VLLMBackend(LLMBackend):
    """vLLM backend for high-throughput inference."""

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        quantization: Optional[str] = "awq",
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
            gpu_memory_utilization: GPU memory fraction to use
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

        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=["}\n\n", "}\n```", "```"],
        )

    def generate(self, prompts: List[str]) -> List[str]:
        """Generate responses for batch of prompts."""
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [output.outputs[0].text for output in outputs]

    def generate_single(self, prompt: str) -> str:
        """Generate response for single prompt."""
        return self.generate([prompt])[0]


class TransformersBackend(LLMBackend):
    """Transformers backend for local inference."""

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        device: str = "cuda",
        max_tokens: int = 2048,
        temperature: float = 0.1,
        load_in_4bit: bool = True,
    ):
        """
        Initialize transformers backend.

        Args:
            model_name: Model name or path
            device: Device to run on
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            load_in_4bit: Use 4-bit quantization
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

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=4096
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
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        backend: str = "vllm",
        quantization: Optional[str] = "awq",
        max_tokens: int = 2048,
        temperature: float = 0.1,
        max_retries: int = 3,
        should_respond_ratio: float = 0.7,
    ):
        """
        Initialize LLM annotator.

        Args:
            model_name: Model name or path
            backend: Backend to use ('vllm', 'transformers', 'mock')
            quantization: Quantization method
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            max_retries: Maximum retries for JSON parsing
            should_respond_ratio: Ratio of samples to generate QA for
        """
        self.model_name = model_name
        self.backend_name = backend
        self.max_retries = max_retries
        self.should_respond_ratio = should_respond_ratio

        # Initialize backend
        if backend == "vllm":
            self.backend = VLLMBackend(
                model_name=model_name,
                quantization=quantization,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        elif backend == "transformers":
            self.backend = TransformersBackend(
                model_name=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        elif backend == "mock":
            self.backend = MockBackend()
        else:
            raise ValueError(f"Unknown backend: {backend}")

        # Statistics
        self._stats = {
            "total": 0,
            "success": 0,
            "json_errors": 0,
            "retries": 0,
        }

    def _build_prompt(self, text: str, include_response: bool) -> str:
        """Build the full prompt with system message."""
        user_prompt = get_annotation_prompt(text, include_response)

        # Format for chat models
        if "llama" in self.model_name.lower() or "mistral" in self.model_name.lower():
            return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        else:
            # Generic format
            return f"{SYSTEM_PROMPT}\n\n{user_prompt}"

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
                response = self.backend.generate_single(prompt)
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
            responses = self.backend.generate(prompts)
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
