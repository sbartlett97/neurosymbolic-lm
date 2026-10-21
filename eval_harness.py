"""Standard LLM benchmarks for NeuroSymbolicCausalLM via lm-evaluation-harness.

The model is not a plain Hugging Face model: GNN node features enter as
gated soft tokens between the prompt and the response. This wrapper maps
the three lm-eval request types onto the model API:

  loglikelihood(context, continuation)
      forward(input_ids=context, y_ids=continuation) returns logits for
      the continuation only, with the soft nodes between the two
  loglikelihood_rolling(text)
      rolling windows over the text, each scored as above
  generate_until(context, until)
      model.generate(), cut at the first stop string

Usage (pip install "lm-eval==0.4.*"):

  # Our model, injection on
  python eval_harness.py --checkpoint ckpt/final_causal_model.pt \\
      --tasks mmlu,arc_easy,arc_challenge,hellaswag,piqa,winogrande \\
      --output-dir results/v1

  # Same checkpoint, injection off (the gate = 0 ablation)
  python eval_harness.py ... --no-soft-nodes --output-dir results/v1_nonodes

  # Baseline: the plain backbone, through lm-eval's own hf backend
  lm_eval --model hf --model_args pretrained=Qwen/Qwen3-0.6B-Base \\
      --tasks mmlu,... --batch_size 8 --output_path results/base
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.utils import get_rolling_token_windows, make_disjoint_window


@register_model("neurosym")
class NeuroSymLM(LM):
    """lm-eval adapter for a train_causal.py checkpoint.

    Pass ``checkpoint`` to load from disk, or ``model`` + ``tokenizer`` to
    wrap a model that is already in memory (tests, notebooks).
    """

    def __init__(self, checkpoint: Optional[str] = None, model=None,
                 tokenizer=None, device: str = "cuda", batch_size: int = 8,
                 use_soft_nodes: bool = True, dtype: str = "float32",
                 max_length: Optional[int] = None):
        super().__init__()
        if model is None:
            from train_causal import load_causal_checkpoint
            model, tokenizer, _ = load_causal_checkpoint(checkpoint, device)
        self.model = model.to(device=device, dtype=getattr(torch, dtype))
        self.model.eval()
        self.tokenizer = tokenizer
        self._device = device
        self.batch_size = int(batch_size)
        self.use_soft_nodes = bool(use_soft_nodes)
        # The prompt budget the model trained with; longer contexts lose
        # their start (as in the causal collator)
        self.max_length = int(max_length or model.max_input_length)
        self.eos_id = tokenizer.eos_token_id
        self.pad_id = (tokenizer.pad_token_id
                       if tokenizer.pad_token_id is not None else self.eos_id)

    # -- tokenization -------------------------------------------------

    def _encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _encode_pair(self, context: str,
                     continuation: str) -> Tuple[List[int], List[int]]:
        """Split as lm-eval's hf backend does: encode the joined text, so
        the continuation tokens match a natural tokenization."""
        n_space = len(context) - len(context.rstrip())
        if n_space:
            continuation = context[-n_space:] + continuation
            context = context[:-n_space]
        whole = self._encode(context + continuation)
        ctx = self._encode(context)
        return whole[:len(ctx)], whole[len(ctx):]

    # -- scoring ------------------------------------------------------

    def _pad(self, seqs: List[List[int]]):
        width = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), width), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), width), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
            mask[i, :len(s)] = 1
        return ids.to(self.device), mask.to(self.device)

    @torch.no_grad()
    def _score(self, pairs: List[Tuple[List[int], List[int]]]
               ) -> List[Tuple[float, bool]]:
        """Sum of log P(continuation | context, nodes), and whether greedy
        decoding gives the continuation, for token-level pairs."""
        prepared = []
        for ctx, cont in pairs:
            # The model needs at least one prompt token to read
            ctx = ctx or [self.eos_id]
            # Left-truncate: keep the context end next to the continuation
            prepared.append((ctx[-self.max_length:], cont))

        # Longest first: similar lengths share a batch, less padding
        order = sorted(range(len(prepared)),
                       key=lambda i: -(len(prepared[i][0]) + len(prepared[i][1])))
        results: List[Optional[Tuple[float, bool]]] = [None] * len(prepared)
        for start in range(0, len(order), self.batch_size):
            idx = order[start:start + self.batch_size]
            input_ids, attention_mask = self._pad([prepared[i][0] for i in idx])
            y_ids, y_mask = self._pad([prepared[i][1] for i in idx])
            out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                             y_ids=y_ids, use_soft_nodes=self.use_soft_nodes)
            logprobs = F.log_softmax(out["logits"].float(), dim=-1)
            token_lp = logprobs.gather(-1, y_ids.unsqueeze(-1)).squeeze(-1)
            greedy = logprobs.argmax(dim=-1) == y_ids
            for row, i in enumerate(idx):
                n = len(prepared[i][1])
                results[i] = (float(token_lp[row, :n].sum()),
                              bool(greedy[row, :n].all()))
        return results

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        pairs = [self._encode_pair(*req.args) for req in requests]
        return self._score(pairs)

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        out = []
        for req in requests:
            (text,) = req.args
            windows = [
                make_disjoint_window(w) for w in get_rolling_token_windows(
                    token_list=self._encode(text),
                    prefix_token=self.eos_id,
                    max_seq_len=self.max_length,
                    context_len=1,
                )
            ]
            out.append(sum(lp for lp, _ in self._score(windows)))
        return out

    # -- generation ---------------------------------------------------

    @torch.no_grad()
    def generate_until(self, requests, disable_tqdm: bool = False):
        out = []
        for req in requests:
            context, gen_kwargs = req.args
            kwargs = dict(gen_kwargs)
            until = kwargs.pop("until", None) or []
            if isinstance(until, str):
                until = [until]
            max_new = int(kwargs.pop("max_gen_toks", 256))
            # Greedy unless the task asks for sampling
            do_sample = bool(kwargs.pop("do_sample", False))
            temperature = kwargs.pop("temperature", 0.0)
            if not temperature:
                do_sample = False

            ctx = (self._encode(context) or [self.eos_id])[-self.max_length:]
            input_ids, attention_mask = self._pad([ctx])
            gen = self.model.generate(
                input_ids, attention_mask, max_length=max_new,
                use_soft_nodes=self.use_soft_nodes, do_sample=do_sample,
                **({"temperature": temperature} if do_sample else {}),
                eos_token_id=self.eos_id, pad_token_id=self.pad_id,
            )
            text = self.tokenizer.decode(gen[0], skip_special_tokens=True)
            for stop in until:
                if stop and stop in text:
                    text = text.split(stop, 1)[0]
            out.append(text)
        return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", required=True,
                   help="train_causal.py checkpoint (.pt)")
    p.add_argument("--tasks", required=True,
                   help="Comma-separated lm-eval task names")
    p.add_argument("--num-fewshot", type=int, default=None,
                   help="Few-shot examples (default: the task default)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float32",
                   choices=["float32", "bfloat16", "float16"])
    p.add_argument("--no-soft-nodes", action="store_true",
                   help="Ablation: do not inject the node tokens")
    p.add_argument("--limit", type=float, default=None,
                   help="Samples per task (a fraction if < 1), for smoke runs")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args(argv)

    from lm_eval import simple_evaluate

    lm = NeuroSymLM(checkpoint=args.checkpoint, device=args.device,
                    batch_size=args.batch_size, dtype=args.dtype,
                    use_soft_nodes=not args.no_soft_nodes)
    results = simple_evaluate(
        model=lm,
        tasks=args.tasks.split(","),
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        log_samples=False,
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results["config"]["neurosym"] = {
        "checkpoint": args.checkpoint,
        "use_soft_nodes": not args.no_soft_nodes,
        "dtype": args.dtype,
    }
    (out_dir / "results.json").write_text(
        json.dumps(results, indent=2, default=str))

    from lm_eval.utils import make_table
    print(make_table(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
