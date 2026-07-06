"""Collator for the decoder-only NeuroSymbolicCausalLM.

Subclasses CognitiveCollator, so everything about the *prompt* side —
entity/concept/relation encoding, gold char-span alignment, explicit
entity types, chat templating, per-message trace-annotation flattening —
is inherited unchanged. Only the *response* side differs:

- T5 needs shift-right decoder inputs starting from a decoder-start token;
  a causal LM consumes the response tokens as-is (the model packs them
  after the prompt and soft-node slots and returns logits pre-aligned to
  them).
- BPE tokenizers generally do not append EOS via ``add_special_tokens``
  (unlike T5's sentencepiece), so EOS is appended explicitly here — it is
  load-bearing: EOS-immediately is the abstention mechanism.

Batch keys are unchanged (``decoder_input_ids``/``decoder_labels``), so the
existing stage trainers work verbatim: ``decoder_input_ids`` are the
(unshifted) response tokens fed to the model as ``y_ids``, and
``decoder_labels`` align position-for-position with the model's returned
``logits``.
"""

from typing import List, Tuple

import torch

from .collator import CognitiveCollator


class CausalCognitiveCollator(CognitiveCollator):
    """CognitiveCollator variant producing causal-LM response targets."""

    def _causal_targets(
        self, responses: List[str], should_respond_mask: List[bool]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize responses with explicit EOS; abstention rows = [EOS].

        Returns (response_input_ids, labels) of identical shape:
        labels are the same tokens with padding replaced by -100.
        """
        eos_id = self.tokenizer.eos_token_id
        if eos_id is None:
            eos_id = 1
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = eos_id

        seqs: List[List[int]] = []
        for resp, should_respond in zip(responses, should_respond_mask):
            if should_respond:
                ids = self.tokenizer.encode(resp, add_special_tokens=False)
                ids = ids[: self.max_output_length - 1] + [eos_id]
            else:
                # Abstain: learn to emit EOS immediately
                ids = [eos_id]
            seqs.append(ids)

        width = max(len(s) for s in seqs) if seqs else 1
        input_ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
        labels = torch.full((len(seqs), width), -100, dtype=torch.long)
        for i, s in enumerate(seqs):
            t = torch.tensor(s, dtype=torch.long)
            input_ids[i, : len(s)] = t
            labels[i, : len(s)] = t

        return input_ids, labels

    def _process_responses(self, batch: List[dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        responses = []
        mask = []
        for x in batch:
            if x.get("should_respond", 0) == 1 and x.get("response", "").strip():
                responses.append(x["response"])
                mask.append(True)
            else:
                responses.append("")
                mask.append(False)
        return self._causal_targets(responses, mask)

    def _process_chat_responses(
        self, batch: List[dict], chat_targets: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        responses = []
        mask = []
        for sample, target in zip(batch, chat_targets):
            if sample.get("should_respond", 0) == 1 and target.strip():
                responses.append(target)
                mask.append(True)
            else:
                responses.append("")
                mask.append(False)
        return self._causal_targets(responses, mask)
