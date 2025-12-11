import torch
import torch.nn as nn
from transformers import LongT5ForConditionalGeneration, AutoTokenizer


class EntityRelationHead(nn.Module):
    def __init__(self, hidden, num_ent_labels, num_rel_labels):
        super().__init__()
        self.proj = nn.Linear(hidden, hidden)
        self.entity_logits = nn.Linear(hidden, num_ent_labels)
        self.relation_logits = nn.Bilinear(hidden, hidden, num_rel_labels)

    def forward(self, h, mask=None):
        h_proj = torch.relu(self.proj(h))
        ent_logits = self.entity_logits(h_proj)

        B, L, H = h_proj.size()
        h_i = h_proj.unsqueeze(2).expand(B, L, L, H)
        h_j = h_proj.unsqueeze(1).expand(B, L, L, H)
        rel_logits = self.relation_logits(h_i, h_j)

        return ent_logits, rel_logits


class SymbolicProjector(nn.Module):
    def __init__(self, hidden, ent_dim, rel_dim):
        super().__init__()
        self.ent_proj = nn.Linear(ent_dim, hidden)
        self.rel_proj = nn.Linear(rel_dim, hidden)

    def forward(self, h, ent_feats, rel_feats):
        h_aug = h + self.ent_proj(ent_feats) + self.rel_proj(rel_feats)
        return h_aug


class NSLM(nn.Module):
    def __init__(self, model_name, num_ent_labels=10, num_rel_labels=10):
        super().__init__()
        self.backbone = LongT5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        hidden = self.backbone.config.d_model
        self.relation = EntityRelationHead(hidden, num_ent_labels, num_rel_labels)
        self.symbolic_projector = SymbolicProjector(hidden, num_ent_labels, num_rel_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        enc_out = self.backbone.model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = enc_out.last_hidden_state

        ent_logits, rel_logits = self.relation(h, attention_mask)

        ent_feats = torch.softmax(ent_logits, dim=-1)
        rel_feats = torch.softmax(rel_logits, dim=-1)
        h_aug = self.symbolic_projector(h, ent_feats, rel_feats)

        dec_out = self.backbone.model.decoder(
            input_ids=labels if labels is not None else None,
            encoder_hidden_states=h_aug,
            encoder_attention_mask=attention_mask
        )
        lm_logits = self.backbone.lm_head(dec_out.last_hidden_state)

        return {
            "entity_logits": ent_logits,
            "relation_logits": rel_logits,
            "lm_logits": lm_logits,
            "encoder_hidden_state": h_aug,
        }

    def generate(self, input_ids, attention_mask=None, max_length=64):
        with torch.no_grad():
            enc = self.backbone.model.encoder(input_ids=input_ids, attention_mask=attention_mask)
            h = enc.last_hidden_state
            ent, rel = self.relation(h, attention_mask)
            h_aug = self.symbolic_projector(h, torch.softmax(ent, -1), torch.softmax(rel, -1))

            outputs = self.backbone.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                encoder_outputs=(h_aug,)
            )
        return outputs
