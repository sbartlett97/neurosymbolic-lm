import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from model.model import NSLM


def build_maps(dataset_path):
    ent_set = set()
    rel_set = set()
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            for e in obj.get('entities', []):
                ent_set.add(e.get('type'))
            for r in obj.get('relations', []):
                rel_set.add(r.get('type'))

    # reserve 0 for 'O' / 'no_relation'
    ent_list = ['O'] + sorted(x for x in ent_set if x is not None)
    rel_list = ['no_relation'] + sorted(x for x in rel_set if x is not None)

    ent_map = {t: i for i, t in enumerate(ent_list)}
    rel_map = {t: i for i, t in enumerate(rel_list)}
    return ent_map, rel_map


def charspan_to_token_indices(offsets, start, end):
    toks = []
    for i, (s, e) in enumerate(offsets):
        if s is None or e is None:
            continue
        if e <= start:
            continue
        if s >= end:
            break
        toks.append(i)
    return toks


class JsonlDataset(Dataset):
    def __init__(self, path, model: NSLM, ent_map, rel_map, max_input_len=256, max_output_len=64):
        self.lines = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.lines.append(json.loads(line))

        self.model = model
        self.tokenizer = model.tokenizer
        self.ent_map = ent_map
        self.rel_map = rel_map
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        obj = self.lines[idx]
        text = obj.get('text', '')
        targets = obj.get('target', '')

        # tokenize input with offsets
        tok = self.tokenizer(text, padding='max_length', truncation=True,
                             max_length=self.max_input_len, return_offsets_mapping=True)
        input_ids = torch.tensor(tok['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(tok['attention_mask'], dtype=torch.long)
        offsets = tok.get('offset_mapping')

        L = len(input_ids)

        ent_targets = torch.zeros(L, dtype=torch.long)
        rel_targets = torch.zeros((L, L), dtype=torch.long)

        # align entities
        for e in obj.get('entities', []):
            etext = e.get('text')
            etype = e.get('type')
            if not etext or etype not in self.ent_map:
                continue
            # find all occurrences of etext in text
            start_search = 0
            while True:
                found = text.find(etext, start_search)
                if found == -1:
                    break
                start = found
                end = found + len(etext)
                tok_idxs = charspan_to_token_indices(offsets, start, end)
                if tok_idxs:
                    for ti in tok_idxs:
                        ent_targets[ti] = self.ent_map[etype]
                start_search = end

        # align relations (head/tail are entity text strings)
        for r in obj.get('relations', []):
            htext = r.get('head')
            ttext = r.get('tail')
            rtype = r.get('type')
            if not htext or not ttext or rtype not in self.rel_map:
                continue

            # find first occurrence of head and tail
            h_found = text.find(htext)
            t_found = text.find(ttext)
            if h_found == -1 or t_found == -1:
                continue
            h_start, h_end = h_found, h_found + len(htext)
            t_start, t_end = t_found, t_found + len(ttext)
            h_idxs = charspan_to_token_indices(offsets, h_start, h_end)
            t_idxs = charspan_to_token_indices(offsets, t_start, t_end)
            if not h_idxs or not t_idxs:
                continue
            # pick first token index for head/tail
            hi = h_idxs[0]
            ti = t_idxs[0]
            rel_targets[hi, ti] = self.rel_map[rtype]

        # tokenize target for decoder labels
        labels_tok = self.tokenizer(targets, padding='max_length', truncation=True,
                                    max_length=self.max_output_len)
        labels = torch.tensor(labels_tok['input_ids'], dtype=torch.long)
        # replace pad token id with -100 for loss
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'ent_targets': ent_targets,
            'rel_targets': rel_targets
        }


def collate_fn(batch):
    # inputs are already fixed-length by tokenizer
    input_ids = torch.stack([b['input_ids'] for b in batch])
    attention_mask = torch.stack([b['attention_mask'] for b in batch])
    labels = torch.stack([b['labels'] for b in batch])
    ent_targets = torch.stack([b['ent_targets'] for b in batch])
    rel_targets = torch.stack([b['rel_targets'] for b in batch])
    return input_ids, attention_mask, labels, ent_targets, rel_targets


def train(dataset_path='dataset.jsonl', model_name='google/long-t5-tglobal-base',
          epochs=3, batch_size=4, lr=5e-5, device='cuda' if torch.cuda.is_available() else 'cpu'):

    ent_map, rel_map = build_maps(dataset_path)
    num_ent = len(ent_map)
    num_rel = len(rel_map)

    model = NSLM(model_name=model_name, num_ent_labels=num_ent, num_rel_labels=num_rel)

    # freeze encoder
    for p in model.backbone.encoder.parameters():
        p.requires_grad = False

    model.to(device)

    ds = JsonlDataset(dataset_path, model, ent_map, rel_map)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    ce_ent = nn.CrossEntropyLoss()
    ce_rel = nn.CrossEntropyLoss()
    ce_lm = nn.CrossEntropyLoss(ignore_index=-100)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for step, (input_ids, attention_mask, labels, ent_targets, rel_targets) in enumerate(dl):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            ent_targets = ent_targets.to(device)
            rel_targets = rel_targets.to(device)

            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            # LM loss
            lm_logits = out['lm_logits']
            B, T, V = lm_logits.size()
            lm_loss = ce_lm(lm_logits.view(B * T, V), labels.view(B * T))

            # entity loss
            ent_logits = out['entity_logits']  # (B, L, num_ent)
            B, L, C = ent_logits.size()
            ent_loss = ce_ent(ent_logits.view(B * L, C), ent_targets.view(B * L))

            # relation loss
            rel_logits = out['relation_logits']  # (B, L, L, num_rel)
            B, L1, L2, R = rel_logits.size()
            rel_loss = ce_rel(rel_logits.view(B * L1 * L2, R), rel_targets.view(B * L1 * L2))

            loss = lm_loss + ent_loss + rel_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if step % 20 == 0:
                print(f"Epoch {epoch} step {step} loss {loss.item():.4f}")

        avg = total_loss / (step + 1)
        print(f"Epoch {epoch} avg loss {avg:.4f}")
        # save checkpoint
        torch.save({'model_state': model.state_dict(), 'ent_map': ent_map, 'rel_map': rel_map},
                   f'model_epoch_{epoch}.pt')


if __name__ == '__main__':
    train()
