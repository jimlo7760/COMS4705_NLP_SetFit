"""all-MiniLM-L6-v2 as a trainable sentence encoder.

Same computation as the sentence-transformers pipeline for this checkpoint:
Transformer -> attention-masked mean pooling -> L2 normalisation (384-d), but written
directly against Hugging Face transformers so CE and SetFit can backpropagate through it.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from . import config as C
from .utils import get_device


class SentenceEncoder(nn.Module):
    def __init__(self, model_id: str = C.MODEL_ID, max_length: int = C.MAX_LENGTH,
                 device=None, use_amp: bool = False):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer  # only needed for real runs

        self.device = torch.device(device) if device is not None else get_device()
        self.use_amp = bool(use_amp) and self.device.type == "cuda"
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        try:  # the checkpoint has no pooler weights; mean pooling does not use one
            self.model = AutoModel.from_pretrained(model_id, add_pooling_layer=False)
        except TypeError:
            self.model = AutoModel.from_pretrained(model_id)
        self.model.to(self.device)
        self.dim = int(self.model.config.hidden_size)
        self.commit = getattr(self.model.config, "_commit_hash", None)

    def _tokenize(self, texts):
        batch = self.tokenizer(list(texts), padding=True, truncation=True,
                               max_length=self.max_length, return_tensors="pt")
        return {k: v.to(self.device) for k, v in batch.items()}

    def forward(self, texts) -> torch.Tensor:
        """Unit-normalised float32 embeddings (B, dim); differentiable."""
        batch = self._tokenize(texts)
        with torch.autocast(device_type=self.device.type, dtype=torch.float16,
                            enabled=self.use_amp):
            hidden = self.model(**batch).last_hidden_state
        hidden = hidden.float()
        mask = batch["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return F.normalize(pooled, p=2, dim=-1)

    @torch.no_grad()
    def encode(self, texts, batch_size: int = C.ENCODE_BATCH_SIZE) -> np.ndarray:
        """Frozen-mode embeddings as a float32 numpy array (eval mode, no dropout)."""
        was_training = self.training
        self.eval()
        texts = list(texts)
        out = [self.forward(texts[i:i + batch_size]).cpu() for i in range(0, len(texts), batch_size)]
        self.train(was_training)
        return torch.cat(out).numpy().astype(np.float32)

    def truncation_rate(self, texts) -> float:
        """Share of texts longer than max_length tokens (special tokens included)."""
        ids = self.tokenizer(list(texts), truncation=False)["input_ids"]
        return float(np.mean([len(x) > self.max_length for x in ids]))
