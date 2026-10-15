from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np


class FrozenVerifierAdapter:
    """Inference-only adapter for the repository's legacy ``best_model.pt`` format."""

    def __init__(self, model, tokenizer, device, *, max_length: int = 256,
                 batch_size: int = 16, head: str = "fact", checkpoint_hash: str = "") -> None:
        if head not in {"fact", "g"}:
            raise ValueError("strict response collection currently supports head='fact' or 'g'")
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.head = head
        self.checkpoint_hash = checkpoint_hash
        self.model.requires_grad_(False)
        self.model.eval()

    @classmethod
    def from_checkpoint(cls, checkpoint_dir: str | Path, device: str = "cpu", *,
                        max_length: int = 256, batch_size: int = 16,
                        head: str = "fact") -> "FrozenVerifierAdapter":
        import torch
        from transformers import AutoTokenizer

        from proxcabi.model import ProxCABIModel
        from proxcabi.proxies import ProxyBuilder
        from .io import file_sha256

        directory = Path(checkpoint_dir)
        checkpoint_path = directory / "best_model.pt"
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        proxy = ProxyBuilder.load(directory / "proxy_builder.json")
        tokenizer_path = directory / "tokenizer"
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path if tokenizer_path.exists() else ckpt["backbone"], use_fast=True
        )
        model = ProxCABIModel(
            backbone_name=ckpt["backbone"], num_labels=3, z_buckets=proxy.z_buckets,
            w_buckets=proxy.w_buckets, **ckpt.get("model_config", {}),
        )
        model.load_state_dict(ckpt["model_state"])
        resolved_device = torch.device(device)
        model.to(resolved_device)
        digest = file_sha256(checkpoint_path)
        return cls(model, tokenizer, resolved_device, max_length=max_length,
                   batch_size=batch_size, head=head, checkpoint_hash=digest)

    def predict_proba(self, claims: list[str], evidences: list[str]) -> np.ndarray:
        import torch

        if len(claims) != len(evidences):
            raise ValueError("claims and evidences must have equal length")
        outputs = []
        self.model.eval()
        with torch.inference_mode():
            for start in range(0, len(claims), self.batch_size):
                encoded = self.tokenizer(
                    claims[start:start + self.batch_size], evidences[start:start + self.batch_size],
                    truncation=True, padding=True, max_length=self.max_length, return_tensors="pt",
                )
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)
                m = self.model.encode(input_ids, attention_mask)
                if self.head == "fact":
                    logits = self.model.fact_head(m)
                else:
                    # A causal audit must supply independent Z explicitly; silently deriving
                    # legacy text hashes is prohibited, so g is exposed only with neutral Z.
                    z = torch.zeros((len(input_ids),), dtype=torch.long, device=self.device)
                    logits = self.model.g_head(torch.cat([m, self.model.z_embeddings(z)], dim=-1))
                outputs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(outputs, axis=0) if outputs else np.empty((0, self.model.num_labels))

    def per_sample_response(self, claims: list[str], evidences: list[str], gold_labels: list[int],
                            kind: Literal["nll", "correct_prob"] = "nll") -> np.ndarray:
        probabilities = self.predict_proba(claims, evidences)
        labels = np.asarray(gold_labels, dtype=int)
        if len(labels) != len(probabilities):
            raise ValueError("gold_labels length does not match inputs")
        if np.any(labels < 0) or np.any(labels >= probabilities.shape[1]):
            raise ValueError("gold label outside model label range")
        correct = probabilities[np.arange(len(labels)), labels]
        if kind == "correct_prob":
            return correct
        if kind == "nll":
            return -np.log(np.maximum(correct, 1e-8))
        raise ValueError(f"unsupported response kind: {kind}")
