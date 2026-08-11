"""ESMC-based contact-map predictor — drop-in replacement for ESM2's contact head.

Wraps a ContactHead (single logistic regression over symmetrized/APC'd ESMC
attention maps, trained via the Rao-et-al. protocol) so that
`protein_init_with_embedding` can build its contact-map graph from ESMC instead
of loading ESM2-650M as a second model.

Produces the SAME artifact ESM2 did: an (L, L) contact-probability map in [0, 1]
that `contact_map()` thresholds into edges. Long sequences (> max_len) are handled
with overlapping windows (block-diagonal accumulation, mirroring esm_extract),
so the returned map is always full length L == len(seq).

Usage:
    predictor = ESMCContactPredictor(
        model_name="biohub/ESMC-600M",
        head_ckpt="esmc_600m_contact_head.pt",
        device="cuda:0")
    prob = predictor.predict(sequence)   # torch.FloatTensor (L, L), on CPU
"""
from __future__ import annotations

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Attention preprocessing — identical math to ESM2's ContactPredictionHead.    #
# --------------------------------------------------------------------------- #
def symmetrize(x: torch.Tensor) -> torch.Tensor:
    return x + x.transpose(-1, -2)


def apc(x: torch.Tensor) -> torch.Tensor:
    a1 = x.sum(-1, keepdim=True)
    a2 = x.sum(-2, keepdim=True)
    a12 = x.sum((-1, -2), keepdim=True)
    return x - (a1 * a2) / a12


class ContactHead(nn.Module):
    """(B, L, L, F) -> (B, L, L) contact logits. Single nn.Linear (same as ESM2)."""

    def __init__(self, in_features: int, use_apc: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.use_apc = use_apc
        self.regression = nn.Linear(in_features, 1)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.regression(feats).squeeze(-1)


class ESMCContactPredictor:
    """Load ESMC + a trained ContactHead and predict (L, L) contact maps."""

    def __init__(
        self,
        model_name: str = "biohub/ESMC-600M",
        head_ckpt: str = "esmc_600m_contact_head.pt",
        device: str = "cuda:0",
        max_len: int = 510,
    ):
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        if not torch.cuda.is_available() and str(device).startswith("cuda"):
            print("WARN: CUDA unavailable for ESMC contacts; falling back to CPU")
            device = "cpu"
        self.device = torch.device(device)
        self.max_len = max_len

        print(f"Loading ESMC contact model: {model_name}")
        # eager attention is REQUIRED for output_attentions
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_name, attn_implementation="eager", device_map={"": self.device}
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        ckpt = torch.load(head_ckpt, map_location="cpu")
        self.head = ContactHead(ckpt["in_features"], ckpt["use_apc"])
        self.head.load_state_dict(ckpt["state_dict"])
        self.head = self.head.to(self.device).eval()
        self.use_apc = bool(ckpt["use_apc"])
        cfg = self.model.config
        self.hidden_size = None
        for _attr in ("hidden_size", "d_model", "embed_dim", "hidden_dim", "n_embd", "dim"):
            if hasattr(cfg, _attr):
                self.hidden_size = int(getattr(cfg, _attr))
                break

        print(f"Loaded ContactHead: in_features={ckpt['in_features']}, "
              f"use_apc={self.use_apc}, hidden_size={self.hidden_size or 'auto'}, "
              )

    @torch.inference_mode()
    def _extract_features(self, sequence: str) -> torch.Tensor:
        """sequence (len l) -> (l, l, F) attention features on self.device."""
        inputs = self.tokenizer(sequence, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model(**inputs, output_attentions=True, output_hidden_states = True)

        assert out.attentions is not None, (
            "out.attentions is None — model must be loaded with attn_implementation='eager'"
        )
        assert out.hidden_states is not None, "output_hidden_states=True was not honored"
        
        emb = out.hidden_states[-1][0, 1: -1, :]

        # tuple(n_layers) of (1, n_heads, S, S), S = l + 2 (BOS/EOS)
        attn = torch.stack(out.attentions, dim=1)               # (1, n_layers, n_heads, S, S)
        B, n_layers, n_heads, S, _ = attn.shape
        attn = attn.reshape(B, n_layers * n_heads, S, S)        # (1, F, S, S)
        attn = attn[..., 1:-1, 1:-1]                            # (1, F, l, l) strip BOS/EOS
        attn = attn.float()
        attn = symmetrize(attn)
        if self.use_apc:
            attn = apc(attn)
        feats = attn.permute(0, 2, 3, 1).squeeze(0)             # (l, l, F)

        if feats.shape[-1] != self.head.in_features:
            raise ValueError(
                f"ESMC feature dim {feats.shape[-1]} != ContactHead.in_features "
                f"{self.head.in_features}. The model used at inference must match the "
                f"model the head was trained on."
                )

        return feats, emb

    @torch.inference_mode()
    def _head_prob(self, feats: torch.Tensor) -> torch.Tensor:
        """(l, l, F) -> (l, l) probabilities."""
        logits = self.head(feats.unsqueeze(0))[0]               # (l, l)
        return torch.sigmoid(logits)

    @torch.inference_mode()
    def _window(self, sequence: str):
        """One pass on a window (len l <= max_len) -> (emb (l,d), prob (l,l)) on CPU."""
        feats, emb = self._extract_features(sequence)
        prob = self._head_prob(feats)
        emb, prob = emb.float().cpu(), prob.cpu()
        assert emb.shape[0] == prob.shape[0] == len(sequence)
        return emb, prob

    @torch.inference_mode()
    def encode(self, sequence: str):
        """sequence (len L) -> (emb (L,d) fp32 CPU, prob (L,L) CPU).

        For L <= max_len: single pass. For L > max_len: overlapping windows
        (50% stride); embeddings averaged per residue, contacts accumulated
        block-diagonally and averaged. Contacts farther apart than max_len are
        not captured (same locality approximation ESM2's chunked path makes).
        """
        L = len(sequence)
        if L <= self.max_len:
            return self._window(sequence)

        W = self.max_len
        step = W // 2
        starts = list(range(0, L - W + 1, step))
        if not starts or starts[-1] != L - W:
            starts.append(L - W)

        emb_sum  = None
        emb_cnt  = torch.zeros(L, 1)
        prob_sum = torch.zeros(L, L)
        prob_cnt = torch.zeros(L, L)
        for s in starts:
            e = s + W
            w_emb, w_prob = self._window(sequence[s:e])
            if emb_sum is None:
                emb_sum = torch.zeros(L, w_emb.shape[1])
            emb_sum[s:e]        += w_emb
            emb_cnt[s:e]        += 1.0
            prob_sum[s:e, s:e]  += w_prob
            prob_cnt[s:e, s:e]  += 1.0
        return (emb_sum / emb_cnt.clamp(min=1.0),
                prob_sum / prob_cnt.clamp(min=1.0))

    @torch.inference_mode()
    def predict(self, sequence: str) -> torch.Tensor:
        """Contacts only — kept so existing contact_source='esmc' callers work unchanged."""
        return self.encode(sequence)[1]
