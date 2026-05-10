import json
import numpy as np
from numpy.typing import NDArray
from typing import Dict, List


NEG_INF = -1e9


def clean_token(token_str: str) -> str:
    """Convert raw vocab token to readable string."""
    return (
        token_str
        .replace("Ġ", " ")
        .replace("Ċ", "\n")
        .replace("▁", " ")
        .replace("Ĳ", "ij")
    )


class VocabManager:
    """Loads the model vocabulary and builds logit masks for constrained decoding."""

    def __init__(self, model, function_names: List[str]) -> None:
        self.model = model
        self.function_names = function_names

        # Load raw vocab: token_str -> token_id
        path = model.get_path_to_vocab_file()
        with open(path, "r", encoding="utf-8") as f:
            raw: Dict[str, int] = json.load(f)

        self.token_to_id: Dict[str, int] = raw
        self.id_to_token: Dict[int, str] = {v: k for k, v in raw.items()}

        # Use logits size as ground truth — JSON vocab may have extra special tokens
        dummy_ids = self.model.encode("a")[0].tolist()
        self.model_vocab_size: int = len(self.model.get_logits_from_input_ids(dummy_ids))

        # Precompute static masks (built once, reused every call)
        self.M_numbers: NDArray = self._build_number_mask()
        self.M_chars: NDArray = self._build_chars_mask()

    # ── Static masks ────────────────────────────────────────────────────

    def _build_number_mask(self) -> NDArray:
        """Allow only tokens that are purely digits (0-9)."""
        mask = np.full(self.model_vocab_size, NEG_INF, dtype=np.float32)
        for token_str, token_id in self.token_to_id.items():
            if token_id >= self.model_vocab_size:
                continue
            cleaned = clean_token(token_str).strip()
            if cleaned.isdigit():
                mask[token_id] = 0.0
        return mask

    def _build_chars_mask(self) -> NDArray:
        """Allow any token that contains only printable characters."""
        mask = np.full(self.model_vocab_size, NEG_INF, dtype=np.float32)
        for token_str, token_id in self.token_to_id.items():
            if token_id >= self.model_vocab_size:
                continue
            cleaned = clean_token(token_str)
            if cleaned.isprintable() and cleaned != "":
                mask[token_id] = 0.0
        return mask

    # ── Dynamic mask (rebuilt every token step) ─────────────────────────

    def get_function_name_mask(self, already_generated: str) -> NDArray:
        """
        Build a mask that only allows tokens which can continue
        one of the valid function names given what was already generated.

        Args:
            already_generated: The function name characters generated so far.

        Returns:
            A float mask array — 0.0 for allowed tokens, NEG_INF for blocked.
        """
        mask = np.full(self.model_vocab_size, NEG_INF, dtype=np.float32)

        for token_str, token_id in self.token_to_id.items():
            if token_id >= self.model_vocab_size:
                continue
            cleaned = clean_token(token_str).lstrip(" ")
            if not cleaned:
                continue
            candidate = already_generated + cleaned
            for fn_name in self.function_names:
                if fn_name.startswith(candidate) or candidate.startswith(fn_name):
                    mask[token_id] = 0.0
                    break

        return mask