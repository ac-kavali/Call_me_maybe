import json
from typing import Any, Dict, List

import numpy as np
from numpy.typing import NDArray


NEG_INF = -1e9
FLOAT_DTYPE = np.float32


def clean_token(token_str: str) -> str:
    """Convert a raw vocabulary token into a readable string."""
    return (
        token_str
        .replace("Ġ", " ")
        .replace("Ċ", "\n")
        .replace("▁", " ")
        .replace("Ĳ", "ij")
    )


class VocabManager:
    """
    Manage vocabulary loading and constrained decoding masks.

    This class loads the tokenizer vocabulary and builds reusable masks
    used during constrained decoding.
    """

    def __init__(
        self,
        model: Any,
        function_names: List[str],
    ) -> None:
        """
        Initialize the vocabulary manager.

        Args:
            model: Model object exposing tokenizer and logits methods.
            function_names: Allowed function names for decoding.

        Raises:
            ValueError: If inputs are invalid.
            RuntimeError: If vocab loading fails.
        """
        if model is None:
            raise ValueError("Model cannot be None.")

        if not isinstance(function_names, list):
            raise ValueError("function_names must be a list.")

        self.model = model
        self.function_names = function_names

        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

        self.model_vocab_size = self._get_model_vocab_size()

        self._load_vocab()

        # Precompute reusable masks once.
        self.M_numbers: NDArray[np.float32] = (
            self._build_number_mask()
        )
        self.M_chars: NDArray[np.float32] = (
            self._build_chars_mask()
        )

    def _load_vocab(self) -> None:
        """
        Load vocabulary mappings from the tokenizer file.

        Raises:
            RuntimeError: If vocab loading fails.
        """
        try:
            path = self.model.get_path_to_vocab_file()

            with open(path, "r", encoding="utf-8") as file:
                raw: Dict[str, int] = json.load(file)

            self.token_to_id = raw
            self.id_to_token = {
                token_id: token_str
                for token_str, token_id in raw.items()
            }

        except FileNotFoundError as exc:
            raise RuntimeError(
                "Vocabulary file was not found."
            ) from exc

        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Vocabulary file contains invalid JSON."
            ) from exc

        except OSError as exc:
            raise RuntimeError(
                "Failed to read vocabulary file."
            ) from exc

    def _get_model_vocab_size(self) -> int:
        """
        Get the real model vocabulary size from logits.

        Returns:
            Vocabulary size.

        Raises:
            RuntimeError: If vocab size retrieval fails.
        """
        try:
            dummy_ids = self.model.encode("a")[0].tolist()

            logits = self.model.get_logits_from_input_ids(
                dummy_ids
            )

            if not isinstance(logits, (list, np.ndarray)):
                raise TypeError("Logits must be array-like.")

            vocab_size = len(logits)

            if vocab_size <= 0:
                raise ValueError(
                    "Vocabulary size must be positive."
                )

            return vocab_size

        except Exception as exc:
            raise RuntimeError(
                "Failed to determine model vocabulary size."
            ) from exc

    def _create_empty_mask(self) -> NDArray[np.float32]:
        """
        Create a mask fully initialized with NEG_INF.

        Returns:
            Initialized mask array.
        """
        return np.full(
            self.model_vocab_size,
            NEG_INF,
            dtype=FLOAT_DTYPE,
        )

    def _is_valid_token_id(self, token_id: int) -> bool:
        """
        Check whether a token ID is inside vocab bounds.

        Args:
            token_id: Token identifier.

        Returns:
            True if valid, otherwise False.
        """
        return 0 <= token_id < self.model_vocab_size

    def _build_number_mask(self) -> NDArray[np.float32]:
        """
        Build a mask allowing only digit tokens.

        Returns:
            Number mask array.
        """
        mask = self._create_empty_mask()

        for token_str, token_id in self.token_to_id.items():
            if not self._is_valid_token_id(token_id):
                continue

            cleaned = clean_token(token_str).strip()

            if cleaned.isdigit():
                mask[token_id] = 0.0

        return mask

    def _build_chars_mask(self) -> NDArray[np.float32]:
        """
        Build a mask allowing printable character tokens.

        Returns:
            Character mask array.
        """
        mask = self._create_empty_mask()

        for token_str, token_id in self.token_to_id.items():
            if not self._is_valid_token_id(token_id):
                continue

            cleaned = clean_token(token_str)

            if cleaned and cleaned.isprintable():
                mask[token_id] = 0.0

        return mask

    def get_function_name_mask(
        self,
        already_generated: str,
    ) -> NDArray[np.float32]:
        """
        Build a mask for valid function-name continuation tokens.

        Only tokens that can continue one of the allowed function
        names remain available.

        Args:
            already_generated: Generated function name prefix.

        Returns:
            Function-name mask array.

        Raises:
            ValueError: If input is invalid.
        """
        if not isinstance(already_generated, str):
            raise ValueError(
                "already_generated must be a string."
            )

        mask = self._create_empty_mask()

        for token_str, token_id in self.token_to_id.items():
            if not self._is_valid_token_id(token_id):
                continue

            # Remove leading spaces added by tokenizers.
            cleaned = clean_token(token_str).lstrip(" ")

            if not cleaned:
                continue

            candidate = already_generated + cleaned

            for fn_name in self.function_names:
                is_valid = (
                    fn_name.startswith(candidate)
                    or candidate.startswith(fn_name)
                )

                if is_valid:
                    mask[token_id] = 0.0
                    break

        return mask
