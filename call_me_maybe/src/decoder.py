"""Constrained decoder: vocabulary masking + greedy generation helpers."""

import json
import sys
from typing import Dict, List, Set

import numpy as np
from numpy.typing import NDArray

from llm_sdk import Small_LLM_Model
from src.build_prompts import build_prompt_for_argument
from src.models import Data, FunctionDef

# Practical negative-infinity used for additive logit masking
_NEG_INF: float = -1e9

# Maximum tokens to generate for any single value
_MAX_TOKENS: int = 100

# Maximum tokens allowed for a string value
_STRING_MAX: int = 128


# ── module-level singletons (initialised once at import time) ────────────────

model = Small_LLM_Model()
data = Data()


# ── Vocabulary ───────────────────────────────────────────────────────────────

class Vocabulary:
    """Wraps the LLM vocabulary and pre-computes logit masks.

    Attributes:
        token_to_id:    Full vocab mapping token-string → token-id.
        id_to_token:    Full vocab mapping token-id → token-string.
        size:           Total number of entries in the model's logit vector.
        fun_token_to_id: Restricted vocab for valid function-name tokens.
        fun_id_to_token: Reverse of fun_token_to_id.
        fun_size:       Number of entries in the function-name vocabulary.
        M_fun_name:     Additive mask [vocab_size]; 0.0 for valid fn-name
                        tokens, _NEG_INF for everything else.
        M_chars:        Additive mask [vocab_size]; 0.0 for printable tokens
                        (used during string generation), _NEG_INF otherwise.
    """

    def __init__(self) -> None:
        """Initialise vocabulary mappings and pre-compute both masks."""
        functions: List[FunctionDef] = data.functions_definition
        vocab_path: str = model.get_path_to_vocab_file()

        # Load the raw token→id mapping from the model's vocabulary file
        try:
            with open(vocab_path, "r", encoding="utf-8") as f:
                raw: Dict[str, int] = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            print(
                f"[ERROR] Cannot load vocabulary file '{vocab_path}': {exc}",
                file=sys.stderr,
            )
            sys.exit(1)

        self.token_to_id: Dict[str, int] = raw
        self.id_to_token: Dict[int, str] = {v: k for k, v in raw.items()}

        # Determine vocab size from a live logit probe
        logit_example: NDArray = np.asarray(
            model.get_logits_from_input_ids([1]), dtype=np.float32
        )
        self.size: int = len(logit_example)

        # Build the function-name-only vocabulary subset
        self.fun_token_to_id: Dict[str, int] = (
            self._build_fun_vocab(functions)
        )
        self.fun_id_to_token: Dict[int, str] = {
            v: k for k, v in self.fun_token_to_id.items()
        }
        self.fun_size: int = len(self.fun_token_to_id)

        # Pre-computed masks (computed once, reused on every decode step)
        self.M_fun_name: NDArray = self._build_fun_name_mask(functions)
        self.M_chars: NDArray = self._build_chars_mask()

    # ── mask builders ────────────────────────────────────────────────────────

    def _build_chars_mask(self) -> NDArray:
        """Build an additive mask allowing any printable-character token.

        Double-quote tokens are *not* blocked here; the generation loop is
        responsible for stopping when it encounters an unescaped quote.

        Returns:
            Float32 array of shape [vocab_size]: 0.0 = allowed,
            _NEG_INF = blocked.
        """
        mask: NDArray = np.full(
            self.size, _NEG_INF, dtype=np.float32
        )
        for token_str, token_id in self.token_to_id.items():
            clean: str = self._clean_token(token_str)
            # Allow any non-empty token whose characters are all printable
            if clean and clean.isprintable():
                mask[token_id] = 0.0
        return mask

    def _build_fun_vocab(
        self, functions: List[FunctionDef]
    ) -> Dict[str, int]:
        """Build a vocabulary restricted to substrings of function names.

        Tokens are accepted only if every character they contain also appears
        in at least one function name *and* the token is a substring of at
        least one function name.  This two-stage filter keeps the allowed set
        small while preserving all tokens needed for greedy name generation.

        Args:
            functions: The available functions whose names define the set.

        Returns:
            Dict mapping token-string → token-id for accepted tokens only.
        """
        function_names: List[str] = [fn.name for fn in functions]

        # Fast character-level pre-filter before the substring check
        allowed_chars: Set[str] = set("".join(function_names))

        fun_vocab: Dict[str, int] = {}
        for token_str, token_id in self.token_to_id.items():
            # Strip BPE leading-space marker (Ġ / space prefix)
            clean_token: str = token_str.lstrip(" ")

            # Reject tokens containing characters absent from all fn names
            if not all(ch in allowed_chars for ch in clean_token):
                continue

            # Accept only if the token is a substring of at least one name
            for fn_name in function_names:
                if clean_token in fn_name:
                    fun_vocab[token_str] = token_id
                    break

        return fun_vocab

    def _build_fun_name_mask(
        self, functions: List[FunctionDef]
    ) -> NDArray:
        """Build an additive logit mask for function-name token generation.

        Valid function-name tokens receive 0.0 (logit unchanged).
        All other tokens receive _NEG_INF (effectively blocked).

        The mask is added directly to raw logits before argmax:
            next_token_id = argmax(logits + M_fun_name)

        Args:
            functions: Functions whose names define the valid token set.

        Returns:
            Float32 array of shape [vocab_size]. ------------>
        """
        # Start with everything blocked
        mask: NDArray = np.full(
            self.size, _NEG_INF, dtype=np.float32
        )
        # Unblock only tokens that belong to the function-name vocabulary
        for token_id in self.fun_token_to_id.values():
            mask[token_id] = 0.0
        return mask

    # ── public helpers ───────────────────────────────────────────────────────

    def token_str(self, token_id: int) -> str:
        """Return the raw vocabulary string for *token_id*, or '' if unknown.

        Args:
            token_id: Integer token identifier.

        Returns:
            The corresponding token string, or an empty string.
        """
        return self.id_to_token.get(token_id, "")

    def ids_for_strings(self, strings: List[str]) -> Set[int]:
        """Return token ids whose string representation is in *strings*.

        Args:
            strings: Collection of token strings to look up.

        Returns:
            Set of integer token ids for strings present in the vocabulary.
        """
        return {
            self.token_to_id[s]
            for s in strings
            if s in self.token_to_id
        }

    # ── internal utility ─────────────────────────────────────────────────────

    @staticmethod
    def _clean_token(token_str: str) -> str:
        """Normalise BPE space markers so token strings compare cleanly.

        Args:
            token_str: Raw token string that may contain 'Ġ' markers.

        Returns:
            Token string with 'Ġ' replaced by a regular ASCII space.
        """
        return token_str.replace("Ġ", " ")


# Module-level vocabulary singleton (shared with Constrained_Decoder)
vocab = Vocabulary()


# ── Constrained_Decoder ──────────────────────────────────────────────────────

class Constrained_Decoder:
    """Greedy constrained decoder that wraps the Small_LLM_Model.

    All generation methods apply additive logit masks so the model can
    only produce tokens that are valid for the target JSON type.
    """

    def __init__(self) -> None:
        """Store a reference to the module-level model singleton."""
        self.model = model

    # ── public API ───────────────────────────────────────────────────────────

    def select_function_name(
        self,
        prompt: str,
        functions: List[FunctionDef],
    ) -> str:
        """Select the best-matching function name via constrained decoding.

        Generates one token at a time using the pre-computed M_fun_name mask,
        which blocks every token that is not part of a valid function name.
        Generation stops when a closing quote is produced or the accumulated
        name already matches one of the allowed function names exactly.

        Args:
            prompt:    The natural-language request string.
            functions: List of available functions to choose from.

        Returns:
            The selected function name as a plain Python string.
        """
        # Build one description line per function for the selection prompt
        function_descriptions: List[str] = [
            f"name: {fn.name} - description: {fn.description}\n"
            for fn in functions
        ]

        # Full prompt that ends with an open quote so the model completes it
        selection_prompt: str = (
            f"choose a function name from the following functions"
            f'\n\n{"".join(function_descriptions)}'
            f'\nfor the following prompt '
            f'"{prompt}"\nchosen name: "'
        )

        selected_name: str = ""
        allowed_fn_names: List[str] = [
            fn.name for fn in data.functions_definition
        ]

        # Token-by-token greedy generation with function-name mask
        while True:
            try:
                input_ids: List[int] = (
                    self.model.encode(selection_prompt)[0].tolist()
                )
                raw_logits = self.model.get_logits_from_input_ids(
                    input_ids
                )
            except Exception as exc:
                print(
                    f"[ERROR] Model inference error in "
                    f"select_function_name: {exc}",
                    file=sys.stderr,
                )
                return ""

            # Validate logit vector before masking
            logits: NDArray = np.asarray(raw_logits, dtype=np.float32)
            if logits.ndim != 1 or len(logits) != vocab.size:
                print(
                    "[ERROR] Unexpected logit shape in "
                    "select_function_name; aborting.",
                    file=sys.stderr,
                )
                return ""

            masked_logits: NDArray = logits + vocab.M_fun_name
            next_token_id: int = int(np.argmax(masked_logits))

            try:
                next_token: str = self.model.decode(next_token_id)
            except Exception as exc:
                print(
                    f"[ERROR] Decode error in select_function_name: {exc}",
                    file=sys.stderr,
                )
                return selected_name

            # Stop when closing quote emitted or name is already complete
            if (
                '"' in next_token
                or selected_name in allowed_fn_names
            ):
                break

            # Grow the running prompt and accumulated name
            selection_prompt += next_token
            selected_name += next_token

        return selected_name

    def select_arguments(
        self, prompt: str, function: FunctionDef
    ) -> Dict[str, object]:
        """Extract all arguments for *function* from *prompt*.

        Iterates over each declared parameter in declaration order,
        dispatches to the correct typed generator, and accumulates results.
        Errors in individual parameters are caught and logged; the remaining
        parameters are still attempted.

        Args:
            prompt:   The original natural-language request.
            function: The FunctionDef whose parameters are being filled.

        Returns:
            Dict mapping parameter name → extracted Python value.
        """
        already_extracted: Dict[str, object] = {}

        for name, param in function.parameters.items():
            param_type: str = param.type
            param_prompt: str = build_prompt_for_argument(
                prompt,
                function,
                name,
                param_type,
                already_extracted,
            )

            try:
                if param_type in ("number", "float"):
                    arg: object = self.generate_number(param_prompt)
                elif param_type == "boolean":
                    arg = self._generate_boolean(param_prompt)
                else:
                    # Default: treat as string
                    arg = self.generate_str(
                        param_prompt, name, prompt
                    )
            except Exception as exc:
                print(
                    f"[ERROR] Argument generation failed for "
                    f"'{name}': {exc}",
                    file=sys.stderr,
                )
                continue  # skip this parameter, try the next one

            already_extracted[name] = arg

        return already_extracted

    # ── typed value generators ────────────────────────────────────────────

    def generate_number(self, prompt: str) -> float:
        """Generate a JSON number value via constrained greedy decoding.

        Builds a boolean mask on each step that allows only tokens valid for
        the current numeric position (digits, sign, decimal point, exponent).
        Generation stops when a terminator character is decoded or the mask
        becomes all-False.

        Args:
            prompt: The full prompt ending just before the number value.

        Returns:
            The decoded float, or 0.0 on parse failure or empty output.
        """
        generated: str = ""

        try:
            input_ids: List[int] = (
                self.model.encode(prompt).tolist()[0]
            )
        except Exception as exc:
            print(
                f"[ERROR] Encoding failed in generate_number: {exc}",
                file=sys.stderr,
            )
            return 0.0

        for _ in range(_MAX_TOKENS):
            try:
                raw_logits = self.model.get_logits_from_input_ids(
                    input_ids
                )
            except Exception as exc:
                print(
                    f"[ERROR] Logit fetch failed in generate_number: {exc}",
                    file=sys.stderr,
                )
                break

            logits_len: int = len(raw_logits)
            logits: NDArray = np.asarray(raw_logits, dtype=np.float32)

            mask: NDArray = self._build_number_mask(
                generated, logits_len
            )

            # No valid next token: end generation
            if not mask.any():
                break

            logits = self._apply_mask(logits, mask)
            next_id: int = int(np.argmax(logits))
            next_str: str = self._clean_token(
                vocab.token_str(next_id)
            )

            # Terminator token signals end of number
            if next_str and next_str[0] in {
                " ", "\n", ",", "}", "]", "\t"
            }:
                break

            generated += next_str
            input_ids = input_ids + [next_id]

        try:
            return float(generated) if generated else 0.0
        except ValueError:
            return 0.0

    def generate_str(
        self,
        prompt: str,
        param_name: str,
        original_prompt: str,
    ) -> str:
        """Generate a JSON string value using the printable-chars mask.

        Uses vocab.M_chars to restrict generation to printable tokens.
        Stops when an unescaped closing double-quote is decoded.

        Args:
            prompt:          Full prompt ending with the opening quote.
            param_name:      Name of the parameter (kept for future context).
            original_prompt: The original natural-language request.

        Returns:
            The extracted string value without surrounding quotes.
        """
        p_prompt: str = prompt
        s_accum: str = ""

        for _ in range(_STRING_MAX):
            try:
                input_ids: List[int] = (
                    self.model.encode(p_prompt)[0].tolist()
                )
                raw_logits = self.model.get_logits_from_input_ids(
                    input_ids
                )
            except Exception as exc:
                print(
                    f"[ERROR] Model error in generate_str: {exc}",
                    file=sys.stderr,
                )
                break

            logits: NDArray = np.asarray(raw_logits, dtype=np.float32)
            masked_logits: NDArray = logits + vocab.M_chars
            s_tok_id: int = int(np.argmax(masked_logits))

            try:
                s_tok: str = self.model.decode(s_tok_id)
            except Exception as exc:
                print(
                    f"[ERROR] Decode error in generate_str: {exc}",
                    file=sys.stderr,
                )
                break

            # Unescaped closing quote signals end of the string value
            if '"' in s_tok and '\\"' not in s_tok:
                # Keep only the content that precedes the closing quote
                s_tok = s_tok.split('"', 1)[0] + '"'
                s_accum += s_tok.split('"', 1)[0]
                p_prompt += s_tok
                break

            p_prompt += s_tok
            s_accum += s_tok

            # Clean escaped-quote artefacts to avoid false-positive stops
            if '\\"' in s_accum:
                s_accum = s_accum.replace("\\", "")

        return s_accum

    def _generate_boolean(self, prompt: str) -> bool:
        """Generate a JSON boolean value via constrained greedy decoding.

        Applies a boolean mask that allows only 'true' / 'false' tokens
        (and common prefixes such as 't', 'tr', 'f', 'fa').  The returned
        value is determined by whether the top token starts with 't'.

        Args:
            prompt: Full prompt ending just before the boolean value.

        Returns:
            True if the decoded token starts with 't', False otherwise.
        """
        try:
            input_ids: List[int] = (
                self.model.encode(prompt).tolist()[0]
            )
            raw_logits = self.model.get_logits_from_input_ids(input_ids)
        except Exception as exc:
            print(
                f"[ERROR] Model error in _generate_boolean: {exc}",
                file=sys.stderr,
            )
            return False

        logits: NDArray = np.asarray(raw_logits, dtype=np.float32)
        mask: NDArray = self._build_boolean_mask(len(logits))

        if mask.any():
            logits = self._apply_mask(logits, mask)

        next_id: int = int(np.argmax(logits))
        token: str = (
            self._clean_token(vocab.token_str(next_id)).lower().strip()
        )
        return token.startswith("t")

    # ── mask builders ────────────────────────────────────────────────────────

    def _build_number_mask(
        self, already_generated: str, logits_len: int
    ) -> NDArray:
        """Build a boolean mask permitting only valid JSON number tokens.

        Allowed token categories depend on the current generation state:
        - Digits (0-9) are always allowed.
        - '-' is allowed only at position 0.
        - '.' is allowed once, and not at position 0.
        - 'e'/'E' is allowed once, and not at position 0.
        - Terminator characters (space, newline, comma, '}', ']', tab)
          signal end-of-number and are always allowed so the loop can exit.

        Args:
            already_generated: The numeric string accumulated so far.
            logits_len:        Length of the logit vector to mask.

        Returns:
            Boolean NDArray of shape [logits_len].
        """
        mask: NDArray = np.zeros(logits_len, dtype=bool)

        is_first: bool = already_generated == ""
        has_dot: bool = "." in already_generated
        has_exp: bool = "e" in already_generated.lower()

        terminator_chars = {" ", "\n", ",", "}", "]", "\t"}

        for token_id, token_str in vocab.id_to_token.items():
            s: str = self._clean_token(token_str)
            if not s:
                continue

            # Terminators are always permitted so decoding can halt
            if s[0] in terminator_chars:
                mask[token_id] = True
                continue

            # Unary minus is valid only as the very first character
            if s == "-" and is_first:
                mask[token_id] = True
                continue

            # Decimal point: only one, and not as the first character
            if s == "." and not has_dot and not is_first:
                mask[token_id] = True
                continue

            # Exponent marker: only one, and not as the first character
            if s.lower() == "e" and not has_exp and not is_first:
                mask[token_id] = True
                continue

            # Single-digit or all-digit tokens
            if s.isdigit() or all(c.isdigit() for c in s):
                mask[token_id] = True
                continue

            # Multi-character tokens that extend the number legitimately
            try:
                float(already_generated + s)
                mask[token_id] = True
            except ValueError:
                pass

        return mask

    def _build_boolean_mask(self, logit_len: int) -> NDArray:
        """Build a boolean mask permitting only 'true'/'false' tokens.

        Also allows common BPE prefixes ('t', 'tr', 'tru', 'f', 'fa',
        'fals') so the model can decode multi-token booleans.

        Args:
            logit_len: Length of the logit vector to mask.

        Returns:
            Boolean NDArray of shape [logit_len].
        """
        mask: NDArray = np.zeros(logit_len, dtype=bool)
        valid: Set[str] = {
            "true", "false", "tru", "fals", "tr", "fa", "t", "f"
        }
        for token_id, token_str in vocab.id_to_token.items():
            s: str = self._clean_token(token_str).lower()
            if s in valid:
                mask[token_id] = True
        return mask

    # ── shared utilities ─────────────────────────────────────────────────────

    @staticmethod
    def _apply_mask(
        logits: NDArray, mask: NDArray
    ) -> NDArray:
        """Set logits to _NEG_INF wherever *mask* is False.

        Args:
            logits: Float32 logit array to modify in-place.
            mask:   Boolean array; True = keep logit, False = block.

        Returns:
            The modified logits array (same object, mutated in-place).
        """
        logits[~mask] = _NEG_INF
        return logits

    @staticmethod
    def _clean_token(token_str: str) -> str:
        """Normalise BPE space markers in a token string.

        Args:
            token_str: Raw token string, possibly containing 'Ġ'.

        Returns:
            Token string with 'Ġ' replaced by a regular space.
        """
        return token_str.replace("Ġ", " ")
