import numpy as np
import json
from src.models import Data

from llm_sdk import Small_LLM_Model
from typing import List, Set, Dict
from numpy.typing import NDArray
from src.models import FunctionDef
from src.build_prompts import build_prompt_for_argument


_NEG_INF = -1e9          # practical negative-infinity for logit masking
_MAX_TOKENS = 100         # maximum tokens to generate for any single value
_STRING_MAX = 128        # maximum tokens for a string value


# The llm Main class
model = Small_LLM_Model()
data = Data()

class Vocabulary:    #The class that controls the llm vocabulary
    def __init__(self) -> None:
        """
        Initialize vocabulary and precompute the function name mask.

        """
        functions: list[FunctionDef] = data.functions_definition
        vocab_path: str = model.get_path_to_vocab_file()
        with open(vocab_path, "r", encoding="utf-8") as f:
            raw: Dict[str, int] = json.load(f)

        # Full vocabulary mappings
        self.token_to_id: Dict[str, int] = raw
        self.id_to_token: Dict[int, str] = {v: k for k, v in raw.items()}
        logit_example = model.get_logits_from_input_ids([1])
        self.size: int = len(logit_example)

        # Function-name-only vocabulary (tokens that appear in any function name)
        self.fun_token_to_id: Dict[str, int] = self._build_fun_vocab(functions)
        self.fun_id_to_token: Dict[int, str] = {
            v: k for k, v in self.fun_token_to_id.items()
        }
        self.fun_size: int = len(self.fun_token_to_id)

        # Precomputed additive mask for function name generation
        # Shape: [vocab_size] — 0.0 for valid tokens, -inf for invalid ones
        self.M_fun_name: NDArray = self._build_fun_name_mask(functions)

    def _build_fun_vocab(self, functions: List[FunctionDef]) -> Dict[str, int]:
        """
        Build a restricted vocabulary of tokens that are valid
        substrings of at least one allowed function name.

        Args:
            functions: List of Functions whose names define the allowed tokens.

        Returns:
            Dict mapping token_string -> token_id for valid tokens only.
        """
        function_names: List[str] = [fn.name for fn in functions]

        # Fast pre-filter: only consider tokens whose characters
        # are all present in the union of function name characters
        allowed_chars: set = set("".join(function_names))

        fun_vocab: Dict[str, int] = {}

        for token_str, token_id in self.token_to_id.items():
            # Strip the BPE leading space marker (Ġ or space prefix)
            clean_token: str = token_str.lstrip(" ")

            # Skip tokens with characters not in any function name
            if not all(ch in allowed_chars for ch in clean_token):
                continue

            # Accept the token if it is a substring of at least one function name
            for fn_name in function_names:
                if clean_token in fn_name:
                    fun_vocab[token_str] = token_id
                    break

        return fun_vocab

    def _build_fun_name_mask(self, functions: List[FunctionDef]) -> NDArray:
        """
        Build an additive logit mask of shape [vocab_size].

        Valid function name tokens get 0.0 (no change to logits).
        All other tokens get -inf (effectively blocked).

        The mask is added directly to raw logits before argmax:
            next_token_id = argmax(logits + M_fun_name)

        Args:
            functions: List of Functions whose names define the valid tokens.

        Returns:
            ndarray of shape [vocab_size] with 0.0 or -inf values.
        """
        # Start with everything blocked
        mask: NDArray = np.full(self.size, _NEG_INF, dtype=np.float32) #------------------------>

        # Unblock only tokens present in fun_token_to_id
        for token_id in self.fun_token_to_id.values():
            mask[token_id] = 0.0

        return mask

    def token_str(self, token_id: int) -> str:
        """Return the raw string for a token id (empty string if unknown)."""
        return self.id_to_token.get(token_id, "")

    def ids_for_strings(self, strings: List[str]) -> Set[int]:
        """Return token ids whose string is in *strings*."""
        return {self.token_to_id[s] for s in strings if s in self.token_to_id}


vocab = Vocabulary()

class Constrained_Decoder:
    def __init__(self):
        self.model = model

    def select_function_name (
            self,
            prompt: str,
            functions: List[FunctionDef],
    ) -> str:
        """
        Select the best function name for a given prompt using constrained decoding.

        Uses the precomputed function name mask (M_fun_name) to restrict token
        selection to valid function name tokens only, generating the name
        character by character until a closing quote is produced.

        Args:
            prompt: The Prompts object containing the natural language request.
            functions: List of available Functions to choose from.

        Returns:
            The selected function name as a string.
        """
        # Build the list of function descriptions for the selection prompt
        function_descriptions: List[str] = [
            f"name: {fn.name} - description: {fn.description}\n"
            for fn in functions
        ]

        # Build the full selection prompt shown to the model
        selection_prompt: str = (
            f'choose a function name from the following functions'
            f'\n\n{"".join(function_descriptions)}'
            f'\nfor the following prompt '
            f'"{prompt}"\nchosen name: "'
        )

        selected_name: str = ""

        # Generate the function name token by token using constrained decoding
        while True:
            input_ids: List[int] = model.encode(selection_prompt)[0].tolist()

            # Add M_fun_name mask to restrict logits to valid function name tokens
            masked_logits = model.get_logits_from_input_ids(input_ids) + vocab.M_fun_name

            next_token_id = np.argmax(masked_logits)
            next_token: str = model.decode(next_token_id)
            allowed_functions = [fn.name for fn in data.functions_definition]
            # Stop when the model generates a closing quote
            if '"' in next_token or selected_name in allowed_functions:
                break

            # Append the token to both the running prompt and the name
            selection_prompt += next_token
            selected_name += next_token

        return selected_name


    def select_arguments (self, prompt, function: FunctionDef):
        already_extracted: Dict = {}
        for name, param in function.parameters.items():
            param_type = param.type
            param_prompt = build_prompt_for_argument(prompt, function, name, param_type, already_extracted)

            try:
                if param_type in ("number", "float"):
                    arg = self.generate_number(param_prompt)
                elif param_type == "boolean":
                    arg = self._generate_boolean(prompt)
                else:
                    arg = self.generate_str(prompt)
                    print(prompt)

                already_extracted[name] = arg

            except Exception as e:
                print(e)

        return already_extracted



    def generate_number(self, prompt):
        """
        Generate a JSON number value.

        Args:
            prompt: Prompt ending right before the number value.

        Returns:
            The extracted float value.
        """
        generated = ""
        input_ids = self.model.encode(prompt).tolist()[0]
        for _ in range(_MAX_TOKENS):
            logits = model.get_logits_from_input_ids(input_ids)
            logits_len = len(logits)
            logits = np.array(logits, dtype=np.float32)  # Convert logits to np.NDArray
            mask = self._build_number_mask(generated, logits_len)

            if not mask.any():
                break

            logits = self._apply_mask(logits, mask)
            next_id = int(np.argmax(logits))
            next_str = self._clean_token(vocab.token_str(next_id))

            # A terminator token signals end of number
            if next_str and next_str[0] in {" ", "\n", ",", "}", "]", "\t"}:
                break

            generated += next_str
            input_ids = input_ids + [next_id]

        try:
            return float(generated) if generated else 0.0
        except ValueError:
            return 0.0

    def _build_prefix_mask (self, allowed_functions, already_generated, mask) -> NDArray:
        # List initialized of False used to be modified to make just the allowed function names True

        # Loop over all tokens and check if this can be a part of at least one allowed function
        for token_str, token_id in vocab.fun_token_to_id.items():
            print("already generated: ", already_generated)
            cleaned_token = self._clean_token(token_str)
            candidate = already_generated + cleaned_token
            print("one_session--------------------------------")
            for function_name in allowed_functions:
                if function_name.startswith(candidate):
                    mask[token_id] = True
                    print("this token selected: ", vocab.id_to_token[token_id])

        return mask

    def _apply_mask (self, logits: NDArray, mask: NDArray) -> NDArray:
        logits[~mask] = _NEG_INF
        return logits

    def generate_str(self, prompt):
        """
                Generate a JSON string value (without surrounding quotes).

                Args:
                    prompt: Prompt ending right before the opening quote of the value.

                Returns:
                    The extracted string (without quotes).
                """
        # We inject the opening quote in the pipeline; here we just collect content
        generated = ""
        # The prompt already ends with ': ' so we add the opening quote
        full_prompt = prompt + '"'
        input_ids = self.model.encode(full_prompt).tolist()[0]

        for _ in range(_STRING_MAX):
            logits = model.get_logits_from_input_ids(input_ids)
            logits = np.array(logits, dtype=np.float32)  # Convert logits to np.NDArray
            next_id = int(np.argmax(logits))
            next_str = self._clean_token(vocab.token_str(next_id))

            # Stop at closing double quote
            if '"' in next_str:
                # Take content before the quote
                generated += next_str.split('"')[0]
                break

            generated += next_str
            input_ids = input_ids + [next_id]

        return generated.strip()


    def _generate_boolean(self, prompt):
        """
                Generate a JSON boolean value.

                Args:
                    prompt: Prompt ending right before the boolean value.

                Returns:
                    True or False.
                """
        input_ids = self.model.encode(prompt).tolist()[0]
        logits = model.get_logits_from_input_ids(input_ids)
        logits = np.array(logits, dtype=np.float32)  # Convert logits to np.NDArray
        mask = self._build_boolean_mask(len(logits))

        if mask.any():
            logits = self._apply_mask(logits, mask)

        next_id = int(np.argmax(logits))
        token = self._clean_token(vocab.token_str(next_id)).lower().strip()
        return token.startswith("t")

    def _build_number_mask (self, already_generated: str, logits_len) -> np.ndarray:
        """
        Keep only tokens that can appear in a JSON number.

        Allowed characters: digits 0-9, '-' (only at start), '.', 'e', 'E', '+'.
        We also allow a closing token (space / newline / comma / '}') to end the number.
        """
        mask = np.zeros(logits_len, dtype=bool)

        is_first = already_generated == ""
        has_dot = "." in already_generated
        has_exp = "e" in already_generated.lower()

        terminator_chars = {" ", "\n", ",", "}", "]", "\t"}

        for token_id, token_str in vocab.id_to_token.items():
            s = self._clean_token(token_str)
            if not s:
                continue

            # Allow terminators so the loop can stop
            if s[0] in terminator_chars:
                mask[token_id] = True
                continue

            # Leading minus allowed only at start
            if s == "-" and is_first:
                mask[token_id] = True
                continue

            # Decimal point — only one allowed, not at start
            if s == "." and not has_dot and not is_first:
                mask[token_id] = True
                continue

            # Exponent
            if s.lower() == "e" and not has_exp and not is_first:
                mask[token_id] = True
                continue

            # Pure digit tokens
            if s.isdigit() or all(c.isdigit() for c in s):
                mask[token_id] = True
                continue

            # Tokens that are entirely numeric (e.g. "42", "3.14")
            try:
                float(already_generated + s)
                mask[token_id] = True
            except ValueError:
                pass

        return mask

    def _build_boolean_mask (self, logit_len) -> np.ndarray:
        """Keep only 'true' and 'false' tokens."""
        mask = np.zeros(logit_len, dtype=bool)
        for token_id, token_str in vocab.id_to_token.items():
            s = self._clean_token(token_str).lower()
            if s in ("true", "false", "tru", "fals", "tr", "fa", "t", "f"):
                mask[token_id] = True
        return mask


    def _clean_token(self, token_str: str) -> str:
        return (
            token_str
            .replace("Ġ", " ")
        )





