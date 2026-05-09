import numpy as np
import json
from src.build_prompts import build_prompt_for_argument
from numpy.typing import NDArray

from llm_sdk import Small_LLM_Model
from typing import List, Set, Dict
from numpy.typing import NDArray

from src.models import FunctionDef

# The llm Main class
model = Small_LLM_Model()

class Vocabulary:    #The class that controls the llm vocabulary
    """Loads and indexes the model vocabulary once."""
    def __init__(self) -> None:
        path = model.get_path_to_vocab_file()
        with open(path, "r", encoding="utf-8") as f:
            raw: Dict[str, int] = json.load(f)
        # id → token_string
        self.id_to_token: Dict[int, str] = {v: k for k, v in raw.items()}
        # token_string → id
        self.token_to_id: Dict[str, int] = raw
        self.size: int = len(raw)

    def token_str(self, token_id: int) -> str:
        """Return the raw string for a token id (empty string if unknown)."""
        return self.id_to_token.get(token_id, "")

    def ids_for_strings(self, strings: List[str]) -> Set[int]:
        """Return token ids whose string is in *strings*."""
        return {self.token_to_id[s] for s in strings if s in self.token_to_id}

_NEG_INF = -1e9          # practical negative-infinity for logit masking
_MAX_TOKENS = 30         # maximum tokens to generate for any single value
_STRING_MAX = 128        # maximum tokens for a string value


def build_prefix_mask(prompt, remaining_names, already_generated, mask)-> NDArray:
    vocab = Vocabulary()
    # List initialized of False used to be modified to make just the allowed function names True

    #Loop over all tokens and check if this can be a part of at least one allowed function
    for token_str, token_id  in vocab.token_to_id.items():
        cleaned_token = clean_token(token_str)
        candidate = already_generated + cleaned_token
        for function_name in remaining_names:
            if function_name.startswith(candidate) :  #check both is important
                mask[token_id] = True
                break

    return mask

def apply_mask(logits: NDArray, mask: NDArray) -> NDArray:
    logits[~mask] = _NEG_INF
    return logits


def normalize(s):
    return s.replace("▁", "").strip()


class Constrained_Decoder:
    def __init__(self, model):
        self.model = model

    def select_function(self, prompt, allowed_functions: List[str]) -> str:

        vocab = Vocabulary()
        already_generated = ""
        prompt_ids = model.encode(prompt).tolist()[0]
        remaining_fun = set(allowed_functions)
        for i in range(_MAX_TOKENS):
            logits = model.get_logits_from_input_ids(prompt_ids)
            logits = np.array(logits, dtype=np.float32) #Convert logits to np.NDArray
            mask = np.zeros(len(logits), dtype=bool)
            mask: NDArray = build_prefix_mask(prompt, remaining_fun, already_generated, mask)
            if not mask.any():
                break

            logits = apply_mask(logits, mask)

            next_id = int(np.argmax(logits))
            next_str = clean_token(vocab.token_str(next_id))
            already_generated += next_str
            remaining_fun = [
                fn for fn in remaining_fun
                if fn.startswith(already_generated)
            ]
            if already_generated in allowed_functions:
                return already_generated

            if '"' in already_generated:
                return already_generated

        return ""

    def select_arguments (self, prompt, function: FunctionDef):

        already_extracted = {}
        for name, param in function.parameters.items():
            param_type = param.type
            param_prompt = build_prompt_for_argument(prompt, function, name, param_type, already_extracted)

            try:
                if param_type in ("number", "float"):
                    generate_number(prompt, )


def clean_token(token_str: str) -> str:
    return (
        token_str
        .replace("Ġ", " ")
        .replace("Ċ", "\n")
        .replace("▁", " ")
        .replace("Ĳ", "ij")
    )


def generate_number(prompt, ):
    pass


