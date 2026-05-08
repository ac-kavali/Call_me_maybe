import numpy as np
import json
from llm_sdk import Small_LLM_Model
from typing import List, Set, Dict
from numpy.typing import NDArray

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



def build_prefix_mask(prompt, allowed_names)-> NDArray:
    vocab = Vocabulary()
    mask = np.zeros(vocab.size, dtype=bool)
    print(vocab.size)

    return mask




class Constrained_Decoder:
    def __init__(self, model):
        self.model = model


    def select_function(self, prompt, allowed_functions: List[str]):
        mask: NDArray = build_prefix_mask(prompt, allowed_functions)







def _clean_token(token_str: str) -> str:
    return (
        token_str
        .replace("Ġ", " ")
        .replace("Ċ", "\n")
        .replace("▁", " ")
        .replace("Ĳ", "ij")
    )



_NEG_INF = -1e9          # practical negative-infinity for logit masking
_MAX_TOKENS = 64         # maximum tokens to generate for any single value
_STRING_MAX = 128        # maximum tokens for a string value



