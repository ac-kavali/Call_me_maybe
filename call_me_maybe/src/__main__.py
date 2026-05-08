from src.models import Data
from llm_sdk import  Small_LLM_Model
from typing import List
from src.decoder import Constrained_Decoder
from src.build_prompts import build_prompt_for_argument, build_function_selection_prompt


data = Data()
allowed_functions = [fn.name for fn in data.functions_definition]
print(allowed_functions)
model = Small_LLM_Model()
prompts: List[str] = data.prompts
decoder = Constrained_Decoder(Small_LLM_Model)

for i, prompt in enumerate(prompts):
    print(f"[{i + 1}/{len(data.prompts)}] processing...")

    # prompt to make ids from
    structured_prompt = build_function_selection_prompt(prompt, data.functions_definition)

    # the picked function name
    function = decoder.select_function(structured_prompt, allowed_functions)


    break

