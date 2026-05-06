import json
from pydantic import BaseModel, StrictStr
from src.argparsing import arg_parsing
from typing import Dict
from llm_sdk import  Small_LLM_Model
import numpy as np
from typing import List
import json
from src.prompt_ingeneer import build_prompt_for_argument, build_prompt_for_function

functions_definition, prompt_json, output = arg_parsing()

model = Small_LLM_Model()

with open(functions_definition, "r") as f:
   json_fundef : List[dict] = json.load(f)

with open(prompt_json, "r") as f:
    json_promptdef: List[dict] = json.load(f)


class Prompt(BaseModel):
    prompt: str

class Parameter(BaseModel):
    type: str

class FunctionDef(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Parameter]
    returns: Parameter

# List of Prompt object just be valid with the BaseModel .
prompt_validation : List[Prompt]= [Prompt(prompt=p["prompt"]) for p in json_promptdef ]

#List of the prompt filtred to be like ["example of the prompt1", "example of prompt 2" ...]
prompts: List[str] = [p.prompt for p in prompt_validation]

# List of Functions(obj) metadata to be added to the prompt
functions : List[FunctionDef] = [FunctionDef(**fn) for fn in json_fundef]


#Testing the prompt function
# for p in prompts:
#     function_prompt = build_prompt_for_function(p,functions)
#     print(function_prompt)
#     break

argument_prompt = build_prompt_for_argument(prompts[1], functions[0].name, "number", "string")
# print(argument_prompt);

print(prompts[0])

# text =""
# while "}" not in text:
#     ids = model.encode(prompt).tolist()[0]
#     logits = model.get_logits_from_input_ids(ids)
#     text = model.decode(np.argmax(logits))
#     print(text)
#     prompt += text


