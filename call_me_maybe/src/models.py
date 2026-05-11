from pydantic import BaseModel
from typing import Dict, List
from argparse import ArgumentParser
import json


class Prompt(BaseModel):
    prompt: str

class Parameter(BaseModel):
    type: str

class FunctionDef(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Parameter]
    returns: Parameter



class Data:
    def __init__(self):
        functions_definition_path, prompts_path, output_path = self.arg_parsing()
        # Open function definitions
        with open(functions_definition_path, "r") as f:
            json_function_def: List[dict] = json.load(f)

        # Open prompts JSON file
        with open(prompts_path, "r") as f:
            json_prompts_def: List[dict] = json.load(f)

        # List of Prompt object just be valid with the BaseModel .
        prompt_validation: List[Prompt] = [Prompt(prompt=p["prompt"]) for p in json_prompts_def]
        # List of the prompt filtered to be like ["example of the prompt1", "example of prompt 2" ...]
        self.prompts: List[str] = [p.prompt for p in prompt_validation]
        # List of Functions(obj) metadata to be added to the prompt
        self.functions_definition: List[FunctionDef] = [
            FunctionDef(**fn) for fn in json_function_def
        ]
        self.output_path = output_path

    def arg_parsing (self):
        parser = ArgumentParser()

        parser.add_argument("--functions_definition", "-f", default="data/input/functions_definition.json")
        parser.add_argument("--input", "-i", default="data/input/function_calling_tests.json")
        parser.add_argument("--output", "-o", default="data/output/function_calling_tests.json")
        args = parser.parse_args()

        return [args.functions_definition, args.input, args.output]

