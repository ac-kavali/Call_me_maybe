import json
import sys
from argparse import ArgumentParser
from typing import Dict, List, Any
from pydantic import BaseModel, ValidationError


class Prompt(BaseModel):
    """Wraps a single prompt string for Pydantic validation."""

    prompt: str


class Parameter(BaseModel):
    """Describes one parameter of a function (type only)."""

    type: str


class FunctionDef(BaseModel):
    """Full metadata for one callable function exposed to the LLM."""

    name: str
    description: str
    parameters: Dict[str, Parameter]
    returns: Parameter


class Data:
    """Loads and exposes functions, prompts, and output path from CLI args."""

    def __init__(self) -> None:
        """Parse CLI args, read JSON files, and validate all records."""
        functions_path, prompts_path, output_path, model = self._arg_parsing()
        print(prompts_path)
        # ── load function definitions ────────────────────────────────────
        try:
            with open(functions_path, "r", encoding="utf-8") as f:
                json_function_def: List[dict[Any, Any]] = json.load(f)
        except FileNotFoundError:
            print(
                f"[ERROR] Functions file not found: '{functions_path}'",
                file=sys.stderr,
            )
            sys.exit(1)
        except json.JSONDecodeError as exc:
            print(
                f"[ERROR] Invalid JSON in functions file: {exc}",
                file=sys.stderr,
            )
            sys.exit(1)

        # ── load prompts ─────────────────────────────────────────────────
        try:
            with open(prompts_path, "r", encoding="utf-8") as f:
                json_prompts_def: List[dict[Any, Any]] = json.load(f)
        except FileNotFoundError:
            print(
                f"[ERROR] Prompts file not found: '{prompts_path}'",
                file=sys.stderr,
            )
            sys.exit(1)
        except json.JSONDecodeError as exc:
            print(
                f"[ERROR] Invalid JSON in prompts file: {exc}",
                file=sys.stderr,
            )
            sys.exit(1)

        # ── validate and unwrap prompts ──────────────────────────────────
        # Pydantic validation ensures each entry has a 'prompt' string key.
        try:
            prompt_objs: List[Prompt] = [
                Prompt(prompt=p["prompt"]) for p in json_prompts_def
            ]
        except (ValidationError, KeyError) as exc:
            print(
                f"[ERROR] Prompt validation failed: {exc}",
                file=sys.stderr,
            )
            sys.exit(1)

        # Plain list of strings for iteration throughout the pipeline
        self.prompts: List[str] = [p.prompt for p in prompt_objs]

        # ── validate and build FunctionDef objects ───────────────────────
        try:
            self.functions_definition: List[FunctionDef] = [
                FunctionDef(**fn) for fn in json_function_def
            ]
        except (ValidationError, TypeError) as exc:
            print(
                f"[ERROR] FunctionDef validation failed: {exc}",
                file=sys.stderr,
            )
            sys.exit(1)
        # defining output path from argument or default
        self.output_path: str = output_path
        # Defining the model used in constrained decoding from args or default
        self.model = model

    # ── private helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _arg_parsing() -> tuple[Any, Any, Any, Any]:
        """Register and parse CLI arguments.

        Returns:
            Tuple of (functions_definition_path, prompts_path, output_path).
        """
        parser = ArgumentParser()
        parser.add_argument(
            "--functions_definition",
            "-f",
            default="data/input/functions_definition.json",
            help="Path to the JSON file with function definitions."
                 "Default: -f data/input/functions_definition.json",
        )
        parser.add_argument(
            "--input",
            "-i",
            default="data/input/function_calling_tests.json",
            help="Path to the JSON file with prompts.\n"
                 "Default: -i data/input/function_calling_tests.json",
        )
        parser.add_argument(
            "--output",
            "-o",
            default="data/output/function_calls.json",
            help="Path where decoded results will be written."
            "Default: -o data/output/function_calls.json",
        )
        parser.add_argument(
            "--model",
            "-m",
            default="Qwen/Qwen3-0.6B",
            help="HuggingFace model identifier to load via llm_sdk."
            "Default: -m Qwen/Qwen3-0.6B",
        )
        args = parser.parse_args()
        return args.functions_definition, args.input, args.output, args.model
