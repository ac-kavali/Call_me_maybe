"""Pydantic data-models and CLI data loader used across the project."""

import json
import sys
from argparse import ArgumentParser
from typing import Dict, List, Tuple

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
        functions_path, prompts_path, output_path = self._arg_parsing()

        # ── load function definitions ────────────────────────────────────
        try:
            with open(functions_path, "r", encoding="utf-8") as f:
                json_function_def: List[dict] = json.load(f)
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
                json_prompts_def: List[dict] = json.load(f)
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

        self.output_path: str = output_path

    # ── private helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _arg_parsing() -> Tuple[str, str, str]:
        """Register and parse CLI arguments.

        Returns:
            Tuple of (functions_definition_path, prompts_path, output_path).
        """
        parser = ArgumentParser(
            description="Constrained LLM function-calling decoder."
        )
        parser.add_argument(
            "--functions_definition",
            "-f",
            default="data/input/functions_definition.json",
            help="Path to the JSON file with function definitions.",
        )
        parser.add_argument(
            "--input",
            "-i",
            default="data/input/function_calling_tests.json",
            help="Path to the JSON file with prompts.",
        )
        parser.add_argument(
            "--output",
            "-o",
            default="data/output/function_calling_tests.json",
            help="Path where decoded results will be written.",
        )
        args = parser.parse_args()
        return args.functions_definition, args.input, args.output
