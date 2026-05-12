"""Entry point: runs constrained decoding over all prompts and saves JSON."""

import json
import sys
from typing import Dict, List

from src.decoder import Constrained_Decoder
from src.models import Data, FunctionDef


def main() -> None:
    """Load data, decode each prompt, and write results to the output file."""
    try:
        data = Data()
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        print(f"[ERROR] Failed to initialise Data: {exc}", file=sys.stderr)
        sys.exit(1)

    # Store simple list of data to easy access cross the program
    allowed_functions: List[str] = [
        fn.name for fn in data.functions_definition
    ]
    fn_map: Dict[str, FunctionDef] = {
        fn.name: fn for fn in data.functions_definition
    }

    try:
        decoder = Constrained_Decoder()
    except Exception as exc:  # broad catch: SDK may raise anything on init
        print(
            f"[ERROR] Failed to initialise decoder: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    prompts: List[str] = data.prompts
    json_output: List[Dict] = []

    for i, prompt in enumerate(prompts):
        print(f"[{i + 1}/{len(prompts)}] processing...")

        # ── select function name ────────────────────────────────────────────
        try:
            chosen_function: str = decoder.select_function_name(
                prompt, data.functions_definition
            )
        except Exception as exc:
            print(
                f"[ERROR] select_function_name failed for prompt "
                f"{i + 1}: {exc}",
                file=sys.stderr,
            )
            # Append a sentinel record so output stays aligned with input
            json_output.append(
                {"prompt": prompt, "name": "", "parameters": {}}
            )
            continue

        # ── generate arguments only when a valid function was decoded ───────
        arguments: Dict = {}
        if chosen_function in fn_map:
            try:
                arguments = decoder.select_arguments(
                    prompt, fn_map[chosen_function]
                )
            except Exception as exc:
                print(
                    f"[ERROR] select_arguments failed for prompt "
                    f"{i + 1}: {exc}",
                    file=sys.stderr,
                )
                # arguments stays {} – we still record the function name

        # ── validate that the decoded name is in the allowed set ────────────
        if chosen_function not in allowed_functions:
            print(
                f"Cannot generate function name: {prompts[i]}",
                file=sys.stderr,
            )
            chosen_function = ""

        # Diagnostic print (arrow comment kept from original) ------------>
        print(f"{chosen_function}")

        json_output.append(
            {
                "prompt": prompt,
                "name": chosen_function,
                "parameters": arguments,
            }
        )

    # ── write results ───────────────────────────────────────────────────────
    try:
        with open(data.output_path, "w", encoding="utf-8") as f:
            json.dump(json_output, f, indent=2)
    except OSError as exc:
        print(
            f"[ERROR] Could not write output file "
            f"'{data.output_path}': {exc}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
