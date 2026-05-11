from typing import Dict

from src.models import FunctionDef


def build_prompt_for_argument(
    user_prompt: str,
    function: FunctionDef,
    param_name: str,
    param_type: str,
    already_extracted: Dict[str, object],
) -> str:
    """Build the full LLM prompt used to extract one function argument.

    Args:
        user_prompt:       The original natural-language request.
        function:          The FunctionDef whose argument is being extracted.
        param_name:        Name of the parameter currently being generated.
        param_type:        JSON-schema type string of that parameter.
        already_extracted: Dict of parameters decoded in earlier iterations;
                           they are inlined into the partial JSON so the model
                           has context about what has already been decided.

    Returns:
        A fully formatted prompt string ending just before the parameter
        value so the model can complete it directly.
    """
    # Serialise previously extracted params as partial JSON key-value pairs
    already_extracted_json: str = ""
    if already_extracted:
        pairs = ", ".join(
            f'"{k}": {_format_value(v)}'
            for k, v in already_extracted.items()
        )
        # Trailing comma+space separates these pairs from the new param key
        already_extracted_json = f"{pairs}, "

    # String values start with an opening double-quote; other types do not
    value_start: str = '"' if param_type == "string" else ""

    hint: str = _build_extraction_hint(user_prompt, param_name)

    return (
        f"Task: Complete the JSON function call.\n"
        f"Function: {function.name} \u2014 {function.description}\n"
        f"\nRULES:\n"
        f"1. You must write the SHORTEST possible pattern.\n"
        f"2. For numbers, output EXACTLY [0-9]+ and immediately stop.\n"
        f"3. For vowels, output EXACTLY aeiouAEIOU and immediately stop.\n"
        f"4. DO NOT repeat patterns. "
        f'ALWAYS close the string with a double quote (")!\n'
        f"5. ALWAYS escape double quotes in parameters with (\\)!\n"
        f"6. When generating a path, ALWAYS generate the full path "
        f"not just the file name.\n"
        f'7. NEVER include "database" for database parameter, '
        f"only the name of the database.\n"
        f"\n--- EXAMPLES ---\n"
        f"Input: \"Replace all vowels in 'this is a test' "
        f'with asterisks"\n'
        f'JSON: {{"name": "fn_substitute_string_with_regex", '
        f'"parameters": {{"source_string": "this is a test", '
        f'"regex": "[aeiouAEIOU]", "replacement": "*"}}}}\n'
        f"\n"
        f"Input: \"Replace all numbers in 'Phone 555-1234' "
        f'with NUMBERS"\n'
        f'JSON: {{"name": "fn_substitute_string_with_regex", '
        f'"parameters": {{"source_string": "Phone 555-1234", '
        f'"regex": "[0-9]+", "replacement": "NUMBERS"}}}}\n'
        f"\n"
        f'Input: "Run the query \'INSERT INTO logs VALUES (1, 2, 3)\' '
        f'on the system database"\n'
        f'JSON: {{"name": "fn_execute_sql_query", "parameters": '
        f'{{"query": "INSERT INTO logs VALUES (1, 2, 3)", '
        f'"database": "system"}}}}\n'
        f"\n"
        f'Input: "Format template: Say \\"hello\\" to {{name}}"\n'
        f'JSON: {{"name": "fn_format_template", "parameters": '
        f'{{"template": "Say \\\\"hello\\\\" to {{{{name}}}}"}}}}\n'
        f"\n--- NOW COMPLETE ---\n"
        f"Hint: {hint}\n"
        f'Input: "{user_prompt}"\n'
        f"JSON:\n"
        f"{{\n"
        f'    "name": "{function.name}",\n'
        f'    "parameters": {{'
        f"{already_extracted_json}"
        f'"{param_name}": {value_start}'
    )


# ── private helpers ──────────────────────────────────────────────────────────

def _build_extraction_hint(user_prompt: str, param_name: str) -> str:
    """Return a short hint string that guides extraction for *param_name*.

    For the special ``regex`` parameter the hint is derived dynamically from
    keywords found in *user_prompt* (vowel, number/digit, word/substitute).
    All other parameter names are looked up in a static hints table; unknown
    names fall back to a generic template.

    Args:
        user_prompt: The original natural-language request (used for regex).
        param_name:  Name of the parameter being extracted.

    Returns:
        A human-readable hint sentence.
    """
    # Static per-parameter hints used for most common argument names
    hints: Dict[str, str] = {
        "s": (
            "extract only the string to reverse, not the full sentence"
        ),
        "source_string": (
            "extract only the input text to search in, "
            "not the full sentence"
        ),
        "replacement": (
            "extract only the replacement value, "
            "a single character or word"
        ),
        "name": "extract only the name from the request",
        "a": "extract only the number value",
        "b": "extract only the second number value",
    }

    # Dynamic hint for the 'regex' parameter: inspect the prompt for clues
    if param_name == "regex":
        user_lower = user_prompt.lower()
        if "vowel" in user_lower:
            return "use the regex pattern [aeiouAEIOU] for vowels"
        if "number" in user_lower or "digit" in user_lower:
            return "use the regex pattern [0-9]+ for numbers"
        if (
            "word" in user_lower
            or "substitute" in user_lower
            or "replace" in user_lower
        ):
            # Encourage the model to emit the literal word/pattern only
            return (
                "extract only the exact word or pattern to match, "
                "no capturing groups"
            )

    return hints.get(
        param_name,
        f'extract only the value for "{param_name}"',
    )


def _type_hint(param_type: str) -> str:
    """Return a human-readable description for a JSON-schema type string.

    Args:
        param_type: One of 'number', 'float', 'integer', 'string', 'boolean'.

    Returns:
        A short description suitable for use inside a prompt.
    """
    hints: Dict[str, str] = {
        "number": "a numeric value, no quotes",
        "float": "a decimal number, no quotes",
        "integer": "an integer, no quotes",
        "string": "a text value in double quotes",
        "boolean": "true or false (no quotes)",
    }
    return hints.get(param_type, "a value")


def _format_value(value: object) -> str:
    """Serialise an extracted parameter value for inline JSON display.

    Args:
        value: The Python value to format (str, bool, or numeric).

    Returns:
        A JSON-compatible string representation.
    """
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, bool):
        # bool check must come before int because bool is a subclass of int
        return "true" if value else "false"
    return str(value)
