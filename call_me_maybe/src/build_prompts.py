from src.models import FunctionDef
from typing import Dict

def build_function_selection_prompt(
    user_prompt: str,
    functions,
) -> str:
    """
    Build a prompt that asks the LLM to select the correct function name.

    The prompt is designed so the model's next token is naturally one of
    the valid function names. We rely on constrained decoding — not this
    prompt alone — to enforce the output, but a clear prompt still helps.

    Args:
        user_prompt: The user's natural language request.
        functions: Available function definitions.

    Returns:
        A formatted prompt string.
    """
    fn_descriptions = "\n".join(
        f'  - "{fn.name}": {fn.description}' for fn in functions
    )

    return (
        f"You are a function-calling assistant.\n"
        f"Select the best function for the user's request.\n\n"
        f'Example: prompt contains "Greet" you return "fn_greet"'
        f"Available functions:\n{fn_descriptions}\n\n"
        f'User request: "{user_prompt.lower()}"\n\n'
        f"The function to call is: \""
    )


def build_prompt_for_argument(
    user_prompt: str,
    function: FunctionDef,
    param_name: str,
    param_type: str,
    already_extracted: dict,  # type: ignore[type-arg]
) -> str:
    already_extracted_json: str = ""
    if already_extracted:
        pairs = ", ".join(
            f'"{k}": {_format_value(v)}' for k, v in already_extracted.items()
        )
        already_extracted_json = f"{pairs}, "

    if param_type == "string":
        value_start: str = '"'
    else:
        value_start = ""

    hint: str = _build_extraction_hint(user_prompt, param_name)

    return (
        f'Task: Complete the JSON function call.\n'
        f'Function: {function.name} — {function.description}\n'
        f'\nRULES:\n'
        f'1. You must write the SHORTEST possible pattern.\n'
        f'2. For numbers, you must output EXACTLY [0-9]+ and immediately stop.\n'
        f'3. For vowels, you must output EXACTLY aeiouAEIOU and immediately stop.\n'
        f'4. DO NOT repeat patterns. ALWAYS close the string with a double quote (")!\n'
        f'5. ALWAYS escape double quotes in parameters with (\\)!\n'
        f'6. When generating a path, ALWAYS generate the full path not just the file name.\n'
        f'7. NEVER include "database" for database parameter, only the name of the database.\n'
        f'\n--- EXAMPLES ---\n'
        f'Input: "Replace all vowels in \'this is a test\' with asterisks"\n'
        f'JSON: {{"name": "fn_substitute_string_with_regex", "parameters": '
        f'{{"source_string": "this is a test", "regex": "[aeiouAEIOU]", "replacement": "*"}}}}\n'
        f'\n'
        f'Input: "Replace all numbers in \'Phone 555-1234\' with NUMBERS"\n'
        f'JSON: {{"name": "fn_substitute_string_with_regex", "parameters": '
        f'{{"source_string": "Phone 555-1234", "regex": "[0-9]+", "replacement": "NUMBERS"}}}}\n'
        f'\n'
        f'Input: "Run the query \'INSERT INTO logs VALUES (1, 2, 3)\' on the system database"\n'
        f'JSON: {{"name": "fn_execute_sql_query", "parameters": '
        f'{{"query": "INSERT INTO logs VALUES (1, 2, 3)", "database": "system"}}}}\n'
        f'\n'
        f'Input: "Format template: Say \\"hello\\" to {{name}}"\n'
        f'JSON: {{"name": "fn_format_template", "parameters": '
        f'{{"template": "Say \\\\"hello\\\\" to {{{{name}}}}"}}}}\n'
        f'\n--- NOW COMPLETE ---\n'
        f'Hint: {hint}\n'
        f'Input: "{user_prompt}"\n'
        f'JSON:\n'
        f'{{\n'
        f'    "name": "{function.name}",\n'
        f'    "parameters": {{'
        f'{already_extracted_json}'
        f'"{param_name}": {value_start}'
    )


def _build_extraction_hint(user_prompt: str, param_name: str) -> str:
    hints: Dict[str, str] = {
        "s": "extract only the string to reverse, not the full sentence",
        "source_string": "extract only the input text to search in, not the full sentence",
        "replacement": "extract only the replacement value, a single character or word",
        "name": "extract only the name from the request",
        "a": "extract only the number value",
        "b": "extract only the second number value",
    }

    # Dynamic hint for regex based on prompt content
    if param_name == "regex":
        user_lower = user_prompt.lower()
        if "vowel" in user_lower:
            return "use the regex pattern [aeiouAEIOU] for vowels"
        if "number" in user_lower or "digit" in user_lower:
            return "use the regex pattern [0-9]+ for numbers"
        if "word" in user_lower or "substitute" in user_lower or "replace" in user_lower:
            # Extract the word to match from the prompt
            return "extract only the exact word or pattern to match, no capturing groups"

    return hints.get(param_name, f'extract only the value for "{param_name}"')


def _type_hint(param_type: str) -> str:
    """Return a human-readable type description."""
    hints: Dict[str, str] = {
        "number": "a numeric value, no quotes",
        "float": "a decimal number, no quotes",
        "integer": "an integer, no quotes",
        "string": "a text value in double quotes",
        "boolean": "true or false (no quotes)",
    }
    return hints.get(param_type, "a value")


def _format_value(value: object) -> str:
    """Format an extracted value for display in prompts."""
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


