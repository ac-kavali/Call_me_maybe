from src.models import FunctionDef

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
    """
    Build a prompt that asks the LLM to extract one specific argument.

    Args:
        user_prompt: The user's natural language request.
        function: The selected function definition.
        param_name: The name of the parameter to extract.
        param_type: The expected type ("number", "string", "boolean").
        already_extracted: Parameters already extracted (for context).

    Returns:
        A formatted prompt string ending right before the argument value.
    """
    param_context = ""
    if already_extracted:
        pairs = ", ".join(
            f'"{k}": {_format_value(v)}' for k, v in already_extracted.items()
        )
        param_context = f"\nAlready extracted: {pairs}"

    type_hint = _type_hint(param_type)

    return (
        f"You are a function-calling assistant.\n"
        f'Function: "{function.name}" — {function.description}\n'
        f'User request: "{user_prompt}"\n'
        f"{param_context}\n"
        f'Extract the value for parameter "{param_name}" ({type_hint}).\n\n'
        f'The value of "{param_name}" is: '
    )


def _type_hint(param_type: str) -> str:
    """Return a human-readable type description."""
    hints = {
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











