
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
        f"Available functions:\n{fn_descriptions}\n\n"
        f'User request: "{user_prompt}"\n\n'
        f"The function to call is: \""
    )


def build_prompt_for_argument(
    user_prompt: str,
    function_name: str,
    arg_name: str,
    arg_type: str
) -> str:
    """Build the prompt for one argument value.

    Args:
        user_prompt: the natural language request from the user.
        function_name: the chosen function name.
        arg_name: the name of the argument to fill.
        arg_type: the type of the argument (string, number, boolean).

    Returns:
        A prompt string ending so the LLM starts generating
        the argument value immediately.
    """
    lines = [f"User request: {user_prompt}", f"Function: {function_name}"]
    if arg_type == "string":
        lines.append(f'Fill argument "{arg_name}" (type: {arg_type}): "')
    else:
        lines.append(f'Fill argument "{arg_name}" (type: {arg_type}): ')
    return "\n".join(lines)










