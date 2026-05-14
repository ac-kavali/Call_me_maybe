*This project has been created as part of the 42 curriculum by achahi.*

# call me maybe — Introduction to Function Calling in LLMs

---

## Description

**call me maybe** is a function-calling tool that bridges the gap between natural language and machine-executable structured output. Given a plain-English prompt like `"What is the sum of 40 and 2?"`, the system does **not** answer `42` — instead, it produces:

```json
{
  "prompt": "What is the sum of 40 and 2?",
  "name": "fn_add_numbers",
  "parameters": { "a": 40.0, "b": 2.0 }
}
```

The core challenge is reliability: a tiny 0.6B-parameter model left to its own devices will produce valid JSON only ~30% of the time. This project solves that with **constrained decoding** — intercepting the model's logit distribution at every generation step and masking out any token that would break the required JSON schema, pushing reliability to ~100%.

---

## How I Solved It

### The Big Picture

The generation pipeline works like this:

```
Prompt → Tokenize → Input IDs → LLM → Logits → Mask Invalid Tokens → argmax → Next Token → repeat
```

Instead of letting the model freely pick the next token, I intervene at the logits stage every single step.

### Step 1 — Function Name Selection

The first thing to generate is which function to call. I build a mask that allows **only** the exact token sequences that spell out one of the valid function names from `function_definitions.json`. Everything else gets set to `-inf`. The model then has no choice but to pick a valid function name.

This mask is built once and reused across all prompts (performance win).

### Step 2 — Argument Generation

Once the function is chosen, I look up its schema (parameter names and types). For each parameter I run a dedicated generator:

- **Numbers** (`generate_number`): only digits, `-`, `.`, `e/E`, and terminator characters (`space`, `,`, `}`, etc.) are allowed. A state machine tracks whether a dot or exponent has already appeared to prevent malformed numbers like `1..2`.
- **Strings** (`generate_str`): only printable characters are allowed via a prebuilt `M_chars` mask. Generation stops when an unescaped closing `"` is decoded.
- **Booleans** (`_generate_boolean`): only `true`, `false`, and their BPE sub-token prefixes (`t`, `tr`, `tru`, `f`, `fa`, `fals`) are allowed. The top token's first character decides the result.

### Step 3 — Output

All results are collected into a list of dicts with keys `prompt`, `name`, and `parameters`, then written as pretty-printed JSON to the output file. The output directory is created at runtime if it does not exist.

### Why This Works

The model never gets a chance to hallucinate an invalid token because it simply cannot choose one — its logit is `-inf`. This is the difference between *prompting and hoping* versus *structural enforcement*.

---

## Algorithm Explanation

**Constrained decoding** modifies the probability distribution (logits) the LLM outputs at each token step before any sampling or argmax takes place:

1. The model produces raw logits over its full vocabulary (~150k tokens).
2. A mask is computed: `True` for tokens that keep the output valid, `False` for everything else.
3. Invalid positions are set to `-inf` so they can never be selected.
4. `argmax` picks the single highest remaining logit.
5. The chosen token is appended to the prompt and the loop repeats.

For function-name generation, validity means "this token is a prefix of (or exactly) one of the allowed function names." For number generation, validity is determined by a state machine that tracks what characters have already been emitted. For strings, validity means the token is a printable character that is not an unescaped closing quote.

---

## Design Decisions

| Decision | Reason |
|---|---|
| Greedy decoding (argmax) instead of sampling | Deterministic — same prompt always gives the same function call. Sampling would introduce randomness that could break type constraints. |
| Masks built once for function names | The allowed function set does not change between prompts. Pre-building saves repeated vocabulary scans. |
| Per-type generator methods | Each JSON type (number, string, boolean) has different termination logic and character rules. Separating them keeps each one simple and testable. |
| Pydantic for all data classes | Catches malformed input files early with clear error messages before any LLM inference runs. |
| `uv` for dependency management | Required by the project spec; `uv sync` is the single setup command for reviewers. |
| Sentinel records on per-prompt errors | If one prompt fails, the output array stays aligned with the input array rather than silently dropping entries. |

---

## Instructions

### Requirements

- Python 3.10+
- `uv` package manager

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd <repo-directory>

# Copy the provided llm_sdk into the project root (same level as src/)
# Then install dependencies
uv sync
```

### Running the Program

Default paths (reads from `data/input/`, writes to `data/output/`):

```bash
uv run python -m src
```

Custom paths:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

### Makefile targets

```bash
make install   # install dependencies
make run       # run with default paths
make debug     # run under pdb
make lint      # flake8 + mypy
make clean     # remove __pycache__, .mypy_cache
```

---

## Example Usage

Given `data/input/function_calling_tests.json`:

```json
[
  { "prompt": "What is the sum of 2 and 3?" },
  { "prompt": "Greet shrek" },
  { "prompt": "Reverse the string 'hello'" }
]
```

The program produces `data/output/function_calls.json`:

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": { "a": 2.0, "b": 3.0 }
  },
  {
    "prompt": "Greet shrek",
    "name": "fn_greet",
    "parameters": { "name": "shrek" }
  },
  {
    "prompt": "Reverse the string 'hello'",
    "name": "fn_reverse_string",
    "parameters": { "s": "hello" }
  }
]
```

---

## Performance Analysis

| Metric | Result |
|---|---|
| JSON validity | 100% — constrained decoding guarantees parseable output |
| Function selection accuracy | ~90–95% on the provided test set |
| Argument extraction accuracy | ~90%+ for simple numeric and string types |
| Speed | All prompts processed in under 5 minutes on standard hardware |
| Reliability across runs | Deterministic (greedy argmax), identical output on repeated runs |

The Qwen3-0.6B model has only ~500M parameters. Without structural guidance it fails to produce valid JSON most of the time. With constrained decoding it matches the reliability of models ten times its size.

---

## Challenges Faced

**1. Vocabulary mapping.**
The LLM vocabulary contains ~150k tokens, many of which are BPE sub-words with leading space markers (`Ġ`). Correctly normalizing these (`Ġ` → ` `) was essential for string comparisons to work.

**2. Number termination.**
Numbers have no explicit closing delimiter — the model must stop when it sees a separator like `,` or `}`. This required allowing terminator tokens inside the number mask while not treating them as part of the number.

**3. Multi-token function names.**
Function names like `fn_reverse_string` are split into several BPE tokens. The mask must allow any token that is a valid *prefix* of any allowed name, not just the full name in one shot.

**4. String closing-quote detection.**
The closing `"` can appear mid-token (e.g., the model might emit `hello"`). The generator had to detect unescaped quotes mid-token and truncate correctly.

**5. Conflict between subject and correction sheet key names.**
The subject specifies `name` and `parameters` while the correction sheet mentions `fn_name` and `args`. The implementation uses `name` and `parameters` as stated in the subject; the moulinette is the final arbiter.

---

## Testing Strategy

- **Manual inspection** of output JSON for a variety of prompt types: arithmetic, greetings, string operations, edge cases (empty strings, large numbers, special characters).
- **JSON validity check**: `json.loads()` on every output file after generation — zero parse errors expected.
- **Schema check**: verified that all keys and types in the output match the corresponding `function_definitions.json` entry.
- **Error-path testing**: ran with missing input files, malformed JSON, and empty prompt lists to confirm graceful error messages and clean exit codes.
- **Repeatability**: ran the full pipeline multiple times and confirmed identical output (deterministic greedy decoding).

---

## Resources

### Topic references

- [Attention Is All You Need — Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762) — the transformer architecture underlying all modern LLMs.
- [Outlines: Guided Generation for LLMs](https://github.com/outlines-dev/outlines) — the concept this project reimplements from scratch (use of the library is forbidden by the spec).
- [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B) — the default model used.
- [Byte-Pair Encoding (BPE) tokenization](https://huggingface.co/learn/nlp-course/chapter6/5) — explains why vocabulary tokens are sub-words with space markers.
- [JSON specification (RFC 8259)](https://datatracker.ietf.org/doc/html/rfc8259) — the grammar constrained decoding enforces.
- [Pydantic v2 documentation](https://docs.pydantic.dev/) — used for all data validation.

### How AI was used

AI (Claude) was used for:
- **Explaining concepts**: understanding BPE tokenization, logit masking, and the relationship between token IDs and string representations.
- **Debugging assistance**: identifying edge cases in the number and string generators (e.g., mid-token quote detection, terminator-character handling).
- **Code review**: checking that type hints, docstrings, and exception handling were complete before submission.

All code was written, understood, and can be fully explained by the author. No AI-generated code was pasted blindly into the project.
