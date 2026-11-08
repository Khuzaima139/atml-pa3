"""MATH answer extraction: the balanced-brace rule of the supplied README (R32; manual L622).

"For MATH, extraction finds the last literal `\\boxed{` and scans balanced braces, preserving nested LaTeX and
stripping surrounding whitespace. An absent, empty, or unclosed last box gives null; there is no numeric fallback."

Reproduces the supplied MATH `extracted_answer` on all 32,000 rollouts (TASK3_FACTS item 13; test in
tests/task3/test_task3_extract.py).
"""

BOX = "\\boxed{"


def last_boxed(text):
    """Content of the last literal '\\boxed{' with balanced braces, stripped; absent, empty or unclosed -> None."""
    if text is None:
        return None
    i = text.rfind(BOX)
    if i < 0:
        return None
    j = i + len(BOX)
    depth = 1
    for k in range(j, len(text)):
        c = text[k]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                s = text[j:k].strip()
                return s if s else None
    return None
