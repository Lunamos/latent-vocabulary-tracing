"""Human-readable categories for decoded tokenizer pieces."""

from __future__ import annotations

import re
import unicodedata

CATEGORIES = (
    "special",
    "cjk",
    "other_script",
    "junk",
    "junk_id",
    "format",
    "punct",
    "number",
    "math",
    "code",
    "discourse",
    "function",
    "english",
    "latin_piece",
    "other",
)

# A second, deliberately coarser taxonomy for reader-facing analyses.  The
# categories are mutually exclusive.  Any named pair can also be used as
# non-exhaustive plot coordinates while the remaining classes stay visible in
# the accompanying full-category summary.
FUNCTIONAL_CATEGORIES = (
    "reasoning_process",
    "answer_commitment",
    "mathematical_content",
    "symbolic_notation",
    "presentation",
    "general_language",
    "other",
)

# Confirmatory trace categories extend the mathematics-oriented pilot taxonomy
# with agent/tool and code roles.  Keep ``FUNCTIONAL_CATEGORIES`` stable so the
# existing 1.7B appendix analysis remains reproducible.
TRACE_CATEGORIES = (
    "reasoning_process",
    "answer_commitment",
    "mathematical_content",
    "symbolic_notation",
    "tool_action",
    "execution_feedback",
    "code_content",
    "presentation",
    "general_language",
    "other",
)

DISCOURSE = set(
    """wait alternatively so therefore thus hmm okay ok but first second next then now
    let lets let's actually however since because hence verify check recall note suppose
    assume indeed clearly obviously right yes no maybe perhaps finally overall alright
    well also again double-check confirm conclusion answer step steps approach method idea
    consider try let’s""".split()
)
FUNCTION = set(
    """the a an of to in and or is are was were be been being that this these those it
    its for on with as by at from we i you they he she him her his their our your my me
    us them can could will would shall should may might must not do does did done have
    has had having which what who whom whose where when why how if than then there here
    into onto out over under up down about above below between through during before
    after while all any each every some such only own same other another more most less
    least very too just both either neither nor yet still already one two three four five
    six seven eight nine ten""".split()
)
CODE_KEYWORDS = set(
    """def return import from class self print int str float bool list dict set tuple None
    True False for while if elif else try except finally with lambda yield pass break
    continue raise assert global nonlocal async await const let var function new this null
    undefined void static public private protected struct enum typedef include using
    namespace std cout cin endl nullptr auto template virtual override switch case
    default do goto sizeof unsigned char long short double main args argv len range append
    extend sorted sort map filter reduce zip enumerate isinstance input output printf
    scanf malloc free string vector array object json math random numpy np pd torch tf""".split()
)

_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯豈-﫿]")
_OTHER_SCRIPT = re.compile(r"[Ͱ-ϿЀ-ӿ֐-׿؀-ۿऀ-ॿ฀-๿ᄀ-ᇿ]")
_GREEK = re.compile(r"[Ͱ-Ͽ]")
_MATH = re.compile(
    r"(\\[a-zA-Z]+|^\s*[=+\-*/^≤≥≠±×÷∑∏∫√∞∂∇∈∉⊂⊆∪∩→⇒⇔∀∃∘·$%]+\s*$|"
    r"boxed|\bfrac\b|\bsqrt\b|\bcdot\b|\btimes\b|\btheta\b|\balpha\b|\bbeta\b|"
    r"\bgamma\b|\blambda\b|\bsigma\b|\bpi\b)"
)
_NUMBER = re.compile(r"^\s*[-+]?\d[\d,.]*\s*$")
_CODE_OPERATOR = re.compile(
    r"(==|!=|<=|>=|\+=|-=|\*=|/=|=>|->|::|&&|\|\||//|/\*|\*/|#!|\(\)|\[\]|"
    r"\{\}|;\s*$|^\s*(#include|#define|import|from)\b)"
)
_FORMAT = re.compile(r"^\s*$|^[\s*#`>|_=\-~]+$|^\s*(\*\*|###|##|#|---|```|\\n)+\s*$")
_PUNCT = re.compile(
    r"""^\s*[!"'(),.:;?\[\]{}<>/\\|@&#~`«»“”‘’…—–、。，：；！？（）「」『』]+\s*$"""
)
_WORD = re.compile(r"^\s?[A-Za-z]+[’']?[A-Za-z]*$")

_REASONING_PROCESS = {
    "actually",
    "again",
    "alternatively",
    "also",
    "approach",
    "assume",
    "because",
    "but",
    "case",
    "cases",
    "check",
    "consider",
    "consequently",
    "first",
    "given",
    "hence",
    "however",
    "if",
    "indeed",
    "instead",
    "let",
    "lets",
    "method",
    "moreover",
    "need",
    "next",
    "note",
    "now",
    "otherwise",
    "recall",
    "second",
    "similarly",
    "since",
    "so",
    "step",
    "steps",
    "suppose",
    "then",
    "therefore",
    "thus",
    "try",
    "verify",
    "wait",
    "whereas",
    "while",
    "likewise",
    "overall",
}
_ANSWER_COMMITMENT = {
    "answer",
    "answers",
    "boxed",
    "choice",
    "choices",
    "conclude",
    "concludes",
    "conclusion",
    "correct",
    "finally",
    "final",
    "option",
    "options",
    "result",
    "results",
    "solution",
    "solutions",
}
_MATHEMATICAL_CONTENT = {
    "algebra",
    "angle",
    "angles",
    "coefficient",
    "denominator",
    "derivative",
    "divisor",
    "equation",
    "equations",
    "factor",
    "factors",
    "fraction",
    "function",
    "geometry",
    "integer",
    "integers",
    "integral",
    "matrix",
    "math",
    "multiple",
    "numerator",
    "polynomial",
    "product",
    "proof",
    "ratio",
    "remainder",
    "root",
    "roots",
    "sequence",
    "set",
    "sum",
    "theorem",
    "variable",
    "variables",
    "vector",
}
_SYMBOLIC_WORDS = {
    "alpha",
    "beta",
    "cancel",
    "cdot",
    "cos",
    "cosine",
    "delta",
    "frac",
    "gamma",
    "lambda",
    "log",
    "pi",
    "sigma",
    "sin",
    "sine",
    "sqrt",
    "tan",
    "theta",
    "times",
}
_PRESENTATION_WORDS = {"latex"}
_TOOL_ACTION = {
    "analysis",
    "bash",
    "call",
    "calls",
    "cd",
    "command",
    "commands",
    "curl",
    "duration",
    "execute",
    "find",
    "git",
    "grep",
    "inspect",
    "keystrokes",
    "ls",
    "mkdir",
    "path",
    "pip",
    "plan",
    "python",
    "run",
    "search",
    "shell",
    "sudo",
    "task_complete",
    "terminal",
    "tool",
    "tools",
    "wget",
    "workspace",
}
_EXECUTION_FEEDBACK = {
    "complete",
    "completed",
    "denied",
    "error",
    "errors",
    "exception",
    "exit",
    "failed",
    "failure",
    "missing",
    "observation",
    "output",
    "permission",
    "retry",
    "status",
    "stderr",
    "stdout",
    "success",
    "traceback",
    "warning",
}
_SYMBOLIC = re.compile(
    r"^\s*(?:[-+]?\d[\d,.]*|[=+\-*/^≤≥≠±×÷∑∏∫√∞∂∇∈∉⊂⊆∪∩→⇒⇔∀∃∘·$%]+)\s*$"
)
_LATEX_COMMAND = re.compile(r"^\s*\\([A-Za-z]+)\s*$")


def categorize_token(token: str) -> str:
    """Assign one mutually exclusive, readable category to a token string."""

    if not isinstance(token, str):
        raise TypeError("token must be a string")
    if any(marker in token for marker in ("<|", "<think", "</think", "<tool", "</tool")):
        return "special"
    if "�" in token or any(
        unicodedata.category(char) == "Cc" and char not in "\n\t\r" for char in token
    ):
        return "junk"
    if _CJK.search(token):
        return "cjk"
    if _GREEK.search(token):
        return "math"
    if _OTHER_SCRIPT.search(token):
        return "other_script"
    if _FORMAT.match(token):
        return "format"
    if _PUNCT.match(token):
        return "punct"
    if _NUMBER.match(token):
        return "number"
    if _MATH.search(token) and not _WORD.match(token):
        return "math"

    stripped = token.strip()
    if (
        re.match(r"^[.(\[{]?[A-Za-z_][A-Za-z0-9_]*$", stripped)
        and len(stripped) >= 10
        and ("_" in stripped or len(re.findall(r"[A-Z]", stripped[1:])) >= 2)
    ):
        return "junk_id"
    if (
        _CODE_OPERATOR.search(token)
        or stripped.lower() in CODE_KEYWORDS
        or ("_" in stripped and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", stripped))
        or re.match(r"^[a-z]+[A-Z][A-Za-z]*$", stripped)
        or (len(re.findall(r"[A-Z]", stripped[1:])) >= 2 and re.match(r"^[A-Za-z]+$", stripped))
    ):
        return "code"
    lowered = stripped.lower().rstrip(",.:;")
    if _WORD.match(token):
        if lowered in DISCOURSE or lowered.replace("’", "'") in DISCOURSE:
            return "discourse"
        if lowered in FUNCTION:
            return "function"
        if not token.startswith(" ") and len(stripped) <= 3 and stripped.islower():
            return "latin_piece"
        if token.startswith(" ") or stripped[:1].isupper() or len(stripped) >= 4:
            return "english"
        return "latin_piece"
    if re.match(r"^\s?[A-Za-z]", token):
        return "latin_piece"
    if "\\" in token or "$" in token or _MATH.search(token):
        return "math"
    if "**" in token or "#" in token or "`" in token or "|" in token:
        return "format"
    if re.match(r"^\s*[-–—]+[A-Za-z]*$", token) or re.match(r"^[^\w\s]+$", token):
        return "punct"
    return "other"


def categorize_functional_token(token: str) -> str:
    """Assign a token piece to a reader-facing linguistic function.

    This taxonomy is intended for aggregate comparisons, not for inferring a
    token's context-dependent meaning.  Priority is explicit: answer markers
    precede reasoning markers, which precede mathematical concepts, notation,
    presentation, general Latin-language pieces, and a residual class.
    """

    if not isinstance(token, str):
        raise TypeError("token must be a string")

    stripped = token.strip()
    lowered = stripped.lower().strip(".,:;!?()[]{}<>\"'`*_~")
    lowered = lowered.replace("’", "'")

    # LaTeX commands are sometimes emitted with their backslash and sometimes
    # as a separate alphabetic tokenizer piece, so test both representations.
    command_match = _LATEX_COMMAND.match(token)
    command = command_match.group(1).lower() if command_match else ""
    lexical = command or lowered.lstrip("\\")

    if lexical in _ANSWER_COMMITMENT:
        return "answer_commitment"
    if lexical in _REASONING_PROCESS:
        return "reasoning_process"
    if lexical in _MATHEMATICAL_CONTENT:
        return "mathematical_content"
    if lexical in _SYMBOLIC_WORDS or _SYMBOLIC.match(token):
        return "symbolic_notation"
    if command:
        return "symbolic_notation"
    if lexical in _PRESENTATION_WORDS:
        return "presentation"

    coarse = categorize_token(token)
    if coarse in {"format", "punct"}:
        return "presentation"
    if coarse in {"english", "function", "discourse", "latin_piece"}:
        return "general_language"
    return "other"


def categorize_trace_token(token: str) -> str:
    """Assign a token piece to the cross-domain confirmatory taxonomy.

    This classifier remains intentionally surface based: sequence-role masks
    carry the contextual distinction between tool observations, calls, and
    continuations.  The extra lexical classes prevent agent protocol and code
    pieces from disappearing into ``general_language`` or the residual class.
    """

    if not isinstance(token, str):
        raise TypeError("token must be a string")

    stripped = token.strip()
    lowered = stripped.lower().strip(".,:;!?()[]{}<>\"'`*_~")
    lowered = lowered.replace("’", "'")
    command_match = _LATEX_COMMAND.match(token)
    command = command_match.group(1).lower() if command_match else ""
    lexical = command or lowered.lstrip("\\")

    # Preserve the pilot taxonomy's explicit semantic priorities.
    if lexical in _ANSWER_COMMITMENT:
        return "answer_commitment"
    if lexical in _REASONING_PROCESS:
        return "reasoning_process"
    if lexical in _MATHEMATICAL_CONTENT:
        return "mathematical_content"
    if lexical in _SYMBOLIC_WORDS or _SYMBOLIC.match(token) or command:
        return "symbolic_notation"
    if lexical in _TOOL_ACTION:
        return "tool_action"
    if lexical in _EXECUTION_FEEDBACK:
        return "execution_feedback"

    coarse = categorize_token(token)
    if coarse == "code":
        return "code_content"
    if lexical in _PRESENTATION_WORDS or coarse in {"format", "punct"}:
        return "presentation"
    if coarse in {"english", "function", "discourse", "latin_piece"}:
        return "general_language"
    return "other"


def is_displayable_trace_token(token: str) -> bool:
    """Return whether a token piece can be labelled legibly in a main figure.

    The filter is deterministic and deliberately conservative.  Unfiltered
    rankings remain part of every result; this predicate only supplies a
    companion list that excludes control bytes, special tokens, whitespace,
    and bare alphabetic continuation fragments.  Named lexicon entries,
    standalone notation/numbers, code forms, and whitespace-delimited words
    remain eligible.
    """

    if not isinstance(token, str):
        raise TypeError("token must be a string")
    if not token.strip() or "�" in token or "<|" in token:
        return False
    if any(unicodedata.category(char).startswith("C") for char in token):
        return False

    stripped = token.strip()
    lowered = stripped.lower().strip(".,:;!?()[]{}<>\"'`*_~").replace("’", "'")
    command_match = _LATEX_COMMAND.match(token)
    command = command_match.group(1).lower() if command_match else ""
    lexical = command or lowered.lstrip("\\")
    named = (
        _ANSWER_COMMITMENT
        | _REASONING_PROCESS
        | _MATHEMATICAL_CONTENT
        | _SYMBOLIC_WORDS
        | _TOOL_ACTION
        | _EXECUTION_FEEDBACK
        | CODE_KEYWORDS
    )
    if lexical in named or command or _SYMBOLIC.match(token) or _NUMBER.match(token):
        return True
    if _CODE_OPERATOR.search(token):
        return True
    if token[:1].isspace() and stripped.replace("-", "").isalpha() and len(stripped) >= 2:
        return True
    return bool(_CJK.fullmatch(stripped))
