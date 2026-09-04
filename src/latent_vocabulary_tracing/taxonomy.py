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
