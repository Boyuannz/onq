"""Provide text normalization, near-duplicate indexing and prompt quality checks; shared helper module.

Prompt quality gates shared by the Block-AP base and the OPD prompt pool (stdlib only).

Added after independent review of the 2026-09-29 builds found (1) paraphrased eval problems that the
normalized-exact / 5-gram check missed (LaTeX spellings, "97,968" vs "97968", renamed variables),
(2) NSFW and jailbreak role-play prompts in WildChat-derived data, and (3) code prompts with stripped
figures, missing statements, HTML-escaped samples or truncated starter code."""

import ast
import collections
import difflib
import hashlib
import re

# ----------------------------------------------------------------------------- normalization
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
# LaTeX/formatting words that carry no content ("\frac{1}{2}" vs "1/2", "\le" vs "≤", "\dots" vs "...").
_DROP = frozenset(
    "frac dfrac tfrac left right cdot cdots ldots dots text textbf textit mathrm mathbf mathit mathbb "
    "operatorname displaystyle quad qquad begin end align aligned array le leq ge geq ne neq times sqrt "
    "hspace vspace mbox boxed".split()
)


def tokens(text):
    """Lower-cased alphanumeric tokens with thousands separators and LaTeX formatting words removed."""
    return [t for t in re.findall(r"[a-z0-9]+", _THOUSANDS.sub("", text.casefold())) if t not in _DROP]


def _grams(toks, n):
    return {" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def _h(s):
    return hashlib.blake2b(s.encode(), digest_size=8).digest()


# ----------------------------------------------------------------------------- decontamination
MATH_FAMILY = ("math", "gsm8k", "aime", "amc")
_MATH_SPAN = re.compile(r"\$\$(.+?)\$\$|\$(.+?)\$|\\\((.+?)\\\)|\\\[(.+?)\\\]", re.S)


def numbers(text):
    return tuple(sorted(re.findall(r"\d+(?:\.\d+)?", _THOUSANDS.sub("", text))))


def formula_pieces(text):
    """Whitespace/brace-free pieces (>= 10 chars) of the LaTeX math spans, split at '=' and ','."""
    out = set()
    for m in _MATH_SPAN.finditer(text):
        span = next(g for g in m.groups() if g is not None)
        span = re.sub(r"\\(?:left|right|,|;|!|quad|displaystyle)|[\s{}]", "", span)
        out.update(p for p in re.split(r"[=,]|\\(?:le|ge|leq|geq|ne|neq)", span) if len(p) >= 10)
    return out


def containment(a, b):
    """Matched characters / length of the shorter of two normalized token strings."""
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    if not short:
        return 0.0
    sm = difflib.SequenceMatcher(None, short, long_, autojunk=False)
    return sum(m.size for m in sm.get_matching_blocks()) / len(short)


def paraphrase(a, b):
    """Similar-length texts that mostly align (same-numbers rule)."""
    if not a or not b or not 0.5 <= len(a) / len(b) <= 2:
        return False
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return sm.quick_ratio() >= 0.65 and sm.ratio() >= 0.65 and containment(a, b) >= 0.8


class EvalIndex:
    """Evaluation-set matcher complementary to the builders' exact/5-gram Index. A text matches an
    evaluation item if any of:

    * rare 8-grams (8-grams of normalized tokens found in <= 2 evaluation items, so shared
      boilerplate is ignored): >= 6 shared and >= 50% of the item's rare 8-grams, or >= 4 shared,
      >= 25% and normalized containment >= 0.8;
    * the same multiset of >= 2 numbers (thousands separators removed), length ratio within 0.5..2,
      edit ratio >= 0.65 and containment >= 0.8;
    * >= 2 shared LaTeX formula pieces of >= 10 characters that occur in <= 2 evaluation items
      (catches renamed variables, e.g. roots z_i vs alpha..delta of the same polynomials);
    * short math items (6..11 normalized tokens with a number) contained verbatim in a text at most
      3x as long.
    Thresholds were calibrated on 12 reviewer-confirmed paraphrases (all matched) and on the
    false positives of a heldout-rarity-only rule (boilerplate AIME/MATH openings, not matched).
    """

    N = 8

    def __init__(self, items):
        """items: iterable of (name, text); names ending in a MATH_FAMILY file prefix are math items."""
        docs = [(name, text, tokens(text)) for name, text in items]
        df, grams = collections.Counter(), []
        for _, _, toks in docs:
            g = {_h(x) for x in _grams(toks, self.N)}
            grams.append(g)
            df.update(g)
        piece_df = collections.Counter(p for _, text, _ in docs for p in formula_pieces(text))
        self.names = [name for name, _, _ in docs]
        self.joined = [" ".join(toks) for _, _, toks in docs]
        self.rare_count = []
        self.postings = collections.defaultdict(list)
        self.pieces = collections.defaultdict(list)
        self.by_numbers = collections.defaultdict(list)
        short = collections.defaultdict(dict)
        for i, ((name, text, toks), g) in enumerate(zip(docs, grams)):
            rare = [x for x in g if df[x] <= 2]
            self.rare_count.append(len(rare))
            for x in rare:
                self.postings[x].append(i)
            for p in formula_pieces(text):
                if piece_df[p] <= 2:
                    self.pieces[p].append(i)
            nums = numbers(text)
            if nums:
                self.by_numbers[nums].append(i)
            if name.rsplit("/", 1)[-1].startswith(MATH_FAMILY) and nums and 6 <= len(toks) < 12:
                short[len(toks)][_h(" ".join(toks))] = i
        self.short = dict(short)

    def match(self, text):
        """(reason, eval_name) if the text overlaps an evaluation item, else None."""
        toks = tokens(text)
        joined = " ".join(toks)
        hits = collections.Counter(i for g in {_h(g) for g in _grams(toks, self.N)} for i in self.postings.get(g, ()))
        for i, k in hits.items():
            frac = k / max(1, self.rare_count[i])
            if k >= 6 and frac >= 0.5:
                return "eval_rare_8gram", self.names[i]
            if k >= 4 and frac >= 0.25 and containment(joined, self.joined[i]) >= 0.8:
                return "eval_rare_8gram_paraphrase", self.names[i]
        nums = numbers(text)
        for i in self.by_numbers.get(nums, ()) if len(nums) >= 2 else ():
            if paraphrase(joined, self.joined[i]):
                return "eval_same_numbers_paraphrase", self.names[i]
        shared = collections.Counter(i for p in formula_pieces(text) for i in self.pieces.get(p, ()))
        for i, k in shared.items():
            if k >= 2:
                return "eval_shared_formulas", self.names[i]
        for n, table in self.short.items():
            if len(toks) <= 3 * n:
                for j in range(len(toks) - n + 1):
                    i = table.get(_h(" ".join(toks[j:j + n])))
                    if i is not None:
                        return "eval_short_math_contained", self.names[i]
        return None


class NearIndex:
    """Near-duplicate index on normalized tokens: exact normalized match, or 5-gram containment
    >= 0.8 of the shorter text with >= min(8, its 5-gram count) shared (texts >= 8 tokens)."""

    def __init__(self):
        self.exact, self.sizes = set(), []
        self.lookup = collections.defaultdict(list)

    @staticmethod
    def _shingles(toks):
        return {_h(g) for g in _grams(toks, 5)} if len(toks) >= 8 else set()

    def match(self, text):
        toks = tokens(text)
        if _h(" ".join(toks)) in self.exact:
            return True
        ss = self._shingles(toks)
        hits = collections.Counter(i for s in ss for i in self.lookup.get(s, ()))
        return any(n >= min(8, len(ss), self.sizes[i]) and n >= 0.8 * min(len(ss), self.sizes[i])
                   for i, n in hits.items())

    def add(self, text):
        toks = tokens(text)
        self.exact.add(_h(" ".join(toks)))
        ss = self._shingles(toks)
        i = len(self.sizes)
        self.sizes.append(len(ss))
        for s in ss:
            self.lookup[s].append(i)


# ----------------------------------------------------------------------------- safety
NSFW = re.compile(
    r"\b(?:fart(?!her|hest)\w*|poop\w*|diaper\w*|vore|nude|nudity|naked(?!\s+eye)|in the buff|au naturel|erotic\w*|porn\w*|"
    r"nsfw|horny|orgasm\w*|penis|vagina|boobs?|nipples?|cock|pussy|cum|fetish\w*|bdsm|lewd|rape[ds]?|"
    r"raping|incest|hentai|twinks?|busty|sex worker|make love|have sex|sexual(?:ly)?|sexy|seduc\w*|kink\w*|slutty|"
    r"slut|whore|bondage|bukkake|futa\w*|buttocks|butt|booty|thong|lingerie|strip(?:per|tease)|masturbat\w*|"
    r"genital\w*|arousal|aroused|foreplay|onlyfans|milf|hooker|prostitut\w*|pregnan\w* fetish|feet fetish|"
    r"tickl\w* torture|inflation fetish|giantess|hypno\w*|mind control\w*|brainwash\w*|bare ?feet|foot (?:fetish|worship)|"
    r"feet (?:tickl\w*|worship\w*|licking|smell\w*)|trampl\w*|headscissor\w*|leg ?lock\w*|age regress\w*|"
    r"regress(?:ed|es|ing)? (?:in)?to (?:a )?(?:baby|toddler|infant|child)|making out|steamy|\bsex\b)\b", re.I)
MINOR = re.compile(
    r"(?i)\b(?:1[0-7]|[1-9])[- ]?(?:year|yr)s?[- ]old\b|\bteen(?:age[rd]?|s)?\b|\bdora\b|\bpeppa\b|\bschoolgirl\w*\b|"
    r"\bloli\w*\b|\bshota\w*\b|\bchild(?:ren)?\b|\bkids?\b|\bminors?\b|\bunderage\b|\btoddler\w*\b|\bbaby\b|\bboy\b|"
    r"\bgirl\b|\bdaughter\b|\bson\b|\bstudent\b|\bhigh school\w*\b|\bmiddle school\w*\b|\belementary\b")
ROLEPLAY_JAILBREAK = re.compile(
    r"(?i)\{\{char\}\}|\{\{user\}\}|\(\(ooc|always stay in character|never break character|stay in character|"
    r"no (?:content )?(?:filter|filters|restrictions|limits|censorship)|ignore (?:all |any )?(?:the )?(?:previous|prior|above) "
    r"(?:instructions|prompts?)|developer mode|dan mode|\bdo anything now\b|jailbreak\w*|uncensored|unfiltered|"
    r"no ethical|without (?:any )?(?:moral|ethical)|forget (?:all|everything|your) (?:(?:previous|prior) )?(?:instructions|rules|"
    r"guidelines)|no guidelines|without guidelines|abusegpt|\bdan\b,? (?:an?|the) ai|pretend (?:you are|to be) (?:an? )?"
    r"(?:evil|unrestricted|unfiltered)")

# WildChat first-turn moderation thresholds (OpenAI moderation scores / detoxify), deliberately low.
MODERATION_LIMITS = {"sexual": 0.05, "sexual/minors": 0.01, "harassment": 0.3, "hate": 0.2, "violence": 0.5,
                     "self-harm": 0.2, "detox_sexual_explicit": 0.05}


def moderation_key(text):
    """Key of the WildChat moderation map: sha256 of the Index normalization of the first user turn."""
    return hashlib.sha256(" ".join(re.findall(r"\w+", text.casefold())).encode()).hexdigest()


def safety_reason(prompt, answer="", moderation=None):
    """None if acceptable. moderation: the map record for this prompt (WildChat), or None."""
    if moderation is not None:
        if moderation.get("flagged") or moderation.get("toxic"):
            return "unsafe_wildchat_moderation_flagged"
        for k, limit in MODERATION_LIMITS.items():
            if moderation.get(k, 0.0) >= limit:
                return "unsafe_wildchat_moderation_" + k.replace("/", "_")
    if ROLEPLAY_JAILBREAK.search(prompt):
        return "unsafe_roleplay_or_jailbreak_prompt"
    if NSFW.search(prompt):
        return "unsafe_nsfw_minor" if MINOR.search(prompt) else "unsafe_nsfw_prompt"
    if len({m.group(0).lower() for m in NSFW.finditer(answer or "")}) >= 2:
        return "unsafe_nsfw_answer"
    return None


# ----------------------------------------------------------------------------- figures, escaping, code
MISSING_FIGURE = re.compile(
    r"<image>|\[\s*(?:image|img)\s*\]|\{image\}|\bfigures? \d+\b"
    r"|\b(?:following|below|above|attached|given) (?:figure|picture|image|diagram|illustration|graph|chart)\b"
    r"|\bas shown in the (?:figure|picture|image|diagram)\b|statement (?:is available )?as a pdf", re.I)
HTML_ESCAPED = re.compile(r"&(?:lt|gt|amp|quot|apos|nbsp|#\d+|#x[0-9a-f]+);", re.I)


def statement_words(question):
    """English words outside code blocks and sample I/O-looking lines (digits/spaces only)."""
    text = re.sub(r"```.*?```", " ", question, flags=re.S)
    prose = [line for line in text.splitlines() if not re.fullmatch(r"[\s\d\-.,:;+*/()\[\]]*", line)]
    return len(re.findall(r"[A-Za-z]{2,}", "\n".join(prose)))


def starter_kind(starter):
    """'function' if the starter code defines a function or class and parses (with a body added),
    'program' if it is non-empty but defines none (e.g. HackerRank STDIN boilerplate), 'broken' if a
    def/class header does not parse, '' if empty."""
    s = (starter or "").strip("\n")
    if not s.strip():
        return ""
    if not re.search(r"^\s*(?:def|class)\s", s, re.M):
        return "program"
    code = [line for line in s.splitlines() if line.strip() and not line.strip().startswith("#")]
    last = code[-1]
    lead = last[: len(last) - len(last.lstrip())]
    # A header whose body is only a comment ("# code here") gets a placeholder body, indented with
    # the header's own whitespace style (GfG starters use tabs).
    body = lead + ("\t" if "\t" in lead else "    ") + "pass" if last.rstrip().endswith(":") else ""
    for candidate in (s, s + "\n" + body if body else s):
        try:
            ast.parse(candidate)
            return "function"
        except SyntaxError:
            continue
    return "broken"


def code_reason(question, starter=None, *, min_statement_words=20):
    """Defects of a restored competitive-programming question (source text, without our suffixes)."""
    if MISSING_FIGURE.search(question):
        return "missing_figure_or_image_placeholder"
    if HTML_ESCAPED.search(question):
        return "html_escaped_text"
    if statement_words(question) < min_statement_words:
        return "missing_problem_statement"
    if starter is not None and starter_kind(starter) == "broken":
        return "unparseable_starter_code"
    return None


def special_token_reason(ids, special_ids):
    """Reject prompts whose plain text tokenizes to chat/tool special tokens (e.g. a literal <tool_call>)."""
    return "embedded_special_token" if any(i in special_ids for i in ids) else None
