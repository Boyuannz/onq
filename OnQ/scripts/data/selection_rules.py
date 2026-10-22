"""Apply whole-record eligibility rules for calibration and OPD data; imported by the data builder.

Automatic whole-record selection rules. A rule only decides membership; it never edits text.

Rejected records are replaced by the next admissible record in the same seeded
random order, so quotas stay fixed. Rules come from the data selection policy:

both stages   non-English, multiple choice without options, chat control tokens,
              sexual content involving minors (recall rule, chat/IF sources),
              near-duplicate questions
OPD only      single-turn requests, prompt <= max tokens (in prepare_training_data.py), inline image
              placeholders in code, unusable code statements, IF constraint
              conflicts, answer given in a science prompt, trivial/spam/link
              WildChat requests

The OnQ release applied the recall rules (minors, spam, link, unusable code,
answer-in-prompt) through per-record human adjudication. Here every recall hit is
excluded automatically, which over-excludes (OnQ confirmed 21/153 minors,
2/17 spam, 6/38 link and 40/56 unusable-code hits) but needs no reviewer."""

import ast
import hashlib
import itertools
import re
import sys
import types
import urllib.request
from pathlib import Path

from langdetect import DetectorFactory, detect_langs

DetectorFactory.seed = 0  # deterministic langdetect

# ---------------------------------------------------------------- language / missing options (integrity rules)
NON_LATIN = re.compile(
    '[Ѐ-ԯ԰-֏֐-׿؀-ۿ܀-ݏݐ-ݿހ-޿'
    'ࢠ-ࣿऀ-෿฀-໿ༀ-࿿က-႟Ⴀ-ჿᄀ-ᇿ'
    'ሀ-᎟Ꭰ-᏿ក-៿᠀-᢯⺀-⿟぀-ヿ㄀-ㇿ'
    '㐀-鿿ꀀ-꓏가-힯豈-﫿]')


def prose(text):
    """Text for language identification: code blocks, inline code and LaTeX removed."""
    text = re.sub(r'```.*?```', ' ', text, flags=re.S)
    text = re.sub(r'`[^`]*`', ' ', text)
    return re.sub(r'\$\$.*?\$\$|\$[^$]*\$|\\\(.*?\\\)|\\\[.*?\\\]', ' ', text, flags=re.S)


def non_english(text):
    """True for non-Latin script, or confident (p >= 0.99, >= 8 words) non-English prose."""
    if NON_LATIN.search(text):
        return True
    body = prose(text)
    words = re.findall(r'[A-Za-z]{2,}', body)
    if len(words) < 8:
        return False
    try:
        top = detect_langs(body)[0]
    except Exception:  # undetectable text is uncertain, not rejected
        return False
    return top.lang != 'en' and top.prob >= 0.99


def missing_options(text):
    """Asks to choose among options, but fewer than two labelled options are present."""
    if re.search(r'\b(?:which|what)\s+(?:one\s+)?of\s+the\s+following\b|\b(?:choose|select)\b.{0,30}'
                 r'\b(?:options?|choices?)\s+(?:below|following)\b', text, re.I):
        labels = re.findall(r'(?:^|\n|\s)(?:\(([A-Ja-j1-9])\)|([A-Ja-j1-9])[.):])\s*\S', text)
        bullets = re.findall(r'^\s*[-*•]\s+\S', text, re.M)
        return len({a or b for a, b in labels}) < 2 and len(bullets) < 2
    return False


def admission(prompt, dialogue=None):
    """'missing_choice_candidates' / 'non_english' / None. Block-AP passes the whole dialogue."""
    if any(m['role'] == 'user' and missing_options(m['content']) for m in prompt):
        return 'missing_choice_candidates'
    if any(non_english(m['content']) for m in (dialogue or prompt)):
        return 'non_english'
    return None


def single_turn(messages):
    """Exactly one user turn, optionally preceded only by system messages."""
    users = [i for i, m in enumerate(messages) if m['role'] == 'user']
    return len(users) == 1 and all(m['role'] == 'system' for m in messages[:users[0]])


CONTROL = re.compile(r'<\|(?:im_start|im_end|endoftext)\|>')

# ---------------------------------------------------------------- IF constraint conflicts
IFEVALG_URL = ('https://raw.githubusercontent.com/allenai/open-instruct/'
               '99b1ee970490a2d0d5664663eb0b1acc410b944c/open_instruct/IFEvalG/instructions_registry.py')
IFEVALG_SHA256 = '5436911d616a7c38afcbbc18312551d8a9ed3f06bca44d74920071c1690c7fc3'


def fetch_ifevalg(cache_root):
    """Pinned open-instruct IFEvalG registry (Apache-2.0), downloaded once and hash-checked."""
    path = Path(cache_root) / 'ifevalg' / 'instructions_registry.py'
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(IFEVALG_URL, timeout=60) as r:
            path.write_bytes(r.read())
    if hashlib.sha256(path.read_bytes()).hexdigest() != IFEVALG_SHA256:
        raise ValueError(f'IFEvalG registry hash mismatch: {path}')
    return path


def ifevalg_conflicts(source_text):
    """INSTRUCTION_CONFLICTS symmetrized by the registry's own conflict_make (checkers stubbed)."""
    stub = types.ModuleType('open_instruct.IFEvalG.instructions')
    stub.__getattr__ = lambda name: name
    pkg, sub = types.ModuleType('open_instruct'), types.ModuleType('open_instruct.IFEvalG')
    sub.instructions = stub
    names = ('open_instruct', 'open_instruct.IFEvalG', 'open_instruct.IFEvalG.instructions')
    saved = {k: sys.modules.get(k) for k in names}
    sys.modules.update(zip(names, (pkg, sub, stub)))
    try:
        ns = {}
        exec(compile(source_text, 'instructions_registry.py', 'exec'), ns)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return ns['conflict_make'](ns['INSTRUCTION_CONFLICTS'])


def strip_constraints(prompt, constraint):
    words = constraint.split()
    m = re.search(r'\s+'.join(map(re.escape, words)) + r'\s*$', prompt) if words else None
    return prompt[:m.start()].strip() if m else None


YES_NO_QUESTION = re.compile(r'(?:^|[.!?\n]\s*)(?:is|are|was|were|do|does|did|can|could|will|would|should|'
                             r'has|have|had|may|might|must)\b[^.!?\n]*\?\s*$', re.I)
PARAGRAPH_IDS = {'length_constraints:number_paragraphs', 'paragraphs:paragraphs', 'paragraphs:paragraphs2',
                 'length_constraints:nth_paragraph_first_word'}
STRUCTURE_IDS = PARAGRAPH_IDS | {'length_constraints:number_sentences', 'length_constraints:number_words',
                                 'detectable_format:number_bullet_lists', 'detectable_format:multiple_sections'}


def semantic_conflict(ids, constraint):
    """Constraint pairs that cannot be satisfied together (if_semantic_conflict)."""
    ids = set(ids)
    if len(ids & PARAGRAPH_IDS) > 1:
        return 'multiple_paragraph_constraints'
    repeat = ('count:count_increment_word' in ids or 'copy:repeat_phrase' in ids or any(
        int(n) >= 2 for n in re.findall(r'the word \S+ should appear (?:at least )?(\d+) times', constraint)))
    if 'count:count_unique' in ids and repeat:
        return 'unique_words_vs_repetition'
    if re.search(r'all lowercase words should appear at most \d+ times', constraint) and repeat:
        return 'lowercase_cap_vs_repetition'
    if 'change_case:english_lowercase' in ids and re.search(r'capital letters should appear at least [1-9]',
                                                             constraint):
        return 'lowercase_vs_capital_words'
    if 'change_case:english_capital' in ids and re.search(r'capital letters should appear less than', constraint):
        return 'all_caps_vs_capital_cap'
    if 'punctuation:no_comma' in ids and 'detectable_format:json_format' in ids:
        return 'no_comma_vs_json'
    if 'count:counting_composition' in ids and ids & STRUCTURE_IDS:
        return 'counting_composition_vs_structure'
    if 'combination:repeat_prompt' in ids and ids & {'change_case:english_lowercase', 'change_case:english_capital',
                                                     'detectable_format:square_brackets', 'punctuation:no_comma'}:
        return 'repeat_prompt_vs_format'
    m = re.search(r'Answer with less than (\d+) letters', constraint)
    if m and int(m.group(1)) < 100:
        return 'tiny_total_letter_cap'
    n_par = [int(x) for x in re.findall(r'There should be (\d+) paragraphs', constraint)]
    nth = [int(a or b) for a, b in re.findall(r'The (\d+)-th paragraph|Paragraph (\d+) must start', constraint)]
    if n_par and nth and max(nth) > min(n_par):
        return 'nth_paragraph_beyond_count'
    sections = [int(x) for x in re.findall(r'must have (\d+) sections', constraint)]
    if sections and n_par and max(sections) > min(n_par):
        return 'sections_exceed_paragraphs'
    m = re.search(r'letter (\w) should appear less than (\d+) times', constraint)
    if m and m.group(1).lower() in 'etaoinshrdlu' and int(m.group(2)) <= 10:
        return 'common_letter_cap'
    return None


def if_conflict(row, conflicts):
    """Conflict name for an IF_multi row, from its own ground_truth/constraint metadata."""
    try:
        ids = ast.literal_eval(row['ground_truth'])[0]['instruction_id']
    except Exception:  # metadata cannot be judged: keep
        return None
    if len(set(ids)) != len(ids):
        return 'duplicate_instruction'
    if all(x in conflicts for x in ids) and any(b in conflicts[a] for a, b in itertools.combinations(ids, 2)):
        return 'ifevalg_table'
    semantic = semantic_conflict(ids, row.get('constraint') or '')
    if semantic:
        return semantic
    if 'detectable_format:constrained_response' in ids:
        users = [m['content'] for m in row['messages'] if m['role'] == 'user']
        base = strip_constraints(users[-1], row.get('constraint') or '') if users else None
        if base and not YES_NO_QUESTION.search(base):
            return 'yes_no_answer_for_open_task'
    return None


# ---------------------------------------------------------------- code statements
IMAGE = re.compile(r'<image>|<img\b[^>]*>', re.I)
HEADING = re.compile(r'^\s*(?:examples?|sample\s*(?:input|output)s?|input|output|note|-+|time limit.*|'
                     r'memory limit.*|execution time limit.*)\s*:?\s*$', re.I)


def code_problem(text, dataset, ref):
    """Reason a code statement cannot be solved from its text, else None."""
    if any(IMAGE.search(line) and re.search(r'[A-Za-z0-9]', IMAGE.sub('', line)) for line in text.splitlines()):
        return 'code_inline_image_placeholder'  # the placeholder replaces part of a sentence/formula
    body = '\n'.join(line for line in text.splitlines() if not HEADING.match(line))
    if len(re.findall(r'[A-Za-z]{2,}', body)) < 3:
        return 'code_statement_example_only'
    if 'Q#' in text:
        return 'code_statement_qsharp'
    if re.search(r'following signature:\s*$', text.rstrip()):
        return 'code_statement_missing_signature'
    if dataset == 'open-r1/codeforces' and not (ref.get('input_format') or ref.get('output_format')
                                                  or ref.get('examples')):
        return 'code_statement_not_programming'
    # Recall rule (adjudicated in OnQ): statement only as link/PDF, or almost no prose.
    if (re.search(r'statement\s+(?:as\s+a\s+)?(?:here|pdf)|view the statement|same as the easy version', text, re.I)
            or len(re.findall(r'[A-Za-z]{2,}', re.sub(r'https?://\S+', ' ', text))) < 25):
        return 'code_statement_unusable'
    return None


# ---------------------------------------------------------------- recall rules
MINOR = re.compile(
    r'\b(?:1[0-7]|[5-9]|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen)[- ]?(?:years?|yrs?|yo)\b'
    r'(?:[- ]old)?|\b(?:minors?|underage|under-age|preteens?|pre-teens?|teen|teens|teenage|teenager|teenagers'
    r'|schoolgirls?|schoolboys?|middle[- ]school|junior high|high[- ]school(?:er|ers| student| students| girl'
    r'| boy)|loli|shota|child|children|kids?)\b', re.I)
SEXUAL = re.compile(
    r'\b(?:sex|sexy|sexual|sexually|nude|nudity|naked|erotic|erotica|fetish|porn|pornographic|aroused|arousal'
    r'|orgasm|breasts?|genitals?|penis|vagina|intimate|seduc\w*|lingerie|underwear|panties|masturbat\w*|rape'
    r'|molest\w*|grope\w*|fondl\w*|strip(?:ped|ping)?|kink\w*|lewd|horny|make out|making out|hypnoti[sz]\w*'
    r'|diaper\w*|spank\w*)\b', re.I)
SPAM = re.compile(r'\b(?:casino|slot|slots|gacor|togel|judi|kubet|poker|jackpot|betting|bet88|maxwin|'
                  r'deposit bonus|link alternatif|whatsapp|telegram|promo code|discount code|buy now|backlinks?|'
                  r'escort)\b', re.I)
URL = re.compile(r'https?://|www\.', re.I)
ANSWER_IN_PROMPT = re.compile(r'\n\s*(?:Answer|Correct answer|Correct option|The answer is|Ans)\s*[:：.]'
                              r'|\n\s*\(?[A-J]\)?\s*$', re.I)


def minors_sexual(texts):
    joined = '\n'.join(texts)
    return bool(MINOR.search(joined) and SEXUAL.search(joined))


def opd_reason(kind, row, prompt, ref, conflicts, wildchat=False):
    """OPD-only rules on the original request; ``wildchat`` enables the WildChat rules."""
    text = '\n'.join(m['content'] for m in prompt)
    if kind == 'code':
        return code_problem(text, row['dataset'], ref)
    if kind == 'if':
        c = if_conflict(row, conflicts)
        return c and f'if_constraint_conflict:{c}'
    if kind == 'science' and ANSWER_IN_PROMPT.search(text):
        return 'answer_in_prompt'
    if wildchat:
        if len(text.split()) <= 3:
            return 'trivial_prompt'
        if SPAM.search(text):
            return 'spam_or_ads'
        if URL.search(text):
            return 'requires_external_link'
    return None


def common_reason(kind, texts):
    """Rules shared by both stages, over every message the stage consumes.

    The minors recall rule runs on chat and IF sources only: all 27 OnQ recall hits in
    code/science problems were adjudicated false positives (tree ``child`` nodes, ``strip()``).
    """
    if any(CONTROL.search(t) for t in texts):
        return 'native_chat_control_tokens'
    if kind in ('chat', 'if') and minors_sexual(texts):
        return 'sexual_content_involving_minors'
    return None


# ---------------------------------------------------------------- near-duplicate questions
VERSION = re.compile(r'\b(easy|easier|hard|harder)\s+version\b', re.I)


class NearDuplicates:
    """Word 5-gram containment >= threshold against the shorter text; texts with fewer than
    ``min_grams`` 5-grams match only an identical word sequence.

    Easy/hard versions of one problem are distinct tasks. r11 used this for OCR2
    cross-dataset identity; here it applies to every question within a stage.
    """

    def __init__(self, threshold=0.8, min_grams=20):
        self.threshold, self.min_grams = threshold, min_grams
        self.items, self.index, self.exact = [], {}, {}

    @staticmethod
    def _grams(text):
        words = tuple(re.findall(r'[a-z0-9]+', text.lower()))
        return words, {' '.join(words[i:i + 5]) for i in range(len(words) - 4)}

    @staticmethod
    def _version(text):
        m = VERSION.search(text)
        return m and ('easy' if m.group(1).lower().startswith('eas') else 'hard')

    def match(self, text):
        """Index of an earlier near-duplicate, else None."""
        words, grams = self._grams(text)
        if not words:  # no alphanumeric words: only exact normalized dedup applies
            return None
        if len(grams) < self.min_grams:
            return self.exact.get(words)
        tag, shared = self._version(text), {}
        for g in grams:
            for j in self.index.get(g, ()):
                shared[j] = shared.get(j, 0) + 1
        for j, k in shared.items():
            size, other_tag = self.items[j]
            if k / min(len(grams), size) >= self.threshold and not (tag and other_tag and tag != other_tag):
                return j
        return None

    def add(self, text):
        words, grams = self._grams(text)
        j = len(self.items)
        self.items.append((len(grams), self._version(text)))
        if not words:
            return j
        if len(grams) < self.min_grams:
            self.exact.setdefault(words, j)
        else:
            for g in grams:
                self.index.setdefault(g, []).append(j)
        return j
