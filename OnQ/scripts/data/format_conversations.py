"""Format original dataset records into prompts and calibration conversations without rewriting content.

Original-text extraction and rendering. Source text is never rewritten.

* Selection (dedup, rules, decontamination) uses ``prompt_messages``: the
  original problem fields exactly as published.
* Outputs use ``output_prompt`` / ``output_dialogue``; they differ only for
  open-r1/codeforces, rendered in the dataset authors' official
  ``verifiable-prompts`` layout without its answer-instruction preamble.
* No system prompt, suffix, answer instruction or language requirement is added."""

import copy
import hashlib

from OnQ.scripts.data.dataset_sources import digest

KINDS = ('math', 'science', 'code', 'chat', 'if')
ANSWER_FIELD = {'math': 'generated_solution', 'science': 'output', 'code': 'r1_generation'}


def original_code_fields(dataset, ref):
    """Non-empty original problem fields of an OCR2 reference, in source order."""
    keys = {'taco': ('question', 'starter_code'), 'apps': ('question', 'starter_code'),
            'code_contests': ('name', 'description', 'input_file', 'output_file'),
            'open-r1/codeforces': ('title', 'description', 'input_format', 'output_format',
                                   'interaction_format')}.get(dataset)
    if keys is None:
        raise ValueError(f'Unsupported original code dataset: {dataset}')
    values = [(k, ref[k]) for k in keys if isinstance(ref.get(k), str) and ref[k]]
    if dataset == 'open-r1/codeforces':
        for i, example in enumerate(ref.get('examples') or []):
            values += [(f'examples.{i}.{k}', example[k]) for k in ('input', 'output')
                       if isinstance(example.get(k), str) and example[k]]
        if isinstance(ref.get('note'), str) and ref['note']:
            values.append(('note', ref['note']))
    if not any(k in ('question', 'description') for k, _ in values):
        raise ValueError('Original record has no question/description')
    return values


def code_statement(dataset, ref):
    """Problem statement body used for cross-dataset code identity."""
    return (ref.get('question') if dataset in ('taco', 'apps') else ref.get('description')) or ''


def prompt_messages(kind, row, ref=None):
    """Original request messages (selection view). Chat/IF keep native turns up to the last user."""
    if kind in ('chat', 'if'):
        messages = row['messages']
        last_user = max((i for i, m in enumerate(messages) if m['role'] == 'user'), default=-1)
        if last_user < 0:
            raise ValueError('No user message')
        result = copy.deepcopy(messages[:last_user + 1])
        if any(not isinstance(m.get('content'), str) for m in result):
            raise ValueError('Non-text message')
        return result
    if kind == 'math':
        text = row['problem']
    elif kind == 'science':
        text = row['input']
    elif kind == 'code':
        if ref is None:
            raise ValueError('OCR2 original reference required')
        text = '\n\n'.join(v for _, v in original_code_fields(row['dataset'], ref))
    else:
        raise ValueError(kind)
    if not isinstance(text, str) or not text:
        raise ValueError('Empty original request')
    return [dict(role='user', content=text)]


def render_codeforces(ref):
    """open-r1/codeforces official verifiable-prompts body without the answer preamble.

    Empty sections are omitted; ``## Interaction Format`` is a documented extension
    (the official verifiable subset has no interactive tasks). Values are verbatim.
    """
    if not isinstance(ref.get('description'), str) or not ref['description']:
        raise ValueError('Missing Codeforces description')
    limits = [f'{label}: {ref[key]} {unit}' for key, label, unit in
              (('time_limit', 'Execution time limit', 'seconds'), ('memory_limit', 'Memory limit', 'MB'))
              if ref.get(key) not in (None, '')]
    parts = ['\n'.join(limits)] if limits else []
    for key, heading in (('description', '# Problem'), ('input_format', '## Input Format'),
                         ('output_format', '## Output Format'),
                         ('interaction_format', '## Interaction Format')):
        if ref.get(key) not in (None, ''):
            parts.append(f'{heading}\n{ref[key]}')
    examples = []
    for example in ref.get('examples') or []:
        blocks = [f'```{k}\n{example[k]}\n```' for k in ('input', 'output') if example.get(k) is not None]
        if blocks:
            examples.append('\n'.join(blocks))
    if examples:
        parts.append('## Examples\n' + '\n-----\n'.join(examples))
    if ref.get('note'):
        parts.append('## Note\n' + ref['note'])
    return '\n\n'.join(parts)


def output_prompt(kind, row, ref=None):
    """Prompt messages written to the OPD dataset."""
    if kind == 'code' and row['dataset'] == 'open-r1/codeforces':
        return [dict(role='user', content=render_codeforces(ref))]
    return prompt_messages(kind, row, ref)


def output_dialogue(kind, row, ref=None):
    """Full original conversation (request + original answer) used for Block-AP calibration."""
    if kind in ('chat', 'if'):
        prompt_messages(kind, row)  # validates a native user request exists
        return copy.deepcopy(row['messages'])
    answer = row[ANSWER_FIELD[kind]]
    if not isinstance(answer, str) or not answer:
        raise ValueError('Missing original answer')
    return output_prompt(kind, row, ref) + [dict(role='assistant', content=answer)]


def text_of(messages, roles=('system', 'user')):
    return '\n'.join(m['content'] for m in messages if m['role'] in roles)


def prompt_tokens(tokenizer, messages):
    """Prompt length as the OPD rollout sees it (Qwen3 template, generation prompt, thinking on)."""
    return len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                             enable_thinking=True, return_dict=False))


def official_render(tokenizer, messages):
    """Pinned Qwen3 ``apply_chat_template(messages, tokenize=False)``, default arguments."""
    for m in messages:
        if m.get('role') not in ('system', 'user', 'assistant') or not isinstance(m.get('content'), str):
            raise ValueError('Unsupported message for text-only rendering')
        if m.get('tool_calls') or m.get('reasoning_content'):
            raise ValueError('Structured message fields would be lost')
    return tokenizer.apply_chat_template(messages, tokenize=False)


def template_audit(messages, rendered):
    """Prove the template changed formatting only.

    Allowed: newline normalization around ``<think>``/``</think>`` and an empty think
    block on a final non-reasoning assistant turn. The template drops reasoning of
    assistant turns before the last user turn; that is reported per record. Any
    other text difference raises ``ValueError`` (the record is then excluded).
    """
    last_user = max((i for i, m in enumerate(messages) if m['role'] == 'user'), default=-1)
    parts, historical, empty_think = [], [], 0
    for i, m in enumerate(messages):
        role, content = m['role'], m['content']
        body = content
        if role == 'assistant':
            reasoning, answer = '', content
            if '</think>' in content:
                if content.count('</think>') != 1 or content.count('<think>') > 1:
                    raise ValueError('Multiple think blocks')
                prefix, answer = content.split('</think>')
                if '<think>' in prefix:
                    before, prefix = prefix.split('<think>')
                    if before.strip('\n'):
                        raise ValueError('Text before <think> would be dropped')
                reasoning, answer = prefix.strip('\n'), answer.lstrip('\n')
            if i <= last_user:
                body = answer
                if '</think>' in content:
                    historical.append(dict(message_index=i, removed_reasoning_sha256=hashlib.sha256(
                        reasoning.encode()).hexdigest()))
            elif i == len(messages) - 1 or reasoning:
                body = f'<think>\n{reasoning}\n</think>\n\n{answer.lstrip(chr(10))}'
                empty_think += not reasoning
            else:
                body = answer
        parts.append(f'<|im_start|>{role}\n{body}<|im_end|>\n')
    if ''.join(parts) != rendered:
        raise ValueError('Template output differs beyond audited formatting')
    return dict(historical_reasoning=historical, empty_think_blocks=empty_think,
                content_sha256=digest(messages))
