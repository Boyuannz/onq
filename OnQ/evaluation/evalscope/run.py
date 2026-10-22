"""Run EvalScope 1.5.1 against an OpenAI-compatible model endpoint."""
import argparse
import json
import os
from contextlib import nullcontext
from importlib.metadata import version
from pathlib import Path

VERSION = '1.5.1'
BENCHMARKS = {
    'gsm8k': 'gsm8k', 'math500': 'math_500', 'aime90': 'aime24',
    'humaneval_plus': 'humaneval_plus', 'mbpp_plus': 'mbpp_plus',
    'livecodebench': 'live_code_bench', 'mmlu_redux': 'mmlu_redux',
    'gpqa': 'gpqa_diamond', 'ifeval': 'ifeval',
}
CODE = {'humaneval_plus', 'mbpp_plus', 'livecodebench'}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lcb-judge-image', default='onq-lcb:1.5.1')
    p.add_argument('--judge-image', default='onq-evalplus:0.3.1', help='Locally built isolated full-Plus scorer image')
    p.add_argument('--model', required=True, help='Served model name')
    p.add_argument('--api-url', required=True, help='Full /v1/chat/completions endpoint')
    p.add_argument('--benchmark', choices=BENCHMARKS, required=True)
    p.add_argument('--output', type=Path, required=True, help='New output directory')
    p.add_argument('--max-tokens', type=int, choices=(8192, 32768), default=32768, help='Record the intended row cap explicitly; main-table rows include exceptions')
    p.add_argument('--seed', type=int, default=42, help='Generation seed; dataset seed stays 42')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--dataset-args', type=Path, help='JSON mapping passed to this dataset adapter')
    p.add_argument('--local-data', type=Path, help='Prepared EvalScope JSONL directory')
    p.add_argument('--dry-run', action='store_true', help='Print configuration without imports, downloads or requests')
    return p


def configuration(a):
    if a.batch_size <= 0 or a.seed < 0:
        raise ValueError('Batch size must be positive and seed nonnegative')
    if not a.api_url.rstrip('/').endswith('/v1/chat/completions'):
        raise ValueError('Use a full /v1/chat/completions URL')
    options = json.loads(a.dataset_args.read_text()) if a.dataset_args else {}
    if not isinstance(options, dict):
        raise ValueError('dataset-args must contain a JSON object')
    if a.local_data:
        if not a.local_data.is_dir():
            raise ValueError('local-data must be an existing prepared dataset directory')
        options['dataset_id'] = str(a.local_data.resolve())
    options.setdefault('few_shot_num', 0)
    if a.benchmark == 'aime90':
        if not a.local_data:
            raise ValueError('aime90 requires --local-data prepared with prepare_aime90')
        rows = [json.loads(x) for x in (a.local_data/'default_test.jsonl').read_text().splitlines() if x.strip()]
        from .prepare_aime90 import validate
        validate(rows)
        options['subset_list'] = ['default']
    if a.benchmark == 'livecodebench':
        extra = options.get('extra_params', {})
        if not a.local_data and not (extra.get('start_date') and extra.get('end_date')):
            raise ValueError('Pin LiveCodeBench via local-data or explicit start_date/end_date')
    bench = BENCHMARKS[a.benchmark]
    return dict(model=a.model, model_id=a.model, eval_type='openai_api', api_url=a.api_url,
        model_args={'max_retries': 0}, datasets=[bench], dataset_args={bench: options},
        generation_config=dict(temperature=.6, top_k=20, top_p=1., seed=a.seed,
            max_tokens=a.max_tokens, n=1, timeout=14400., retries=1, stream=False,
            extra_body={'chat_template_kwargs': {'enable_thinking': True}}),
        seed=42, repeats=1, eval_batch_size=a.batch_size, ignore_errors=False,
        limit=None, no_timestamp=True, work_dir=str(a.output.resolve()),
        use_sandbox=a.benchmark in CODE - {'livecodebench'}, sandbox_type='docker')


def local_adapters():
    """Use local JSONL files with the same loaders as the evaluation runs."""
    from evalscope.api.benchmark import DefaultDataAdapter
    from evalscope.benchmarks.aime.aime_adapter import AIME24Adapter
    from evalscope.benchmarks.gsm8k.gsm8k_adapter import GSM8KAdapter
    from evalscope.benchmarks.gpqa.gpqa_adapter import GPQAAdapter
    from evalscope.benchmarks.mmlu_redux.mmlu_redux_adapter import MMLUReduxAdapter
    from evalscope.benchmarks.live_code_bench.live_code_bench_adapter import LiveCodeBenchAdapter
    for cls in (AIME24Adapter, GSM8KAdapter, GPQAAdapter, MMLUReduxAdapter, LiveCodeBenchAdapter):
        cls.load_from_disk = lambda self: DefaultDataAdapter.load_from_disk(self, use_local_loader=True)
    original = MMLUReduxAdapter.record_to_sample
    def redux(self, record):
        if record['error_type'] == 'multiple_correct_answers' and type(record['correct_answer']) is int and record['correct_answer']:
            record = dict(record, correct_answer=str(record['correct_answer']))
        return original(self, record)
    MMLUReduxAdapter.record_to_sample = redux


def main(argv=None):
    a = parser().parse_args(argv)
    config = configuration(a)
    if a.dry_run:
        print(json.dumps(config, indent=2))
        return
    if version('evalscope') != VERSION:
        raise RuntimeError(f'Install evalscope=={VERSION}; adapter versions affect scores')
    resources = None
    inputs = None
    if a.benchmark == 'ifeval':
        from .resources import check_ifeval_resources
        from .ifeval import input_records, strict_scorer, audit_coverage
        resources = check_ifeval_resources()
        inputs = input_records(a.local_data)
    if a.benchmark in {'humaneval_plus', 'mbpp_plus'}:
        from .plus import run
        run(a, config)
        return
    from evalscope import TaskConfig, run_task
    if a.local_data:
        local_adapters()
    if a.benchmark == 'livecodebench':
        from .lcb import install
        install(a.output, a.lcb_judge_image)
    task = TaskConfig(**config, api_key=os.environ.get('EVAL_API_KEY', 'EMPTY'))
    a.output.mkdir(parents=True, exist_ok=False)
    record = dict(evalscope=VERSION, benchmark=a.benchmark, config=config,
        scope='evalscope-native')
    (a.output/'requested_config.json').write_text(json.dumps(record, indent=2)+'\n')
    if resources is not None:
        (a.output/'SCORER_RESOURCES.json').write_text(json.dumps(resources, indent=2)+'\n')
    with strict_scorer() if a.benchmark == 'ifeval' else nullcontext():
        run_task(task)
    if a.benchmark == 'ifeval':
        coverage = audit_coverage(a.output, inputs)
        (a.output/'SCORING_COVERAGE.json').write_text(json.dumps(coverage, indent=2)+'\n')
    (a.output/'RUN_COMPLETE.json').write_text(json.dumps({
        'status': 'RUN_COMPLETE'}, indent=2)+'\n')


if __name__ == '__main__':
    main()
