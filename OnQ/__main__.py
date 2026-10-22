"""Four workflows: preparedata, calibration, opd, evaluation."""
import argparse
import runpy
import sys


ROUTES = {
    'preparedata': {
        'benchmarks': 'scripts.data.prepare_benchmark_texts',
        'build': 'scripts.data.prepare_training_data',
        'check': 'scripts.data.check_data_overlap',
        'prompts': 'integrations.verl_v080.opd.prompt_data',
    },
    'opd': {
        'teacher': 'deployment.teacher',
        'serve': 'integrations.verl_v080.opd.union_teacher_vllm',
        'prepare': 'cli.opd',
        'train': 'integrations.verl_v080.utils.supervise',
        'resume': 'integrations.verl_v080.opd.prepare_dynamic_resume',
    },
    'evaluation': {
        'export': 'quantization.export',
        'run': 'evaluation.evalscope.run',
        'aime': 'evaluation.evalscope.prepare_aime90',
        'native': 'evaluation.evalscope.prepare_native',
        'plus': 'evaluation.evalscope.prepare_plus',
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workflow', choices=('preparedata', 'calibration', 'opd', 'evaluation'))
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    rest = args.arguments
    if args.workflow == 'calibration':
        module = 'cli.train'
        rest = ['--stage', 'calibration', *rest]
    else:
        actions = ROUTES[args.workflow]
        action_parser = argparse.ArgumentParser(prog=f'{parser.prog} {args.workflow}')
        action_parser.add_argument('action', choices=actions)
        action_parser.add_argument('arguments', nargs=argparse.REMAINDER)
        selected = action_parser.parse_args(rest)
        module = actions[selected.action]
        rest = selected.arguments
    sys.argv = [f'{parser.prog} {args.workflow}', *rest]
    runpy.run_module('OnQ.' + module, run_name='__main__')


if __name__ == '__main__':
    main()
