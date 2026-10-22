"""Keep the paper's native LiveCodeBench scorer inside no-network containers."""
import fcntl
import json
import os
from contextlib import ExitStack
from pathlib import Path
import subprocess
import time
from uuid import uuid4


class JudgeInfrastructureError(BaseException):
    """Abort evaluation; infrastructure failures must never count as wrong answers."""


def stop_container(name):
    try:
        subprocess.run(['docker', 'rm', '-f', name], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
    except Exception as exc:
        raise JudgeInfrastructureError(f'Cannot stop LiveCodeBench container: {name}') from exc


def install(output, image):
    import evalscope.benchmarks.live_code_bench.evaluate_utils as evaluation
    image_id = subprocess.check_output(
        ['docker', 'image', 'inspect', '--format', '{{.Id}}', image], text=True).strip()
    scripts = Path(__file__).resolve().parent

    def jailed_check(sample, generation, timeout, debug=True):
        nonce = uuid4().hex
        work = Path(output).resolve() / 'judge' / nonce
        work.mkdir(parents=True)
        count = len(json.loads(sample['input_output'])['inputs'])
        request = dict(sample=sample, generation=generation, timeout=timeout, debug=debug, nonce=nonce)
        (work / 'request.json').write_text(json.dumps(request))
        container = 'onq-lcb-' + nonce
        command = ['docker', 'run', '--rm', '--name', container, '--network', 'none',
            '--read-only', '--cap-drop=ALL', '--security-opt=no-new-privileges',
            '--pids-limit', '256', '--memory', '8g', '--cpus', '4',
            '--user', f'{os.getuid()}:{os.getgid()}', '--tmpfs', '/tmp:rw,nosuid,nodev,size=2g',
            '-v', f'{scripts}:/scripts:ro', '-v', f'{work}:/work:rw',
            image_id, 'python', '/scripts/lcb_judge.py']
        # Match the original eight process-wide CPU slots, independently of API batch size.
        with ExitStack() as stack:
            locks = [stack.enter_context((work.parent / f'cpu_slot{i}.lock').open('a')) for i in range(8)]
            chosen = None
            while chosen is None:
                for lock in locks:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        chosen = lock
                        break
                    except BlockingIOError:
                        pass
                if chosen is None:
                    time.sleep(.1)
            try:
                with (work / 'judge.log').open('x') as log:
                    subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
                                   timeout=(timeout + 1) * count + 30)
                result = json.loads((work / 'result.json').read_text())
                if not isinstance(result['result'], list) or not result['result']:
                    raise ValueError('Missing per-test results')
                return result['result'], result['metadata']
            except subprocess.TimeoutExpired as exc:
                # Killing the Docker client alone does not stop the container.
                stop_container(container)
                if 'JUDGE_START_' + nonce not in (work / 'judge.log').read_text():
                    raise JudgeInfrastructureError(f'LiveCodeBench judge never started: {work}') from exc
                raise TimeoutError('official global code timeout') from exc
            except Exception as exc:
                raise JudgeInfrastructureError(f'LiveCodeBench judge failed; inspect {work}') from exc
            except BaseException:
                stop_container(container)
                raise

    # Preserve EvalScope's original aggregation, metadata and timeout semantics.
    evaluation.codegen_check_correctness = jailed_check
    return image_id
