"""EvalScope prompts/extraction with the full EvalPlus 0.3.1 test suite."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
from uuid import uuid4
from .mbpp_extract import extract, PROTOCOL_ID


def prompt(row, benchmark):
    from evalscope.benchmarks.mbppplus.mbppplus_adapter import PROMPT
    if benchmark=='humaneval_plus':
        return ('Read the following function signature and docstring, and fully implement the function described. '
                'Your response should only contain the code for this function.\n'+row['prompt'])
    parts=row['prompt'].strip().strip('"').strip().split('\nassert ',1)
    if len(parts)!=2:raise ValueError('Unexpected MBPP prompt format')
    return PROMPT.format(question=parts[0],tests='assert '+parts[1])


def solution(row, benchmark, text):
    from evalscope.api.messages.utils import parse_content_with_reasoning
    from evalscope.benchmarks.humaneval.humaneval_adapter import HumanevalAdapter
    from evalscope.benchmarks.mbppplus.mbppplus_adapter import MBPPplusAdapter
    final,_=parse_content_with_reasoning(text)
    if benchmark=='humaneval_plus':return row['prompt']+HumanevalAdapter._postprocess(final)
    code=MBPPplusAdapter.postprocess_completion(final).split('if __name__ ==',1)[0]
    return extract(text,code,row['entry_point'])


def run(a, config):
    from openai import OpenAI
    if not a.local_data:
        raise ValueError('Full Plus scoring requires prepared --local-data and complete coverage (no --limit)')
    dataset='humaneval' if a.benchmark=='humaneval_plus' else 'mbpp'
    filename='HumanEvalPlus.jsonl' if dataset=='humaneval' else 'MbppPlus.jsonl'
    rows=[json.loads(x) for x in (a.local_data/filename).read_text().splitlines() if x.strip()]
    expected=164 if dataset=='humaneval' else 378
    if len(rows)!=expected or len({x['task_id'] for x in rows})!=expected:
        raise ValueError('Unexpected full Plus task coverage')
    image_id=subprocess.check_output(['docker','image','inspect','--format','{{.Id}}',a.judge_image],text=True).strip()
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'requested_config.json').write_text(json.dumps(dict(config=config,extraction=PROTOCOL_ID,evalplus='0.3.1',judge_image=image_id),indent=2))
    client=OpenAI(base_url=a.api_url.rsplit('/chat/completions',1)[0],api_key=os.environ.get('EVAL_API_KEY','EMPTY'),max_retries=0,timeout=14400)
    def generate(row):
        message=prompt(row,a.benchmark)
        response=client.chat.completions.create(model=a.model,messages=[dict(role='user',content=message)],
            temperature=.6,top_p=1.,seed=a.seed,max_tokens=a.max_tokens,n=1,
            extra_body=dict(top_k=20,chat_template_kwargs=dict(enable_thinking=True)))
        if len(response.choices)!=1 or response.choices[0].finish_reason not in ('stop','length'):
            raise ValueError('Incomplete generation')
        sample=dict(task_id=row['task_id'],solution=solution(row,a.benchmark,response.choices[0].message.content or ''))
        return dict(prompt=message,response=response.model_dump(),sample=sample)
    samples={}
    with (a.output/'predictions.jsonl').open('x') as f, concurrent.futures.ThreadPoolExecutor(max_workers=a.batch_size) as pool:
        for future in concurrent.futures.as_completed([pool.submit(generate,r) for r in rows]):
            record=future.result();f.write(json.dumps(record)+'\n');f.flush()
            samples[record['sample']['task_id']]=record['sample']
    judge=a.output/'judge';judge.mkdir()
    (judge/'samples.jsonl').write_text(''.join(json.dumps(samples[r['task_id']])+'\n' for r in rows))
    scripts=Path(__file__).resolve().parent
    container='onq-plus-'+uuid4().hex
    command=['docker','run','--rm','--name',container,'--network','none','--read-only','--cap-drop=ALL','--security-opt=no-new-privileges',
        '--pids-limit','256','--memory','8g','--cpus','4','--user',f'{os.getuid()}:{os.getgid()}',
        '-e','OPENBLAS_NUM_THREADS=1','-e','OMP_NUM_THREADS=1','--tmpfs','/tmp:rw,nosuid,nodev,size=2g','--shm-size','1g',
        '-e','HUMANEVAL_OVERRIDE_PATH=/data/HumanEvalPlus.jsonl','-e','MBPP_OVERRIDE_PATH=/data/MbppPlus.jsonl',
        '-v',f'{a.local_data.resolve()}:/data:ro','-v',f'{scripts}:/scripts:ro','-v',f'{judge.resolve()}:/work:rw',
        image_id,'python','/scripts/plus_judge.py','--dataset',dataset]
    try:
        with (a.output/'judge.log').open('x') as log:
            subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=14400)
    except BaseException:
        # The Docker client exiting does not terminate its container.
        subprocess.run(['docker','rm','-f',container],check=False,
                       stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=30)
        raise
    score=json.loads((judge/'JUDGE_PASS.json').read_text())
    if score['total']!=expected:raise ValueError('Scoring coverage mismatch')
    (a.output/'RUN_COMPLETE.json').write_text(json.dumps(dict(status='RUN_COMPLETE',score=score),indent=2))
