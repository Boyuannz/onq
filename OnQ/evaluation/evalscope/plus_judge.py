"""Container-only full EvalPlus scorer; generated code must not run on the host."""
import json
import os
from pathlib import Path
from evalplus.evaluate import evaluate

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['humaneval','mbpp'],required=True)
    a=p.parse_args()
    os.chdir('/work')
    evaluate(dataset=a.dataset,samples='/work/samples.jsonl',base_only=False,parallel=8,
             test_details=True,min_time_limit=1.,gt_time_limit_factor=4.,mini=False,noextreme=False)
    result=json.loads(Path('samples_eval_results.json').read_text())['eval']
    expected={json.loads(x)['task_id'] for x in Path('samples.jsonl').read_text().splitlines()}
    if set(result)!=expected or any(len(v)!=1 for v in result.values()):
        raise ValueError('Incomplete or duplicated scoring coverage')
    correct=sum(v[0]['base_status']=='pass' and v[0]['plus_status']=='pass' for v in result.values())
    Path('JUDGE_PASS.json').write_text(json.dumps(dict(total=len(result),correct=correct,score=correct/len(result),full_plus=True)))
