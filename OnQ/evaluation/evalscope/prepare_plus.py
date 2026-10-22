"""Download full EvalPlus problems locally; no datasets are shipped in the release."""
import argparse
import json
from pathlib import Path
from importlib.metadata import version


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if version('evalplus')!='0.3.1':raise RuntimeError('Requires evalplus==0.3.1')
    from evalplus.data import get_human_eval_plus,get_mbpp_plus
    a.output.mkdir(parents=True,exist_ok=False)
    for name,loader,expected,revision in [('HumanEvalPlus',get_human_eval_plus,164,'v0.1.10'),('MbppPlus',get_mbpp_plus,378,'v0.2.0')]:
        rows=loader(mini=False,noextreme=False,version=revision)
        if len(rows)!=expected:raise ValueError('Unexpected full benchmark coverage')
        with (a.output/(name+'.jsonl')).open('x') as f:
            for row in rows.values():f.write(json.dumps(row)+'\n')
    (a.output/'PREPARED.json').write_text(json.dumps(dict(evalplus='0.3.1',humaneval=164,mbpp=378,humaneval_revision='v0.1.10',mbpp_revision='v0.2.0',mini=False,noextreme=False)))

if __name__=='__main__':main()
