"""Load the pinned native scorer in the isolated container without model imports."""
import importlib.util
import json
import logging
import resource
import sys
import types
import zipfile
from pathlib import Path
import numpy as np

if __name__=='__main__':
    resource.setrlimit(resource.RLIMIT_AS,(4<<30,4<<30))
    for name in ['evalscope','evalscope.utils','evalscope.utils.logger']:
        sys.modules[name]=types.ModuleType(name)
    sys.modules['evalscope.utils.logger'].get_logger=lambda *a,**k:logging.getLogger('evalscope')
    wheel=next(Path('/opt/scorer').glob('evalscope-1.5.1-*.whl'))
    source=zipfile.ZipFile(wheel).read('evalscope/benchmarks/live_code_bench/testing_util.py')
    module=types.ModuleType('native_lcb_testing_util')
    exec(compile(source,'testing_util.py','exec'),module.__dict__)
    request=json.loads(Path('/work/request.json').read_text())
    print('JUDGE_START_' + request['nonce'], flush=True)
    result,metadata=module.run_test(request['sample'],test=request['generation'],timeout=request['timeout'],debug=request['debug'])
    def convert(x):
        if isinstance(x,np.ndarray):x=x.item(0)
        if isinstance(x,np.generic):return x.item()
        return x
    Path('/work/result.json').write_text(json.dumps(dict(result=[convert(x) for x in result],metadata=metadata),default=str))
