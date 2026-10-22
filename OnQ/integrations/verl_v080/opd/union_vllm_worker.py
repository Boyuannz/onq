"""Run-scoped vLLM prompt-score side channel; ordinary logits never changed."""

import inspect, re, textwrap
from vllm.v1.worker import gpu_model_runner as impl


from ..utils.union_teacher_scores import record_union_chunk

original = impl.GPUModelRunner._get_prompt_logprobs_dict
source = textwrap.dedent(inspect.getsource(original))
needle = "logprobs = self.sampler.compute_logprobs(logits)"
assert source.count(needle) == 1, "Pinned vLLM prompt scoring path changed"
indent = re.search(r"(?m)^( *)" + re.escape(needle), source).group(1)
source = source.replace(
    needle, needle + "\n" + indent + "record_union_chunk(request, start_idx, logprobs)"
)
namespace = {**vars(impl), "record_union_chunk": record_union_chunk}
exec(compile(source, "<reviewed_union_prompt_score_hook>", "exec"), namespace)
impl.GPUModelRunner._get_prompt_logprobs_dict = namespace["_get_prompt_logprobs_dict"]


class UnionWorkerExtension:
    """Import hook only: selected by the isolated teacher worker config."""

    pass
