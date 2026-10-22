"""Curriculum for training; fixed 16K greedy held-out generation."""
from verl.workers.rollout.vllm_rollout import vllm_async_server as native
from .paper_contract import rollout_cap

_original_extract = native.extract_prompt_logprobs


def extract_with_finish(*, output, num_prompt_logprobs, result_dict):
    result = _original_extract(output=output, num_prompt_logprobs=num_prompt_logprobs, result_dict=result_dict)
    if len(output.outputs) != 1:
        raise ValueError('OnQ expects n=1')
    result_dict['qat_finish_reason'] = output.outputs[0].finish_reason
    result_dict['qat_generated_tokens'] = len(output.outputs[0].token_ids)
    return result


class PaperRolloutServer(native.vLLMHttpServer):
    async def generate(self, prompt_ids, sampling_params, request_id, **kwargs):
        params = dict(sampling_params)
        if 'max_tokens' in params or 'max_new_tokens' in params:
            raise ValueError('Unexpected per-request length override')
        cap = rollout_cap(self.global_steps, params.get('temperature'))
        params['max_tokens'] = cap
        result = await super().generate( prompt_ids, params, request_id, **kwargs)
        result.extra_fields['qat_response_cap'] = cap
        return result

    def _post_init(self, cuda_visible_devices):
        if native.extract_prompt_logprobs not in (_original_extract, extract_with_finish):
            raise ValueError('Unknown finish-reason hook')
        native.extract_prompt_logprobs = extract_with_finish
        return super()._post_init(cuda_visible_devices)
