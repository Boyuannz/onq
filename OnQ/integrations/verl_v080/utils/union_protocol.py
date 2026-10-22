"""Capacity of the reviewed single-teacher producer, independent of CUDA."""

MAX_INPUT_TOKENS = 20481
MAX_MODEL_LEN = 20496


def required_input_tokens(config):
    data, rollout = config.data, config.actor_rollout_ref.rollout
    lengths = [data.max_prompt_length, data.max_response_length,
               rollout.get('prompt_length', data.max_prompt_length),
               rollout.get('response_length', data.max_response_length)]
    if any(type(n) is not int or n <= 0 for n in lengths):
        raise ValueError('Union prompt/response capacities must be positive integers')
    # Same conservative extra-token allowance as the legacy teacher admission.
    required = max(lengths[0], lengths[2]) + max(lengths[1], lengths[3]) + 1
    validate_capacity(required, MAX_INPUT_TOKENS, MAX_MODEL_LEN)
    return required


def validate_capacity(required, max_input, max_model):
    if (any(type(n) is not int or n <= 0 for n in (required, max_input, max_model))
            or required > max_input or required + 1 > max_model):
        raise ValueError('Union teacher context capacity is missing or insufficient')
