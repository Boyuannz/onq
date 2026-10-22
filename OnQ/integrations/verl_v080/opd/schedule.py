"""Response-token curriculum for OnQ's one-based updates."""
V4_SCHEDULE = '1-50:4096,51-100:8192,101-400:16384'


def response_cap(step, *, schedule=V4_SCHEDULE):
    if type(step) is not int or not 1 <= step <= 400:
        raise ValueError('Expected a one-based update in [1, 400]')
    if schedule != V4_SCHEDULE:
        raise ValueError('Unsupported OnQ length schedule')
    return next(cap for end, cap in ((50, 4096), (100, 8192), (400, 16384)) if step <= end)
