"""Validate the candidate-wise Dynamic KL numerical options."""
import math



def validate_options(options):
    if not isinstance(options, dict) or set(options) != {'epsilon', 'mass_tolerance', 'chunk_size'}:
        raise ValueError('dynamic_kl requires epsilon, mass_tolerance and chunk_size')
    for name in ('epsilon', 'mass_tolerance'):
        value = options[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'Invalid dynamic_kl {name}')
    if not 0 < options['epsilon'] <= 1e-4 or not 0 <= options['mass_tolerance'] <= 1e-5:
        raise ValueError('Invalid dynamic_kl smoothing or probability tolerance')
    if type(options['chunk_size']) is not int or options['chunk_size'] <= 0:
        raise ValueError('dynamic_kl chunk_size must be a positive integer')
