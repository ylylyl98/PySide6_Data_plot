"""Raw-input reconstruction of the prepared P/N background research pipeline.

The numerical preparation is independent of saved backgrounds and BN. The
supported protocol and its regression evidence are separate from a guarantee
of scientific quality for a new measurement group.
"""

def reconstruct_background(*args, **kwargs):
    from .pipeline import reconstruct_background as run
    return run(*args, **kwargs)


def load_raw_inputs(*args, **kwargs):
    from .io import load_raw_inputs as load
    return load(*args, **kwargs)


def generate_from_files(*args, **kwargs):
    from .io import generate_from_files as generate
    return generate(*args, **kwargs)
