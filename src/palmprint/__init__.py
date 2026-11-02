try:
    import torch  # noqa: F401 - Pre-load torch DLLs before pandas/openmp on Windows
except Exception:
    pass
