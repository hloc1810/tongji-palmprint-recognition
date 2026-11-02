try:
    import torch  # noqa: F401 - Pre-load torch DLLs before pytest plugins/pandas on Windows
except Exception:
    pass
