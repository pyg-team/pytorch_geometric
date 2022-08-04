def torch_version_minor():
    import torch
    _, minor, _ = torch.__version__.split('.')
    try:
        return int(minor)
    except ValueError:
        return 0
