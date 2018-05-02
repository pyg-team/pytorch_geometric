def repr(obj, keys=None):
    params = []

    if getattr(obj, 'in_channels', None) is not None:
        params.append(str(obj.in_channels))

    if getattr(obj, 'out_channels', None) is not None:
        params.append(str(obj.out_channels))

    if keys is not None:
        params.extend([key + '={' + key + '}' for key in keys])

    if getattr(obj, 'weight', None) is not None and \
       getattr(obj, 'bias', None) is None:
        params.append('bias=False')

    out = '{name}(' + ', '.join(params) + ')'
    return out.format(name=obj.__class__.__name__, **obj.__dict__)
