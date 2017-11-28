def repr(obj, keys=None):
    str = '{name}({in_features}, {out_features}'

    if keys is not None:
        str += ', ' + ', '.join([key + '={' + key + '}' for key in keys])

    if obj.bias is None:
        str += ', bias=False'

    str += ')'

    return str.format(name=obj.__class__.__name__, **obj.__dict__)

    # s = ('{name}({in_features}, {out_features}, kernel_size='
    #      '{kernel_size_repr}, is_open_spline={is_open_spline_repr}, '
    #      'degree={degree}')
    # if self.bias is None:
    #     s += ', bias=False'
    # s += ')'
    # return s.format(name=self.__class__.__name__, **self.__dict__)
