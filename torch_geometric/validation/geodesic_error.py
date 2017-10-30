def geodesic_error(position, pred, target):
    direction = position[pred] - position[target]
    return (direction * direction).sum(1).float().sqrt()


def max_geodesic_error_accuracy(position, pred, target, error=None):
    err = geodesic_error(position, pred, target)
    return (err <= error).sum()
