def geodesic_error(position, pred, label):
    direction = position[pred] - position[label]
    return (direction * direction).sum(1).float().sqrt()


def max_geodesic_error_accuracy(position, pred, label, error=0.0):
    err = geodesic_error(position, pred, label)
    return (err <= error).sum()
