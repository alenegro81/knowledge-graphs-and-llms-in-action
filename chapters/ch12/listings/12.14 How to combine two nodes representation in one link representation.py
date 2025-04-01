def catenate(u, v):
    return u + v


def operator_avg(u, v):
    return (u + v) / 2.0


def operator_l1(u, v):
    return np.abs(u - v)


def operator_l2(u, v):
    return (u - v) ** 2


def operator_hadamard(u, v):
    return u * v