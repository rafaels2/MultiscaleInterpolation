import matplotlib.pyplot as plt
import numpy as np
from pykdtree.kdtree import KDTree

from . import register_generation

HALTON_SIZE = 400
HALTON_DIM = 2

# Calculated
SEPARATION_DISTANCE, DEFAULT_FILL_DISTANCE = (0.017527259373766313, 0.09051018919638595)


def next_prime():
    def is_prime(num):
        "Checks if num is a prime value"
        for i in range(2, int(num ** 0.5) + 1):
            if (num % i) == 0:
                return False
        return True

    prime = 3
    while 1:
        if is_prime(prime):
            yield prime
        prime += 2


def vdc(n, base=2):
    vdc, denom = 0, 1
    while n:
        denom *= base
        n, remainder = divmod(n, base)
        vdc += remainder / float(denom)
    return vdc


def halton_sequence(size, dim):
    seq = []
    prime_gen = next_prime()
    next(prime_gen)
    for d in range(dim):
        base = next(prime_gen)
        seq.append([vdc(i, base) for i in range(size)])
    return seq


@register_generation("halton")
def get_scaled_halton(x_min, x_max, y_min, y_max, fill_distance):
    # TODO: generalize to n-dim
    seq = halton_sequence(HALTON_SIZE, HALTON_DIM)
    data_points = np.transpose(np.array(seq))
    orig_tree = KDTree(data_points)
    _, default_fill_distance = measure_fill_and_separation(orig_tree, data_points)

    scaling_ratio = fill_distance / default_fill_distance
    x_duplications = int(np.ceil((x_max - x_min) / scaling_ratio))
    y_duplications = int(np.ceil((y_max - y_min) / scaling_ratio))
    output = np.zeros((HALTON_SIZE * x_duplications * y_duplications, HALTON_DIM))

    for i in range(x_duplications):
        for j in range(y_duplications):
            index = (i + j * x_duplications) * HALTON_SIZE
            # print(index)
            origin = np.array([x_min, y_min]) * np.ones_like(data_points)
            # translation = (1 - default_fill_distance) * np.array([i, j]) * np.ones_like(data_points)
            translation = np.array([i, j]) * np.ones_like(data_points)
            output[index : index + HALTON_SIZE, :] = (
                scaling_ratio * (data_points + translation) + origin
            )

    return output


def measure_fill_and_separation(tree, seq):
    lst = []
    for i in range(seq.shape[0]):
        dist, _ = tree.query(seq[i, :].reshape([1, 2]), k=2)
        # print("distance: ", dist[-1][-1])
        lst.append(dist[-1][-1])
    return min(lst), max(lst)


def main():
    def test():
        mins = list()
        maxs = list()
        sizes = list()
        for size in range(100, 20000, 200):
            print("size :", size)
            seq = halton_sequence(size, 2)
            data = np.array(seq).transpose()
            tree = KDTree(data)
            mn, mx = measure_fill_and_separation(tree, data)
            mins.append(mn)
            maxs.append(mx)
            sizes.append(np.log(size))
            print(mn, mx)

        plt.figure()
        plt.plot(sizes, mins)
        plt.plot(sizes, maxs)
        plt.show()

    # test()
    size = 100
    seq = halton_sequence(size, 2)
    data = np.array(seq).transpose()
    tree = KDTree(data)
    return tree


if __name__ == "__main__":
    _tree = main()
