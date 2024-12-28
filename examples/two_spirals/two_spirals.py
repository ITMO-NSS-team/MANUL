import math
import numpy as np
from matplotlib import pyplot as plt


def spiral_xy(i, spiral_num):
    """
    Create the data for a spiral.

    Arguments:
        i runs from 0 to 96
        spiral_num is 1 or -1
    """
    phi = i / 16 * math.pi
    r = 6.5 * ((104 - i) / 104)
    x = (r * math.cos(phi) * spiral_num) / 13 + 0.5
    y = (r * math.sin(phi) * spiral_num) / 13 + 0.5
    return (x, y)


def spiral(spiral_num):
    return [spiral_xy(i, spiral_num) for i in range(100)]


class_a = spiral(1)
class_b = spiral(-1)
dataset = np.array(class_a+class_b)
labels = [0]*len(class_a)+[1]*len(class_b)

plt.scatter(dataset[:, 1], dataset[:, 0], c=labels)
plt.show()

test_inds = np.random.choice(np.arange(dataset.shape[0]), int(dataset.shape[0]*0.2), replace=False)
test = dataset[test_inds]
