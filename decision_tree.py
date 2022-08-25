from collections import namedtuple
from pprint import pprint

import numpy as np
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import seaborn as sns

Node = namedtuple('Node', ['dim', 'split_value', 'left', 'right'])


def entropy_over_region(y):
    """
    Calculates the entropy in a single region of the partition space
    :param y: the labels in one region m
    :return:
    """
    n, = y.shape
    if n == 0.0:
        return 0.0
    y_unique = np.unique(y)
    ent = 0.0
    for label in y_unique:
        c = (y == label).sum()
        if c == 0:
            continue
        ent += ((c / n) * np.log2(c / n))
    return -ent


def weighted_entropy(regions, n):
    """

    Notes
    -----

    Parameters
    ---------


    Returns
    -------
    

    :param regions:
    :param n:
    :return:
    """
    if n == 0.0:
        return 0.0
    total = 0.0
    for region in regions:
        m, = region.shape
        total += (m / n * entropy_over_region(region))
    return total


def find_best_split(X, y):
    n, d = X.shape
    assert n != 0

    # we find split with the highest information gain
    axis_best, s_best, i_best, E_best = None, None, None, np.inf
    for axis in range(d):
        coords = X[:, axis]
        key = np.argsort(coords)
        y_sorted = y[key]
        C_sorted = coords[key]
        for i in range(n - 1):
            if C_sorted[i] == C_sorted[i + 1]:
                continue
            E = weighted_entropy([y_sorted[:i + 1], y_sorted[i + 1:]], n)
            if E < E_best:
                E_best, axis_best, i_best, s_best = E, axis, i, (C_sorted[i] + C_sorted[i + 1]) / 2
    return axis_best, s_best, i_best, E_best


def build_leaf(y):
    # return the label with the highest count
    y_unique, counts = np.unique(y, return_counts=True)
    return y_unique[np.argmax(counts)]


def build_tree(X, y, leaf_size=5, depth=1000):
    n, d = X.shape
    assert n != 0
    if n <= leaf_size or depth == 0:
        return build_leaf(y)
    else:
        axis_best, s_best, i, ent = find_best_split(X, y)
        indices = np.argsort(X[:, axis_best])
        X_sorted = X[indices]
        y_sorted = y[indices]

        if ent == 0.0:
            return build_leaf(y_sorted)

        return Node(axis_best,
                    s_best,
                    build_tree(X_sorted[:i, :], y_sorted[:i], leaf_size, depth - 1),
                    build_tree(X_sorted[i:, :], y_sorted[i:], leaf_size, depth - 1))


def predict(root: Node, x):
    if not isinstance(root, Node):
        return root
    else:
        if x[root.dim] < root.split_value:
            return predict(root.left, x)
        else:
            return predict(root.right, x)


def predict_vector(root: Node, xs):
    return np.array([predict(root, x) for x in xs])


def eval_classifier(leaf_size, X_train, y_train, X_test, y_test):
    root = build_tree(X_train, y_train, leaf_size)
    n, _ = X_test.shape
    return np.sum(predict_vector(root, X_test) == y_test) * (1 / n)


def cross_validate(leaf_size, X, y, k, verbose=False):
    n, d = X.shape
    X_y = np.concatenate([X, y[:, np.newaxis]], axis=1)
    np.random.shuffle(X_y)
    X, y = X_y[:, :d], X_y[:, d]

    s_data = np.array_split(X, k, axis=0)
    s_labels = np.array_split(y, k)

    score_sum = 0
    for i in range(k):
        X_train = np.concatenate(s_data[:i] + s_data[i + 1:], axis=0)
        y_train = np.concatenate(s_labels[:i] + s_labels[i + 1:], axis=0)
        X_test = np.array(s_data[i])
        y_test = np.array(s_labels[i])
        score = eval_classifier(leaf_size, X_train, y_train,
                                X_test, y_test)
        score_sum += score
        if verbose:
            print(f" iteration {i}, score: {score}")
    return score_sum / k


if __name__ == '__main__':
    iris_X, iris_y = load_breast_cancer(return_X_y=True)
    print(iris_X.shape)
    print(iris_y.shape)

    # # plot two features
    # plt.figure(num=None, figsize=(12, 8), dpi=150, facecolor='w', edgecolor='k')
    #
    # # Scatter the points, using size and color but no label
    # sns.scatterplot(x=iris_X[:, 0], y=iris_X[:, 1], hue=iris_y)
    #
    # plt.grid(True)
    #
    # plt.xlabel("sepal length (cm)")
    # plt.ylabel("sepal width (cm)")
    #
    # plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title='Target Names')
    # plt.show()

    # Split the data into training/testing sets
    iris_X_train = iris_X[:400]
    iris_X_test = iris_X[400:]

    # Split the targets into training/testing sets
    iris_y_train = iris_y[:400]
    iris_y_test = iris_y[400:]

    # plot two features
    # plt.figure(num=None, figsize=(12, 8), dpi=150, facecolor='w', edgecolor='k')
    #
    # # Scatter the points, using size and color but no label
    # sns.scatterplot(x=iris_X_train[:, 0], y=iris_X_train[:, 1], hue=iris_y_train)
    #
    # plt.grid(True)
    #
    # plt.xlabel("sepal length (cm)")
    # plt.ylabel("sepal width (cm)")
    #
    # plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title='Target Names')
    # plt.show()

    tree = build_tree(iris_X_train, iris_y_train, leaf_size=1)
    pprint(tree)
