import numpy as np
import matplotlib.pyplot as plt


def plot_2d_points_with_assignment(points, assignment):
    unique, counts = np.unique(assignment, return_counts=True)
    k = unique.shape[0]

    cmap = plt.cm.get_cmap('hsv', k)
    cmaplist = [cmap(i) for i in range(cmap.N)]

    k = 0
    for i in unique:
        pts = points[assignment == i]
        plt.scatter(pts[:, 0], pts[:, 1], c=cmaplist[k], s=70, alpha=0.5)
        k += 1


def plot_2d_points_and_coreset(points, coreset, weights):
    plt.scatter(points[:, 0], points[:, 1], c='gray', s=70, alpha=0.5)
    plt.scatter(coreset[:, 0], coreset[:, 1], c='red', s=60 + weights, alpha=0.9, marker='*')
