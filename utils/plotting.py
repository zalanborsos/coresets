import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

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


def plot_ellipses(ax, weights, means, covars):
    for n in range(means.shape[0]):
        if weights[n] > 1e-3:
            eig_vals, eig_vecs = np.linalg.eigh(covars[n])
            unit_eig_vec = eig_vecs[0] / np.linalg.norm(eig_vecs[0])
            angle = np.arctan2(unit_eig_vec[1], unit_eig_vec[0])
            angle = 180 * angle / np.pi
            eig_vals = 2 * np.sqrt(2) * np.sqrt(eig_vals)
            alpha = 0.4
            ell = mpl.patches.Ellipse(means[n], eig_vals[0], eig_vals[1],
                                      180 + angle, linewidth=2,
                                      facecolor=(0, 0, 1, alpha), edgecolor='black')
            ell.set_clip_box(ax.bbox)
            ax.add_artist(ell)
            plt.scatter(means[n, 0], means[n, 1], c='black', s=8, alpha=0.8)