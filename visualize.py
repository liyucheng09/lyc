from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from typing import List

MARKERS = ['.', 'o', 'v', '^', '<', '>', 's', '*', '+', 'x']
COLORS = ['aqua', 'azure', 'beige', 'black', 'bleu', 'brown', 'gold', 'green', 'ivory', 'cyan', 'navy', 'pink', 'red', 'teal', 'tan', 'yellow']

def plotDimensionReduction(X, labels: List[str], figure_name, \
        plot_type = 'PCA', n_components = 2, legend_loc = 6,
        bbox_to_anchor = (1, 0.5), **kwargs):
    def pca(X, **kwargs):
        pca = PCA(n_components=n_components, **kwargs)
        X = pca.fit_transform(X)
        return X
    def tSNE(X, **kwargs):
        tsne = TSNE(n_components=n_components, **kwargs)
        X = tsne.fit_transform(X)
        return X
    
    plot_func = {
        'PCA': pca,
        'tSNE': tSNE
    }

    X = plot_func[plot_type](X, **kwargs)

    label_type = []
    for label in labels:
        if label not in label_type: label_type.append(label)

    labels = np.array(labels)
    num_labels = len(label_type)
    assert num_labels <=9, f"Colors not enough, have {len(COLORS)} colors, but got {num_labels} of labels."
    fig, ax = plt.subplots()
    markers, colors = MARKERS[:num_labels], COLORS[: num_labels]

    names = []
    for label in label_type:
        index = (labels == label)
        ax.scatter(X[index, 0], X[index, 1], c=colors[label_type.index(label)])
    
    ax.legend(label_type, bbox_to_anchor=bbox_to_anchor, loc=legend_loc, borderaxespad=0.)
    # plt.show()
    plt.savefig(figure_name, bbox_inches = "tight", dpi=300.)
    print(f'Saved to {figure_name}!')
    return X