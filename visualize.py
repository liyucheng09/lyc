from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

MARKERS = ['.', 'o', 'v', '^', '<', '>', 's', '*', '+', 'x']
COLORS = ['b', 'r', 'c', 'm', 'y', 'k', 'gray', 'navy', 'gold']

def plotPCA(X, labels: list[str], figure_name = 'plotPCA.png'):
    pca = PCA(n_components=2)
    X = pca.fit_transform(X)

    label_type = list(set(labels))
    labels = np.array(labels)
    num_labels = len(label_type)
    assert num_labels <=9, f"markers and colors not enough, have {len(MARKERS)} input number_of_label {num_labels}"
    fig, ax = plt.subplots()
    markers, colors = MARKERS[:num_labels], COLORS[: num_labels]

    names = []
    for label in label_type:
        index = (labels == label)
        ax.scatter(X[index, 0], X[index, 1], c=colors[label_type.index(label)])
    
    ax.legend(label_type, bbox_to_anchor=(1, 0.5), loc='center left')
    plt.savefig(figure_name, bbox_inches='tight', dpi=300.)