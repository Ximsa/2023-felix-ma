from matplotlib import pyplot as plt


def plot_embeddings(embeddings, labels=None, save=False):
    xs = embeddings[:,0]
    ys = embeddings[:,1]
    if labels is None:
        labels = range(len(xs))
    plt.scatter(embeddings[:,0], embeddings[:,1], c=labels)
