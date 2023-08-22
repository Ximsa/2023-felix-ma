from matplotlib import pyplot as plt
import matplotlib
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
def plot_embeddings(embeddings, labels=None, save=False):
    reduced = TSNE(n_components=2).fit_transform(embeddings.detach())
    xs = reduced[:,0]
    ys = reduced[:,1]
    if labels is None:
        labels = range(len(xs))
    plt.scatter(xs, ys, c=labels)

def plot_confusion_matrix(xs,ys):
    matrix = confusion_matrix(xs,ys)
    plt.imshow(matrix, cmap='hot', interpolation='nearest')
