import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

sns.set_context('poster')


def view_emb(emb, dir):
    if emb.shape[1] > 2:
        pca = PCA(n_components=2)
        emb = pca.fit_transform(emb)
    plt.figure(figsize=(10, 10))
    plt.scatter(emb[:, 0], emb[:, 1])
    plt.savefig('{}/emb_pca.png'.format(dir), dpi=100)
