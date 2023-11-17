import os.path as osp


def view_emb(emb, dir):
    """Visualize a embedding matrix.

    Args:
        emb (torch.tensor): Embedding matrix with shape (N, D). D is the
        feature dimension.
        dir (str): Output directory for the embedding figure.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.decomposition import PCA

    sns.set_context('poster')

    if emb.shape[1] > 2:
        pca = PCA(n_components=2)
        emb = pca.fit_transform(emb)
    plt.figure(figsize=(10, 10))
    plt.scatter(emb[:, 0], emb[:, 1])
    plt.savefig(osp.join(dir, 'emb_pca.png'), dpi=100)
