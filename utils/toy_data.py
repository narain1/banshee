from sklearn.datasets import make_moons, make_blobs

def make_dataset(seed=13, n_samples=100):
    np.random.seed(seed)
    random.seed(seed)
    x, y = make_moons(n_samples=100, noise=0.1)
    y = y*2 - 1
    return x, y
