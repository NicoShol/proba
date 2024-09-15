import numpy as np
import matplotlib.pyplot as plt

# Uniform transformation
def f(x):
    return - np.log(x)

def g(x):
    return np.exp(-x)

def define_range_transform() -> tuple:
    """Whan applying transform to uniform values, the image may differ from 0,1"""
    return 0.,10.

def plot_histogram(samples: np.ndarray, _config: dict) -> None:
    kwargs = _config if _config else dict()
    plt.hist(samples, **kwargs)


def uniform_transform(N=10_000) -> np.ndarray:
    # Uniform sampling
    X = np.random.random(size=N)
    return np.array([f(s) for s in X])

def plot_expected_pdf(pdf: callable, interval: tuple, n_points: int, _config: dict) -> None:
    kwargs = _config if _config else dict()
    x_min, x_max = interval
    x = np.linspace(x_min, x_max, n_points)
    y = np.array([pdf(s) for s in x])
    plt.plot(x,y,**kwargs)

def main(N):
    CONFIG_HIST = {"bins": 50, "edgecolor": "black", "alpha": 0.5, "density": True}
    CONFIG_PLOT = {"color": "orange", "linestyle": "solid"}  #, "marker": "o", "markersize": 2}

    plt.figure()
    X = uniform_transform(N)
    plot_histogram(X, CONFIG_HIST)
    plot_expected_pdf(g, define_range_transform(), N//10, CONFIG_PLOT)
    plt.show()

if __name__ == "__main__":
    main(N=100_000)