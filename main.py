import numpy as np
from scipy import integrate


def f(x, y, z):
    return x**2 + y**2 + z**2


def monte_carlo_3d(func, x_range, y_range, z_range, n_samples):
    x = np.random.uniform(x_range[0], x_range[1], n_samples)
    y = np.random.uniform(y_range[0], y_range[1], n_samples)
    z = np.random.uniform(z_range[0], z_range[1], n_samples)
    values = func(x, y, z)
    volume = (
        (x_range[1] - x_range[0])
        * (y_range[1] - y_range[0])
        * (z_range[1] - z_range[0])
    )
    integral_estimate = volume * np.mean(values)
    return integral_estimate


x_range = (0, 1)
y_range = (0, 1)
z_range = (0, 1)
n_samples = 1000
mc_result = monte_carlo_3d(f, x_range, y_range, z_range, n_samples)
true_result, error = integrate.tplquad(
    lambda z, y, x: f(x, y, z),
    x_range[0],
    x_range[1],
    y_range[0],
    y_range[1],
    z_range[0],
    z_range[1],
)
print(mc_result, true_result, error)
