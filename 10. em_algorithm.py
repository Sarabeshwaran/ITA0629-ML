import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
np.random.seed(42)
data = np.concatenate([np.random.normal(-10, 1, 300),
                       np.random.normal(0, 2, 400),
                       np.random.normal(10, 1.5, 300)])
mu = np.array([-5, 0, 5])
sigma = np.array([1, 1, 1])
weights = np.array([1/3, 1/3, 1/3])
max_iter = 100
tolerance = 1e-6
for _ in range(max_iter):
    responsibilities = np.zeros((len(data), len(mu)))
    for i in range(len(mu)):
        responsibilities[:, i] = weights[i] * norm.pdf(data, mu[i], sigma[i])
    responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)
    new_mu = np.sum(responsibilities * data[:, np.newaxis], axis=0) / np.sum(responsibilities, axis=0)
    new_sigma = np.sqrt(np.sum(responsibilities * (data[:, np.newaxis] - new_mu) ** 2, axis=0) / 
                        np.sum(responsibilities, axis=0))
    new_weights = np.mean(responsibilities, axis=0)
    if np.max(np.abs(new_mu - mu)) < tolerance and np.max(np.abs(new_sigma - 
                    sigma)) < tolerance and np.max(np.abs(new_weights - weights)) < tolerance:
        break
    mu, sigma, weights = new_mu, new_sigma, new_weights
plt.hist(data, bins=30, density=True, alpha=0.5, color='gray')
x = np.linspace(np.min(data), np.max(data), 1000)
for i in range(len(mu)):
    plt.plot(x, weights[i] * norm.pdf(x, mu[i], sigma[i]), label=f'Component {i+1}')
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Gaussian Mixture Model')
plt.legend()
plt.show()
