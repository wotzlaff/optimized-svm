from .simple import SimpleSVM
from .optimized_svm import OptimizedSVM
from .problem import Classification, Regression, LSSVR, PoissonSVM
from .kernel import rbf_kernel

__all__ = [
    'OptimizedSVM', 'SimpleSVM',
    'Classification', 'Regression',
    'LSSVR', 'PoissonSVM',
    'rbf_kernel',
]

__version__ = '0.0.1'
