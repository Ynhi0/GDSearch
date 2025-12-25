from src.core.test_functions import Rosenbrock
r=Rosenbrock()
print('grad at (-1.5,2):', r.gradient(-1.5,2))
print('compute:', r.compute(-1.5,2))
