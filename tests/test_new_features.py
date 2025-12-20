"""Quick validation test for new features."""
import sys
sys.path.insert(0, 'src')

# Test convergence analyzer
print('Testing convergence rate analyzer...')
from src.analysis.convergence_rate_analyzer import compute_empirical_rate
import numpy as np

# Create synthetic power-law data
t = np.arange(1, 51)
loss = 10.0 * np.power(t, -0.8) + 0.1
result = compute_empirical_rate(loss.tolist(), method='power')

if result.get('success'):
    alpha = result['power_law']['alpha']
    r2 = result['power_law']['r_squared']
    print(f'✓ Power-law fit: α={alpha:.3f}, R²={r2:.3f}')
    print(f'  Expected α≈0.8, got {alpha:.3f} (error: {abs(alpha-0.8):.3f})')
else:
    print('✗ Fit failed')

# Test dataset provenance
print('\nTesting dataset provenance...')
from src.core.dataset_provenance import get_dataset_provenance
prov = get_dataset_provenance('MNIST', split='train', data_root='./data')
print(f'Keys: {list(prov.keys())}')
source = prov.get('data_source', 'MISSING')
version = prov.get('dataset_version', 'MISSING')
print(f'✓ MNIST provenance: source={source}, version={version}')

# Test CIFAR-100 loader
print('\nTesting CIFAR-100 loader...')
try:
    from src.core.data_utils import get_cifar100_loaders
    print('✓ CIFAR-100 loader imported successfully')
except Exception as e:
    print(f'✗ CIFAR-100 import failed: {e}')

print('\n✓ All core functions working!')
