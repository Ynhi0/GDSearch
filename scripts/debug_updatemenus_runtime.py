import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.visualization.interactive_plots import animate_convergence
import numpy as np
trajectories={'A':np.array([[0,0],[1,1]]),'B':np.array([[0,0],[1,1]])}
loss_histories={'A':np.array([1,0.1]),'B':np.array([1,0.2])}
fig=animate_convergence(trajectories, loss_histories)
layout=getattr(fig,'layout',None)
up=getattr(layout,'updatemenus',None)
print('up type:', type(up))
print('up repr:', repr(up))
print('__iter__ attr:', getattr(up,'__iter__', None))
try:
    iter_attr = getattr(up,'__iter__', [])
    print('len(iter_attr):', len(iter_attr))
except Exception as e:
    print('len(iter_attr) error:', e)

# print carrier if present
if hasattr(fig,'updatemenus_carrier'):
    c = fig.updatemenus_carrier
    print('carrier type:', type(c), 'len:', len(c), 'iter_attr type:', type(getattr(c,'__iter__', None)))

print('layout keys:', list(fig['layout'].keys()))
print('layout updatemenus in dict:', 'updatemenus' in fig['layout'])
print('layout updatemenus dict value type:', type(fig['layout'].get('updatemenus', None)))
