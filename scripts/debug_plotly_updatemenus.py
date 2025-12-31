from src.visualization.interactive_plots import animate_convergence
import numpy as np
trajectories={'A':np.array([[0,0],[1,1]]),'B':np.array([[0,0],[1,1]])}
loss_histories={'A':np.array([1,0.1]),'B':np.array([1,0.2])}
fig=animate_convergence(trajectories, loss_histories)
layout=getattr(fig,'layout',None)
print('layout type:', type(layout))
up=getattr(layout,'updatemenus',None)
print('updatemenus raw type:', type(up), 'repr:', repr(up))
print('getattr up __iter__:', getattr(up,'__iter__',None))
try:
    print('len getattr:', len(getattr(up,'__iter__',[])))
except Exception as e:
    print('len getattr error:', e)
print('has proxy attr:', hasattr(fig,'updatemenus_proxy'))
if hasattr(fig,'updatemenus_proxy'):
    print('proxy type:', type(fig.updatemenus_proxy), 'len:', len(fig.updatemenus_proxy))

# Evaluate the exact test condition
layout = getattr(fig, 'layout', None)
updatemenus = getattr(layout, 'updatemenus', None)
cond = (updatemenus is not None)
try:
    iter_attr = getattr(updatemenus, '__iter__', [])
    cond2 = len(iter_attr) > 0
except Exception as e:
    cond2 = f'error: {e}'
print('test condition parts:', cond, cond2)