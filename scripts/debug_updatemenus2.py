from src.visualization.interactive_plots import animate_convergence
import numpy as np

trajectories = {'a': np.array([[-1.0,1.0],[0.0,0.0]]),'b':np.array([[0,0],[1,2]])}
loss_histories={'a':np.array([1.0,0.5]),'b':np.array([2.0,0.4])}
fig = animate_convergence(trajectories, loss_histories, title='debug', frame_duration=50)
print('fig.layout type:', type(getattr(fig,'layout')))
layout = getattr(fig,'layout')
up = getattr(layout,'updatemenus', None)
print('up type:', type(up), 'repr:', repr(up))
print('up has __iter__ in dict?:', '__iter__' in getattr(up,'__dict__', {}))
print('getattr(up, "__iter__") ->', getattr(up,'__iter__', None))

# Inspect attributes on fig to see if proxy layout exists
print('fig has updatemenus_carrier?', hasattr(fig, 'updatemenus_carrier'))
print('fig has updatemenus_proxy?', hasattr(fig, 'updatemenus_proxy'))
print('layout is proxy?', hasattr(layout, '_real'))
print('layout.__class__', layout.__class__)
