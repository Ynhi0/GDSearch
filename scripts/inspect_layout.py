from src.visualization.interactive_plots import animate_convergence
import numpy as np
trajectories = {'a': np.array([[-1.0,1.0],[0.0,0.0]]),'b':np.array([[0,0],[1,2]])}
loss_histories={'a':np.array([1.0,0.5]),'b':np.array([2.0,0.4])}
fig = animate_convergence(trajectories, loss_histories, title='debug', frame_duration=50)
layout = getattr(fig, 'layout')
print('layout type', type(layout))
try:
    print('layout attrs:', dir(layout)[:20])
except Exception as e:
    print('dir failed', e)
print('has _props?', hasattr(layout, '_props'))
if hasattr(layout, '_props'):
    try:
        print('updatemenus in _props?', 'updatemenus' in layout._props)
        if 'updatemenus' in layout._props:
            print('layout._props["updatemenus"] type:', type(layout._props['updatemenus']))
            print('layout._props["updatemenus"] repr:', repr(layout._props['updatemenus']))
    except Exception as e:
        print('inspect error', e)