if __name__ == '__main__':
    from src.visualization.interactive_plots import animate_convergence
    import numpy as np
    from typing import Iterable, Any, cast

    trajectories = {'a': np.array([[-1.0,1.0],[0.0,0.0]]),'b':np.array([[0,0],[1,2]])}
    loss_histories={'a':np.array([1.0,0.5]),'b':np.array([2.0,0.4])}
    fig = animate_convergence(trajectories, loss_histories, title='debug', frame_duration=50)
    layout = getattr(fig,'layout')
    print('before, up type:', type(getattr(layout,'updatemenus')))

    class MyUp(tuple):
        def __new__(cls, data: object):
            # Coerce to tuple safely for a wide range of container-like inputs
            try:
                sequence = tuple(cast(Iterable[Any], data)) if data is not None else ()
            except Exception:
                sequence = ()
            return super().__new__(cls, sequence)

    props = getattr(layout, '_props')
    up_raw = props.get('updatemenus', [])
    # Only convert to list if it's actually iterable
    try:
        if isinstance(up_raw, Iterable):
            safe_up = list(up_raw)
        else:
            safe_up = []
    except Exception:
        safe_up = []
    props['updatemenus'] = MyUp(safe_up)
    print('set props to MyUp')
    print('after, up type:', type(getattr(layout,'updatemenus')))
    print('getattr up __iter__ ret:', getattr(getattr(layout,'updatemenus'),'__iter__', None))
    # Safe access: only call len when safe
    iter_attr = getattr(getattr(layout,'updatemenus'),'__iter__', [])
    if callable(iter_attr):
        try:
            val = iter_attr()
            from collections.abc import Sized
            try:
                if isinstance(val, Sized):
                    print('len(getattr(...)):', len(val))
                else:
                    # Fallback: attempt to exhaust iterator to count elements only if it is iterable
                    from collections.abc import Iterable
                    if isinstance(val, Iterable):
                        try:
                            ln = len(list(val))
                            print('len(getattr(...)):', ln)
                        except Exception:
                            print('iter returned a non-len-able iterator')
                    else:
                        print('iter returned a non-len-able iterator')
            except TypeError:
                print('iter returned a non-len-able iterator')
        except Exception:
            print('calling __iter__ failed')
    else:
        from collections.abc import Sized
        if isinstance(iter_attr, Sized):
            ln = len(iter_attr)
            print('len(getattr(...)):', ln)
        else:
            print('no len available')
