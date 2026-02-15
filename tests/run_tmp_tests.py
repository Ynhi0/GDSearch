import importlib
m = importlib.import_module('tests.test_experiment_tracker_resume')
print('module loaded:', m)
# Run tests manually
for name in dir(m):
    if name.startswith('test_'):
        print('running', name)
        getattr(m, name)()
print('done')
