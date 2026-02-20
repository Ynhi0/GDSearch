import ast
import pathlib


def _get_func_arg_names(source: str, func_name: str):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return [a.arg for a in node.args.args]
    return []


def test_run_2d_experiments_accepts_quick():
    src = open('run_all_kaggle.py', 'r', encoding='utf-8', errors='replace').read()
    args = _get_func_arg_names(src, "run_2d_experiments")
    assert "quick" in args, "run_2d_experiments must accept a 'quick' kwarg"


def test_run_robustness_analysis_accepts_quick():
    src = open('run_all_kaggle.py', 'r', encoding='utf-8', errors='replace').read()
    args = _get_func_arg_names(src, "run_robustness_analysis")
    assert "quick" in args, "run_robustness_analysis must accept a 'quick' kwarg"


def test_run_sam_sensitivity_accepts_quick():
    src = open('run_all_kaggle.py', 'r', encoding='utf-8', errors='replace').read()
    args = _get_func_arg_names(src, "run_sam_sensitivity")
    assert "quick" in args, "run_sam_sensitivity must accept a 'quick' kwarg"


def test_run_ablation_study_accepts_quick():
    src = open('run_all_kaggle.py', 'r', encoding='utf-8', errors='replace').read()
    args = _get_func_arg_names(src, "run_ablation_study")
    assert "quick" in args, "run_ablation_study must accept a 'quick' kwarg"


def test_experiment_function_map_functions_accept_required_parallel_args():
    """Ensure every function referenced in experiment_function_map accepts the
    set of arguments passed by the parallel dispatcher (or accepts **kwargs).

    This prevents TypeError when the parallel runner forwards common args like
    `quick`, `skip_tuning`, `profiler`, `tracker`, `checkpoint_manager`, etc.
    """
    src = open('run_all_kaggle.py', 'r', encoding='utf-8', errors='replace').read()

    # Find the experiment_function_map literal and extract function names
    import re, ast
    m = re.search(r"experiment_function_map\s*=\s*\{([\s\S]*?)\}\n", src)
    assert m, "Could not find experiment_function_map in run_all_kaggle.py"
    body = m.group(1)

    # Extract RHS identifiers like run_2d_experiments
    fn_names = re.findall(r"'?[a-zA-Z0-9_\-]+'?\s*:\s*([a-zA-Z0-9_]+)", body)
    fn_names = sorted(set(fn_names))

    required = ['results_dir','seeds','quick','skip_tuning','profiler','tracker','checkpoint_manager','resume','resume_behavior']

    def _get_details(source: str, func: str):
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func:
                args = [a.arg for a in node.args.args]
                has_kwargs = node.args.kwarg is not None
                return args, has_kwargs
        return [], False

    missing_map = {}
    for fn in fn_names:
        args, has_kwargs = _get_details(src, fn)
        if has_kwargs:
            continue
        missing = [r for r in required if r not in args]
        if missing:
            missing_map[fn] = missing

    assert not missing_map, f"Some experiment functions are missing required parallel args or **kwargs: {missing_map}"