import ast
import pathlib


def _get_func_arg_names(source: str, func_name: str):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return [a.arg for a in node.args.args]
    return []


def test_run_2d_experiments_accepts_quick():
    src = pathlib.Path("run_all_kaggle.py").read_text()
    args = _get_func_arg_names(src, "run_2d_experiments")
    assert "quick" in args, "run_2d_experiments must accept a 'quick' kwarg"


def test_run_robustness_analysis_accepts_quick():
    src = pathlib.Path("run_all_kaggle.py").read_text()
    args = _get_func_arg_names(src, "run_robustness_analysis")
    assert "quick" in args, "run_robustness_analysis must accept a 'quick' kwarg"


def test_run_sam_sensitivity_accepts_quick():
    src = pathlib.Path("run_all_kaggle.py").read_text()
    args = _get_func_arg_names(src, "run_sam_sensitivity")
    assert "quick" in args, "run_sam_sensitivity must accept a 'quick' kwarg"


def test_run_ablation_study_accepts_quick():
    src = pathlib.Path("run_all_kaggle.py").read_text()
    args = _get_func_arg_names(src, "run_ablation_study")
    assert "quick" in args, "run_ablation_study must accept a 'quick' kwarg"


def test_experiment_function_map_functions_accept_quick():
    """Ensure every function referenced in experiment_function_map accepts `quick`.

    This avoids regressions where the parallel dispatcher passes `quick` and
    the callee raises TypeError.
    """
    src = pathlib.Path("run_all_kaggle.py").read_text()

    # Find the experiment_function_map literal and extract function names
    import re
    m = re.search(r"experiment_function_map\s*=\s*\{([\s\S]*?)\}\n", src)
    assert m, "Could not find experiment_function_map in run_all_kaggle.py"
    body = m.group(1)

    # Extract RHS identifiers like run_2d_experiments
    fn_names = re.findall(r"'?[a-zA-Z0-9_\-]+'?\s*:\s*([a-zA-Z0-9_]+)", body)
    # Deduplicate
    fn_names = sorted(set(fn_names))

    missing_quick = []
    for fn in fn_names:
        args = _get_func_arg_names(src, fn)
        if 'quick' not in args:
            missing_quick.append(fn)

    assert not missing_quick, f"The following experiment functions are missing 'quick': {missing_quick}"