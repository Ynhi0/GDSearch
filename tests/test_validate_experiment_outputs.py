from pathlib import Path
import json
import tempfile
from scripts.validate_experiment_outputs import validate_results_dir


def test_validator_detects_issues(tmp_path):
    # Create directory with various CSVs
    bad1 = tmp_path / 'empty.csv'
    bad1.write_text('')

    bad2 = tmp_path / 'noepoch.csv'
    bad2.write_text('final_test_acc\n0.9\n')

    good = tmp_path / 'ts.csv'
    good.write_text('epoch,final_test_acc\n1,0.85\n2,0.9\n')

    issues = validate_results_dir(tmp_path)
    assert str(bad1) in issues['empty_files']
    assert str(bad2) in issues['missing_epoch']
    # good should not be in any list
    assert str(good) not in issues['missing_epoch']
