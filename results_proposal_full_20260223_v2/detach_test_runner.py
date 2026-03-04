import time
from pathlib import Path

time.sleep(45)
Path("results_proposal_full_20260223_v2/detach_test.txt").write_text("done")
