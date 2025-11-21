import os
import sys
import glob
import shutil
import subprocess

# --- SETUP ---
print("🔍 Finding Repo...")
POSSIBLE_PATHS = ["/kaggle/input/gdsearch-repo/GDSearch", "/kaggle/input/gdsearch-repo", "/kaggle/input/gdsearch-code"]
REPO_PATH = None
for path in POSSIBLE_PATHS:
    if os.path.exists(os.path.join(path, "src")): REPO_PATH = path; break
if not REPO_PATH:
    found = glob.glob("/kaggle/input/**/src", recursive=True)
    if found: REPO_PATH = os.path.dirname(found[0])

if REPO_PATH:
    os.environ['PYTHONPATH'] = f"{REPO_PATH}:{os.environ.get('PYTHONPATH', '')}"
    sys.path.append(REPO_PATH)
    print(f"✅ Repo: {REPO_PATH}")
else:
    raise RuntimeError("Repo not found!")

print("\n📦 Installing libs...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "torchtext", "portalocker", "medpy", "pandas", "matplotlib", "--quiet"])

# Output Dirs
BASE_OUT = "/kaggle/working/thesis_output"
DIRS = {k: f"{BASE_OUT}/{k}" for k in ['results', 'plots', 'checkpoints', 'tables']}
for d in DIRS.values(): os.makedirs(d, exist_ok=True)

def run_cmd(desc, cmd):
    print(f"\n>>> 🚀 {desc}...")
    try: subprocess.run(cmd, shell=True, check=True); print("✅ OK")
    except: print("⚠️ FAIL (Ignored)")

# --- EXECUTION ---

# 1. Basic Benchmarks
run_cmd("MNIST Benchmark", f"python {REPO_PATH}/kaggle/mnist_benchmark/run_mnist.py --batch-size-sweep '64,1024' --epochs 20 --results-dir {DIRS['results']}/mnist --ckpt-dir {DIRS['checkpoints']}")
run_cmd("CIFAR-10 Benchmark", f"python {REPO_PATH}/kaggle/cifar10_benchmark/run_cifar10.py --epochs 20 --results-dir {DIRS['results']}/cifar10 --ckpt-dir {DIRS['checkpoints']}")

# 2. Ablation Studies (New & Old)
run_cmd("SAM Sensitivity (Rho)", f"python {REPO_PATH}/kaggle/resnet18_cifar10.py --optimizer SAM_SGD --rho-sweep '0.01,0.05,0.1,0.2' --epochs 15 --results-dir {DIRS['results']}/sam_sensitivity")
run_cmd("Component Ablation", f"python {REPO_PATH}/scripts/run_nn_ablation.py --dataset MNIST --epochs 20 --results-dir {DIRS['results']}/components")
run_cmd("Robustness (Seed)", f"python {REPO_PATH}/src/experiments/run_initial_condition_robustness.py --model SimpleMLP --dataset MNIST --optimizers 'SGD,Adam,SAM_SGD' --seeds 5 --results_dir {DIRS['results']}/robustness")

# 3. Applications
run_cmd("NLP Benchmark", f"python {REPO_PATH}/kaggle/nlp_benchmark/run_nlp.py --epochs 15 --results-dir {DIRS['results']}/nlp")
run_cmd("Medical Seg", f"python {REPO_PATH}/kaggle/medical_benchmark/run_seg.py --epochs 20 --results-dir {DIRS['results']}/medical")

# 4. Visualization & Hessian (DÙNG FILE PATCH MỚI TẠO)
print("\n>>> 🎨 Visualizing & Computing Hessian...")
adam_ckpts = glob.glob(f"{DIRS['checkpoints']}/*Adam_*.pt")
sam_ckpts = glob.glob(f"{DIRS['checkpoints']}/*SAM_*.pt")

if adam_ckpts and sam_ckpts:
    # Vẽ và tính Hessian cho Adam
    run_cmd("Adam Landscape", f"python patch_landscape.py --ckpt '{adam_ckpts[0]}' --output_dir {DIRS['plots']} --compute_hessian")
    # Vẽ và tính Hessian cho SAM
    run_cmd("SAM Landscape", f"python patch_landscape.py --ckpt '{sam_ckpts[0]}' --output_dir {DIRS['plots']} --compute_hessian")
else:
    print("⚠️ No checkpoints found.")

# 5. Reports
run_cmd("Generating Summaries", f"python {REPO_PATH}/scripts/generate_summaries.py --results_dir {DIRS['results']} --output_dir {DIRS['tables']}")
run_cmd("Generating Tables", f"python {REPO_PATH}/scripts/generate_latex_tables.py --data_dir {DIRS['tables']} --output_dir {DIRS['tables']}")

# 6. Pack
shutil.make_archive("/kaggle/working/FULL_THESIS_DATA", 'zip', BASE_OUT)
print("\n✅ ALL DONE! Download 'FULL_THESIS_DATA.zip'")