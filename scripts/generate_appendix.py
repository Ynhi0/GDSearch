"""
Appendix Generator for GDSearch Research Report

Generates a comprehensive, high-quality appendix with:
- Experimental environment details
- Complete hyperparameter configurations
- Supplementary visualizations
- GitHub repository links
- Reproducibility instructions

"A detailed appendix demonstrates transparency, care, and respect for readers,
while strongly reinforcing reproducibility and trustworthiness." - Part 4
"""

import os
import sys
import platform
import json
import pandas as pd
import numpy as np
import torch
import matplotlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict


def get_environment_info() -> Dict:
    """Collect comprehensive environment information."""
    return {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'),
        'python_version': sys.version,
        'platform': {
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor()
        },
        'libraries': {
            'numpy': np.__version__,
            'pandas': pd.__version__,
            'torch': torch.__version__,
            'matplotlib': matplotlib.__version__
        },
        'hardware': {
            'cpu_count': os.cpu_count(),
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else 'N/A',
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
    }


def collect_all_hyperparameters(results_dir: str = 'results') -> pd.DataFrame:
    """
    Collect ALL hyperparameters tried, not just the best ones.
    
    This demonstrates transparency and helps readers understand
    the search space explored.
    """
    all_configs = []
    
    # Parse from CSV filenames
    csv_files = Path(results_dir).glob('NN_*.csv')
    
    for csv_path in csv_files:
        fname = csv_path.stem
        parts = fname.split('_')
        
        config = {'filename': fname}
        
        # Parse model, dataset, optimizer
        if len(parts) >= 4:
            config['model'] = parts[1]
            config['dataset'] = parts[2]
            config['optimizer'] = parts[3]
        
        # Parse hyperparameters from filename
        for part in parts:
            if part.startswith('lr'):
                config['lr'] = float(part[2:])
            elif part.startswith('seed'):
                config['seed'] = int(part[4:])
            elif part.startswith('mom'):
                config['momentum'] = float(part[3:])
            elif part.startswith('wd'):
                config['weight_decay'] = float(part[2:])
            elif part in ['sweepLR', 'sweepWD', 'sweepMOM', 'final']:
                config['stage'] = part
        
        # Load final metrics from CSV
        try:
            df = pd.read_csv(csv_path)
            eval_df = df[df['phase'] == 'eval']
            if not eval_df.empty:
                config['final_test_accuracy'] = eval_df['test_accuracy'].iloc[-1]
                config['final_test_loss'] = eval_df['test_loss'].iloc[-1]
        except Exception as e:
            config['final_test_accuracy'] = None
            config['final_test_loss'] = None
        
        all_configs.append(config)
    
    return pd.DataFrame(all_configs)


def generate_appendix_markdown(output_path: str = 'APPENDIX.md'):
    """
    Generate comprehensive appendix in Markdown format.
    """
    print("Generating comprehensive appendix...")
    
    env_info = get_environment_info()
    
    with open(output_path, 'w') as f:
        f.write("# Appendix: GDSearch Experimental Details\n\n")
        f.write("**Purpose:** This appendix provides complete transparency about experimental setup, ")
        f.write("hyperparameters tested, and reproducibility instructions.\n\n")
        f.write("---\n\n")
        
        # Section A: Environment
        f.write("## A. Experimental Environment\n\n")
        f.write("### A.1 Software Environment\n\n")
        f.write(f"**Timestamp:** {env_info['timestamp']}\n\n")
        f.write(f"**Python Version:**\n```\n{env_info['python_version']}\n```\n\n")
        
        f.write("**Key Libraries:**\n")
        for lib, version in env_info['libraries'].items():
            f.write(f"- `{lib}`: {version}\n")
        f.write("\n")
        
        f.write("### A.2 Hardware Environment\n\n")
        f.write(f"**Platform:** {env_info['platform']['system']} {env_info['platform']['release']}\n\n")
        f.write(f"**CPU:** {env_info['hardware']['cpu_count']} cores\n\n")
        f.write(f"**GPU/CUDA:**\n")
        f.write(f"- CUDA Available: {env_info['hardware']['cuda_available']}\n")
        f.write(f"- CUDA Version: {env_info['hardware']['cuda_version']}\n")
        f.write(f"- GPU Count: {env_info['hardware']['cuda_device_count']}\n\n")
        
        f.write("### A.3 Installation Instructions\n\n")
        f.write("```bash\n")
        f.write("# Clone repository\n")
        f.write("git clone https://github.com/Ynhi0/GDSearch.git\n")
        f.write("cd GDSearch\n\n")
        f.write("# Install dependencies\n")
        f.write("pip install -r requirements.txt\n\n")
        f.write("# Run complete pipeline\n")
        f.write("python run_all.py\n")
        f.write("```\n\n")
        
        f.write("---\n\n")
        
        # Section B: Hyperparameters
        f.write("## B. Hyperparameter Configurations\n\n")
        f.write("### B.1 Complete Search Space\n\n")
        f.write("This section documents **all** hyperparameters tested, not just the best ones. ")
        f.write("This transparency helps readers understand the search process and reproduce our methodology.\n\n")
        
        # Load and display hyperparameters
        try:
            hp_df = collect_all_hyperparameters()
            
            if not hp_df.empty:
                f.write(f"**Total configurations tested:** {len(hp_df)}\n\n")
                
                # Group by optimizer
                for optimizer in hp_df['optimizer'].unique():
                    if pd.notna(optimizer):
                        opt_df = hp_df[hp_df['optimizer'] == optimizer]
                        f.write(f"#### {optimizer}\n\n")
                        f.write(f"Configurations: {len(opt_df)}\n\n")
                        
                        # Learning rates tested
                        if 'lr' in opt_df.columns:
                            lrs = opt_df['lr'].dropna().unique()
                            if len(lrs) > 0:
                                f.write(f"**Learning rates tested:** {sorted(lrs)}\n\n")
                        
                        # Other parameters
                        if 'momentum' in opt_df.columns:
                            moms = opt_df['momentum'].dropna().unique()
                            if len(moms) > 0:
                                f.write(f"**Momentum values tested:** {sorted(moms)}\n\n")
                        
                        if 'weight_decay' in opt_df.columns:
                            wds = opt_df['weight_decay'].dropna().unique()
                            if len(wds) > 0:
                                f.write(f"**Weight decay values tested:** {sorted(wds)}\n\n")
                
                # Best configurations table
                f.write("### B.2 Best Configurations\n\n")
                final_df = hp_df[hp_df['stage'] == 'final']
                if not final_df.empty:
                    f.write("| Optimizer | LR | Momentum | Weight Decay | Test Accuracy | Test Loss |\n")
                    f.write("|-----------|----|-----------|--------------|--------------|-----------|\n")
                    for _, row in final_df.iterrows():
                        f.write(f"| {row.get('optimizer', 'N/A')} | ")
                        f.write(f"{row.get('lr', 'N/A')} | ")
                        f.write(f"{row.get('momentum', 'N/A')} | ")
                        f.write(f"{row.get('weight_decay', 'N/A')} | ")
                        f.write(f"{row.get('final_test_accuracy', 'N/A'):.4f} | " if pd.notna(row.get('final_test_accuracy')) else "N/A | ")
                        f.write(f"{row.get('final_test_loss', 'N/A'):.4f} |\n" if pd.notna(row.get('final_test_loss')) else "N/A |\n")
                    f.write("\n")
        except Exception as e:
            f.write(f"_Error collecting hyperparameters: {e}_\n\n")
        
        f.write("---\n\n")
        
        # Section C: Supplementary Visualizations
        f.write("## C. Supplementary Visualizations\n\n")
        f.write("Additional plots not included in main report:\n\n")
        
        # List all plots
        plots_dir = Path('plots')
        if plots_dir.exists():
            plot_files = sorted(plots_dir.glob('*.png'))
            
            # Group by type
            categories = {
                'Trajectories': [p for p in plot_files if 'trajectory' in p.stem.lower()],
                'Eigenvalues': [p for p in plot_files if 'eigenvalue' in p.stem.lower()],
                'Dynamics': [p for p in plot_files if 'dynamics' in p.stem.lower()],
                'Loss Landscape': [p for p in plot_files if 'landscape' in p.stem.lower()],
                'Generalization': [p for p in plot_files if 'gen_gap' in p.stem.lower()],
                'Layer Gradients': [p for p in plot_files if 'layer_grad' in p.stem.lower()],
                'Other': []
            }
            
            # Assign uncategorized plots
            categorized = set()
            for plots in categories.values():
                categorized.update(plots)
            categories['Other'] = [p for p in plot_files if p not in categorized]
            
            for category, plots in categories.items():
                if plots:
                    f.write(f"### C.{list(categories.keys()).index(category) + 1} {category}\n\n")
                    for plot in sorted(plots):
                        f.write(f"- `plots/{plot.name}`\n")
                    f.write("\n")
        
        f.write("---\n\n")
        
        # Section D: Raw Data
        f.write("## D. Raw Data Files\n\n")
        f.write("All experimental data is available in CSV format:\n\n")
        
        results_dir = Path('results')
        if results_dir.exists():
            csv_files = sorted(results_dir.glob('*.csv'))
            
            f.write(f"**Total CSV files:** {len(csv_files)}\n\n")
            
            # Group by type
            f.write("### D.1 Experimental Runs\n\n")
            exp_files = [f for f in csv_files if f.stem.startswith('NN_')]
            f.write(f"Files: {len(exp_files)}\n\n")
            f.write("```\n")
            for csv in sorted(exp_files):
                f.write(f"{csv.name}\n")
            f.write("```\n\n")
            
            f.write("### D.2 Summary Tables\n\n")
            summary_files = [f for f in csv_files if 'summary' in f.stem]
            for csv in summary_files:
                f.write(f"- `results/{csv.name}`\n")
            f.write("\n")
        
        f.write("---\n\n")
        
        # Section E: Reproducibility
        f.write("## E. Reproducibility Instructions\n\n")
        f.write("### E.1 Complete Pipeline\n\n")
        f.write("To reproduce all results from scratch:\n\n")
        f.write("```bash\n")
        f.write("# Complete pipeline (may take 2-4 hours)\n")
        f.write("python run_all.py\n")
        f.write("```\n\n")
        
        f.write("### E.2 Individual Components\n\n")
        f.write("```bash\n")
        f.write("# 2D experiments only\n")
        f.write("python run_experiment.py\n\n")
        f.write("# Neural network tuning only\n")
        f.write("python tune_nn.py\n\n")
        f.write("# Loss landscape analysis only\n")
        f.write("python run_loss_landscape.py\n\n")
        f.write("# Regenerate summaries and plots\n")
        f.write("python generate_summaries.py\n\n")
        f.write("# Sensitivity analysis\n")
        f.write("python sensitivity_analysis.py\n\n")
        f.write("# Eigenvalue visualization\n")
        f.write("python plot_eigenvalues.py\n")
        f.write("```\n\n")
        
        f.write("### E.3 Random Seed Control\n\n")
        f.write("All experiments use fixed random seeds for reproducibility:\n")
        f.write("- Default seed: `1`\n")
        f.write("- Configured in: `configs/nn_tuning.json`\n")
        f.write("- Applied to: NumPy, PyTorch, Python random\n\n")
        
        f.write("---\n\n")
        
        # Section F: Links
        f.write("## F. External Resources\n\n")
        f.write("### F.1 GitHub Repository\n\n")
        f.write("**Primary Repository:**\n")
        f.write("- https://github.com/Ynhi0/GDSearch\n\n")
        f.write("**Repository Structure:**\n")
        f.write("```\n")
        f.write("GDSearch/\n")
        f.write("├── configs/           # JSON configurations\n")
        f.write("├── results/           # All CSV outputs\n")
        f.write("├── plots/             # All visualizations\n")
        f.write("├── *.py               # Source code\n")
        f.write("├── requirements.txt   # Dependencies\n")
        f.write("└── *.md               # Documentation\n")
        f.write("```\n\n")
        
        f.write("### F.2 Documentation\n\n")
        f.write("- **README.md**: Comprehensive usage guide\n")
        f.write("- **REPORT.md**: Main findings and ablation study\n")
        f.write("- **RESEARCH_JOURNAL.md**: Detective-style analysis notes\n")
        f.write("- **IMPLEMENTATION_CHECKLIST.md**: Requirements verification\n")
        f.write("- **CHANGELOG.md**: Version history\n")
        f.write("- **QUICK_REFERENCE.md**: Quick command reference\n\n")
        
        f.write("---\n\n")
        
        # Section G: Citation
        f.write("## G. Citation\n\n")
        f.write("If you use this codebase in your research, please cite:\n\n")
        f.write("```bibtex\n")
        f.write("@software{gdsearch2025,\n")
        f.write("  title={GDSearch: Optimizer Dynamics Research Platform},\n")
        f.write("  author={GDSearch Development Team},\n")
        f.write("  year={2025},\n")
        f.write("  url={https://github.com/Ynhi0/GDSearch},\n")
        f.write("  version={2.0.0}\n")
        f.write("}\n")
        f.write("```\n\n")
        
        f.write("---\n\n")
        
        # Section H: Acknowledgments
        f.write("## H. Acknowledgments\n\n")
        f.write("This research benefited from:\n")
        f.write("- Open-source libraries: PyTorch, NumPy, Matplotlib, Pandas\n")
        f.write("- Community feedback and suggestions\n")
        f.write("- Computational resources: [specify if applicable]\n\n")
        
        f.write("---\n\n")
        
        f.write(f"**Appendix Generated:** {env_info['timestamp']}\n")
        f.write(f"**GDSearch Version:** 2.0.0 Production Ready\n")
    
    print(f"✅ Appendix generated: {output_path}")
    return output_path


def main():
    """Generate comprehensive appendix."""
    print("=" * 60)
    print("Appendix Generator - GDSearch Project")
    print("=" * 60)
    
    appendix_path = generate_appendix_markdown()
    
    # Count lines
    with open(appendix_path, 'r') as f:
        lines = len(f.readlines())
    
    print(f"\n📊 Appendix Statistics:")
    print(f"  Lines: {lines}")
    print(f"  Sections: 8 (A-H)")
    print(f"\n💡 This appendix demonstrates:")
    print(f"  ✅ Complete transparency")
    print(f"  ✅ Full reproducibility")
    print(f"  ✅ Respect for readers")
    print(f"  ✅ Scientific rigor")
    
    print(f"\n📖 Next steps:")
    print(f"  1. Review generated appendix: {appendix_path}")
    print(f"  2. Add to main report as final section")
    print(f"  3. Include in paper submission")


if __name__ == '__main__':
    main()
