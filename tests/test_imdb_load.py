"""Test IMDB dataset loading with different methods."""
import sys

print("Testing IMDB dataset loading...")

# Method 1: Try standard load
try:
    from datasets import load_dataset
    print("\n[1] Trying standard 'imdb' dataset...")
    data = load_dataset('imdb', split='train[:10]')
    print(f"✓ Successfully loaded {len(data)} samples")
    sys.exit(0)
except Exception as e:
    print(f"✗ Failed: {str(e)[:100]}")

# Method 2: Try with download_mode
try:
    print("\n[2] Trying with download_mode='force_redownload'...")
    data = load_dataset('imdb', split='train[:10]', download_mode='force_redownload')
    print(f"✓ Successfully loaded {len(data)} samples")
    sys.exit(0)
except Exception as e:
    print(f"✗ Failed: {str(e)[:100]}")

# Method 3: Try stanfordnlp version
try:
    print("\n[3] Trying 'stanfordnlp/imdb'...")
    data = load_dataset('stanfordnlp/imdb', split='train[:10]')
    print(f"✓ Successfully loaded {len(data)} samples")
    sys.exit(0)
except Exception as e:
    print(f"✗ Failed: {str(e)[:100]}")

# Method 4: Try with trust_remote_code
try:
    print("\n[4] Trying with trust_remote_code=True...")
    data = load_dataset('imdb', split='train[:10]', trust_remote_code=True)
    print(f"✓ Successfully loaded {len(data)} samples")
    sys.exit(0)
except Exception as e:
    print(f"✗ Failed: {str(e)[:100]}")

print("\n✗ All methods failed - IMDB dataset cannot be loaded")
print("This is a known issue with Python 3.13 + fsspec + huggingface datasets")
print("Recommendation: Use synthetic data fallback or downgrade to Python 3.11/3.12")
sys.exit(1)
