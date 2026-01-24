import glob, pandas as pd, os
patterns=['**/*CIFAR*csv','**/*cifar*csv','**/*ResNet18*csv','results/**/*.csv','**/*.csv']
seen=set()
all_files=[]
for p in patterns:
    for f in glob.glob(p, recursive=True):
        if os.path.isfile(f) and f not in seen:
            seen.add(f)
            all_files.append(f)

print('Found files:', len(all_files))
if not all_files:
    print('No CSV files found in repository.')
else:
    for i,f in enumerate(sorted(all_files)[:200],1):
        print(f"\n[{i}] {f}")
        try:
            df=pd.read_csv(f)
            print('  columns:', df.columns.tolist())
            print('  rows:', len(df))
            print('  head:')
            print(df.head(2).to_string(index=False))
            print('  NaNs per col:', df.isna().sum().to_dict())
            acc_cols=[c for c in df.columns if ('test' in c.lower() and 'acc' in c.lower()) or ('test' in c.lower() and 'accuracy' in c.lower())]
            print('  detected test acc cols:', acc_cols)
            for c in acc_cols:
                s=pd.to_numeric(df[c], errors='coerce')
                print(f"    {c}: min={s.min()}, max={s.max()}")
            print('  epoch col present?', any('epoch'==c.lower() or 'epoch' in c.lower() for c in df.columns))
        except Exception as e:
            print('  Error reading file:', e)
