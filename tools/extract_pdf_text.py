import sys
from pathlib import Path
try:
    from PyPDF2 import PdfReader
except Exception as e:
    print('PyPDF2 not installed: tried import error', e)
    sys.exit(1)

p=Path('docs') / 'Đăng Ký Đề Tài NCKH.pdf'
if not p.exists():
    print('file missing', p)
    sys.exit(1)

reader=PdfReader(str(p))
text=[]
for i,page in enumerate(reader.pages):
    try:
        txt = page.extract_text() or ''
    except Exception as e:
        txt = ''
    text.append(txt)
full='\n'.join(text)
print(full[:20000])
