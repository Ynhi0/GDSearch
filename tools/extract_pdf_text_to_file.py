from PyPDF2 import PdfReader
p='docs/Đăng Ký Đề Tài NCKH.pdf'
reader=PdfReader(p)
text='\n'.join([page.extract_text() or '' for page in reader.pages])
open('docs/proposal_text.txt','w',encoding='utf-8').write(text)
print('wrote docs/proposal_text.txt, length', len(text))
