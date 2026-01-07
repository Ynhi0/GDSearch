#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract text from research proposal PDF."""
import PyPDF2
import sys

def extract_proposal():
    """Extract and print proposal text."""
    pdf_path = r'c:\Users\MPhuc\Desktop\GDSearch\docs\Đăng Ký Đề Tài NCKH.pdf'
    
    try:
        with open(pdf_path, 'rb') as f:
            pdf = PyPDF2.PdfReader(f)
            text_parts = []
            
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                text_parts.append(f"\n{'='*70}\nPAGE {i+1}\n{'='*70}\n{text}")
            
            full_text = '\n'.join(text_parts)
            print(full_text, flush=True)
            
            # Write to file for persistence
            output_path = r'c:\Users\MPhuc\Desktop\GDSearch\proposal_extracted.txt'
            with open(output_path, 'w', encoding='utf-8') as out_f:
                out_f.write(full_text)
            print(f"\n\nSaved to: {output_path}", file=sys.stderr)
            
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(extract_proposal())
