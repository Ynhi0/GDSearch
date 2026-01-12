#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract text from research proposal PDF."""
import PyPDF2
import sys
import logging
from pathlib import Path
import os

logging.basicConfig(level=logging.INFO, format='%(message)s')

def extract_proposal():
    """Extract and print proposal text."""
    project_root = Path(__file__).resolve().parent.parent
    pdf_path = Path(os.environ.get('EXTRACT_PROPOSAL_PDF', project_root / 'docs' / 'Đăng Ký Đề Tài NCKH.pdf'))
    output_path = Path(os.environ.get('EXTRACT_PROPOSAL_OUTPUT', project_root / 'proposal_extracted.txt'))

    try:
        with open(pdf_path, 'rb') as f:
            pdf = PyPDF2.PdfReader(f)
            text_parts = []

            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                text_parts.append(f"\n{'='*70}\nPAGE {i+1}\n{'='*70}\n{text}")

            full_text = '\n'.join(text_parts)
            logging.info(full_text)

            # Write to file for persistence
            with open(output_path, 'w', encoding='utf-8') as out_f:
                out_f.write(full_text)
            logging.info(f"Saved to: {output_path}")

    except Exception as e:
        logging.exception("ERROR extracting proposal: %s", e)
        return 1

    return 0

if __name__ == '__main__':
    sys.exit(extract_proposal())
