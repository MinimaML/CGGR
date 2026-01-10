import re
import sys
import subprocess
import os
import glob

def preprocess_latex_to_unicode(text):
    """
    Replaces common LaTeX math symbols with their Unicode equivalents
    to allow mdpdf (which doesn't support LaTeX) to render them correctly.
    """
    replacements = {
        r'\\theta': 'θ',
        r'\\nabla': '∇',
        r'\\mathcal{L}': 'L',
        r'\\ell': 'ℓ',
        r'\\sum': '∑',
        r'\\mathbb{I}': 'I',
        r'\\mathbb{E}': 'E',
        r'\\tau': 'τ',
        r'\\ge': '≥',
        r'\\geq': '≥',
        r'\\approx': '≈',
        r'\\ll': '≪',
        r'\\cdot': '·',
        r'\\times': '×',
        r'\\propto': '∝',
        r'\\mathcal{O}': 'O',
        r'\\mathcal{H}': 'H',
        r'\\Delta': 'Δ',
        r'\\partial': '∂',
        r'\\in': '∈',
        r'\\{': '{',
        r'\\}': '}',
        r'\|\|': '||',
        r'\\phi': 'φ',
        r'\\rho': 'ρ',
        r'\\sqrt': '√',
        r'_': '', 
    }
    
    def replace_math_match(match):
        content = match.group(1)
        for tex, uni in replacements.items():
            content = content.replace(tex, uni)
        content = content.replace('{', '').replace('}', '').replace('\\', '')
        return content

    # Replace inline math $...$
    text = re.sub(r'\$(.*?)\$', replace_math_match, text)
    
    # Replace block math $$...$$
    text = re.sub(r'\$\$(.*?)\$\$', lambda m: f'\n> {replace_math_match(m)}\n', text, flags=re.DOTALL)
    
    return text

def main():
    output_pdf = 'CGGR_The_Foundations_of_Modern_Intelligence.pdf'
    temp_file = 'temp_book_complete.md'
    
    # Order of chapters
    chapters = [
        "book/chapter_1_gradient_flow.md",
        "book/chapter_2_transformer_anatomy.md",
        "book/chapter_3_memory_efficiency.md",
        "book/chapter_4_parameter_efficiency.md",
        "book/chapter_5_sparse_architectures.md",
        "book/chapter_6_scaling_laws.md",
        "book/chapter_7_evaluation.md",
        "book/chapter_8_general_intelligence.md"
    ]

    full_text = "# The Foundations of Modern Intelligence\n\nA Technical Guide by CGGR Team\n\n---\n\n"

    for chapter_path in chapters:
        if os.path.exists(chapter_path):
            print(f"Adding {chapter_path}...")
            with open(chapter_path, 'r', encoding='utf-8') as f:
                full_text += f.read() + "\n\n<div style='page-break-after: always;'></div>\n\n"
        else:
            print(f"Warning: {chapter_path} not found.")

    print("Preprocessing LaTeX symbols to Unicode...")
    processed_text = preprocess_latex_to_unicode(full_text)

    with open(temp_file, 'w', encoding='utf-8') as f:
        f.write(processed_text)

    print("Running mdpdf...")
    cmd = [sys.executable, "-m", "mdpdf.cli", "-o", output_pdf, temp_file]
    
    try:
        subprocess.check_call(cmd)
        print(f"\nSUCCESS: Generated {output_pdf}")
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: mdpdf processing failed with output:\n{e}")
    finally:
        if os.path.exists(temp_file):
            print(f"Cleaning up {temp_file}...")
            os.remove(temp_file)

if __name__ == "__main__":
    main()
