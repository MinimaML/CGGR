import re
import sys
import subprocess
import os

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
        r'_': '', # Remove subscript underscores for readability
    }
    
    def replace_math_match(match):
        content = match.group(1)
        # Apply replacements
        for tex, uni in replacements.items():
            content = content.replace(tex, uni)
        # Cleanup extra latex chars
        content = content.replace('{', '').replace('}', '').replace('\\', '')
        return content

    # Replace inline math $...$
    text = re.sub(r'\$(.*?)\$', replace_math_match, text)
    
    # Replace block math $$...$$
    # Render as blockquote context for visual distinction
    text = re.sub(r'\$\$(.*?)\$\$', lambda m: f'\n> {replace_math_match(m)}\n', text, flags=re.DOTALL)
    
    return text

def main():
    input_file = 'CGGR_WHITEPAPER.md'
    temp_file = 'CGGR_WHITEPAPER_PRINT.md'
    output_pdf = 'CGGR_WHITEPAPER.pdf'

    print(f"Reading {input_file}...")
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        sys.exit(1)

    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    print("Preprocessing LaTeX symbols to Unicode...")
    processed_text = preprocess_latex_to_unicode(text)

    with open(temp_file, 'w', encoding='utf-8') as f:
        f.write(processed_text)
    
    print(f"Temporary file {temp_file} created.")
    print("Running mdpdf...")

    # Use sys.executable to ensure we use the active python environment
    # invoking mdpdf.cli module directly to avoid PATH issues
    cmd = [sys.executable, "-m", "mdpdf.cli", "-o", output_pdf, temp_file]
    
    try:
        subprocess.check_call(cmd)
        print(f"\nSUCCESS: Generated {output_pdf}")
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: mdpdf processing failed with output:\n{e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: Failed to run mdpdf: {e}")
        sys.exit(1)
    finally:
        # Optional: clean up temp file, or leave it for debugging
        if os.path.exists(temp_file):
            print(f"Cleaning up {temp_file}...")
            os.remove(temp_file)

if __name__ == "__main__":
    main()
