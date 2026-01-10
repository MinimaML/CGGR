import re

def preprocess_latex_to_unicode(text):
    # Dictionary of common LaTeX to Unicode replacements
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
        r'_': '', # Subscripts - simplistic removal for plain text
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
    text = re.sub(r'\$\$(.*?)\$\$', lambda m: f'\n> {replace_math_match(m)}\n', text, flags=re.DOTALL)
    
    return text

def create_printable_md(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    processed_text = preprocess_latex_to_unicode(text)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(processed_text)
    
    print(f"Created {output_file} with Unicode math.")

if __name__ == "__main__":
    create_printable_md('CGGR_WHITEPAPER.md', 'CGGR_WHITEPAPER_PRINT.md')
