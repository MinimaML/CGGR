import markdown
from xhtml2pdf import pisa
import sys
import re

def preprocess_latex(text):
    # Dictionary of common LaTeX to Unicode replacements
    # We use direct unicode characters ensuring the PDF engine picks them up
    replacements = {
        r'\\theta': 'θ',
        r'\\nabla': '∇',
        r'\\mathcal{L}': 'L', # Calligraphic L often tricky, using L
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
        r'_': '', # Subscripts - simplistic removal
    }
    
    def replace_math(match):
        content = match.group(1)
        for tex, uni in replacements.items():
            content = content.replace(tex, uni)
        # Cleanup extra latex chars
        content = content.replace('{', '').replace('}', '').replace('\\', '')
        return f'<span style="font-family: DejaVu Sans, sans-serif;">{content}</span>'

    # Replace inline math $...$
    text = re.sub(r'\$(.*?)\$', replace_math, text)
    
    # Replace block math $$...$$
    text = re.sub(r'\$\$(.*?)\$\$', lambda m: f'<div style="text-align:center; margin: 1em 0; font-style: italic;">{replace_math(m)}</div>', text, flags=re.DOTALL)
    
    return text

def convert_md_to_pdf(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # Preprocess LaTeX symbols
    text = preprocess_latex(text)

    # Convert Markdown to HTML
    html_content = markdown.markdown(text, extensions=['extra', 'codehilite', 'tables'])

    # Add academic CSS styling
    css = """
    <style>
        @page {
            size: letter;
            margin: 2cm;
        }
        body { 
            font-family: "Times New Roman", Times, serif; 
            line-height: 1.6; 
            font-size: 11pt;
            color: #000;
        }
        h1 { 
            font-family: Helvetica, Arial, sans-serif; 
            font-size: 18pt; 
            font-weight: bold; 
            text-align: center; 
            margin-bottom: 2em;
        }
        h2 { 
            font-family: Helvetica, Arial, sans-serif; 
            font-size: 14pt; 
            font-weight: bold; 
            margin-top: 1.5em; 
            margin-bottom: 0.5em;
            border-bottom: 1px solid #000;
            padding-bottom: 2px;
        }
        h3 { 
            font-family: Helvetica, Arial, sans-serif; 
            font-size: 12pt; 
            font-weight: bold; 
            margin-top: 1.2em;
        }
        p { 
            margin-bottom: 1em; 
            text-align: justify; 
        }
        table { 
            width: 100%; 
            border-collapse: collapse; 
            margin: 1.5em 0; 
            font-size: 10pt;
        }
        th { 
            border-top: 2px solid #000;
            border-bottom: 1px solid #000;
            font-weight: bold; 
            padding: 8px; 
            text-align: left; 
        }
        td { 
            border-bottom: 1px solid #ddd; 
            padding: 8px; 
            vertical-align: top; 
        }
        blockquote {
            background: #f9f9f9;
            border-left: 10px solid #ccc;
            margin: 1.5em 10px;
            padding: 0.5em 10px;
            font-style: italic;
        }
        code { 
            font-family: "Courier New", Courier, monospace; 
            background-color: #f4f4f4; 
            padding: 2px 4px;
            font-size: 90%;
        }
        .abstract {
            margin: 0 4em 2em 4em;
            font-style: italic;
            font-size: 10pt;
        }
    </style>
    """

    # Structure the HTML
    full_html = f"""
    <html>
    <head>
        <meta charset="utf-8">
        {css}
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """

    # Generate PDF
    with open(output_file, "wb") as pdf_file:
        pisa_status = pisa.CreatePDF(full_html, dest=pdf_file)

    if pisa_status.err:
        print(f"Error converting {input_file} to PDF")
        return False
    return True

if __name__ == "__main__":
    success = convert_md_to_pdf('CGGR_WHITEPAPER.md', 'CGGR_WHITEPAPER.pdf')
    if success:
        print("Successfully created CGGR_WHITEPAPER.pdf")
    else:
        sys.exit(1)
