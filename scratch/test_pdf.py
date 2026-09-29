import subprocess
import pypdf

html_content = '''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
@page { size: A4; margin: 15mm; }
body { font-family: 'Segoe UI', Arial, sans-serif; }
h1 { color: #1a365d; }
.urdu { font-family: 'Urdu Typesetting', 'Segoe UI', Tahoma; direction: rtl; font-size: 16px; }
</style>
</head>
<body>
<h1>PDF Generation Test</h1>
<p>This is a test of headless Chromium printing to PDF with native Urdu script:</p>
<p class="urdu">یہ ایک ٹیسٹ ہے: کیا زبردست نظام ہے! 😂</p>
</body>
</html>'''

with open('scratch/test.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

chrome_path = r'C:\Program Files\Google\Chrome\Application\chrome.exe'
cmd = [
    chrome_path,
    '--headless=new',
    '--disable-gpu',
    '--print-to-pdf=scratch/test.pdf',
    '--no-pdf-header-footer',
    'scratch/test.html'
]

res = subprocess.run(cmd, capture_output=True, text=True)
print('Return code:', res.returncode)

reader = pypdf.PdfReader('scratch/test.pdf')
print('Generated PDF Page count:', len(reader.pages))
text = reader.pages[0].extract_text()
print('Extracted text preview:', repr(text[:100]))
