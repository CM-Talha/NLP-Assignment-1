import subprocess
import os
import shutil
import pymupdf

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

def get_chrome_path():
    mac_paths = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        os.path.expanduser("~/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    ]
    for p in mac_paths:
        if os.path.exists(p):
            return p
    win_paths = [
        r'C:\Program Files\Google\Chrome\Application\chrome.exe',
        r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe',
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe")
    ]
    for p in win_paths:
        if os.path.exists(p):
            return p
    for cmd in ['google-chrome', 'google-chrome-stable', 'chromium', 'chromium-browser']:
        found = shutil.which(cmd)
        if found:
            return found
    return None

chrome_path = get_chrome_path()
if chrome_path:
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

    reader = pymupdf.open('scratch/test.pdf')
    print('Generated PDF Page count:', len(reader))
    text = reader[0].get_text()
    print('Extracted text preview:', repr(text[:100]))
else:
    print('Chrome not found on this system.')
