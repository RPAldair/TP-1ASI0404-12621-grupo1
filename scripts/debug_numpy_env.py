import os
from pathlib import Path

venv_np = Path('.').resolve() / '.venv' / 'Lib' / 'site-packages' / 'numpy'
print('Checking', venv_np)
if venv_np.exists():
    for root, dirs, files in os.walk(venv_np):
        level = root.replace(str(venv_np), '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        for f in files[:20]:
            print(f"{indent}  - {f}")
        # stop after a bit
        if level > 2:
            break
else:
    print('No numpy folder found in venv site-packages')
