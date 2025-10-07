import sys
import os

def main():
    print('executable:', sys.executable)
    print('cwd:', os.getcwd())
    print('\n-- sys.path --')
    for p in sys.path:
        print(p)

    print('\n-- search for "numpy" folder in sys.path entries --')
    for p in sys.path:
        try:
            if p and os.path.isdir(p) and os.path.exists(os.path.join(p, 'numpy')):
                print('FOUND numpy in sys.path entry:', p)
        except Exception:
            pass

    print('\n-- search project tree for numpy files/folders --')
    for root, dirs, files in os.walk('.'):
        if 'numpy' in dirs or 'numpy.py' in files:
            print('FOUND in:', root)
            # list a few items for context
            print(' dirs sample:', dirs[:5])
            print(' files sample:', files[:5])

if __name__ == '__main__':
    main()
