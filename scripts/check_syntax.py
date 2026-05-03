import py_compile, traceback

try:
    py_compile.compile('app.py', doraise=True)
    print('OK')
except Exception:
    traceback.print_exc()
