import ast, traceback
p='app.py'
try:
    src=open(p,'r',encoding='utf-8').read()
    ast.parse(src)
    print('AST OK')
except SyntaxError as e:
    print('SyntaxError:', e)
    traceback.print_exc()
except Exception:
    traceback.print_exc()
