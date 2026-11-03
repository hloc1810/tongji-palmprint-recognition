"""Verify that formatting retained the benchmark Python modules' ASTs (Python 3.11)."""
import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[2]
original = json.loads((root / 'outputs/palmnet_m5/ast_before_format.json').read_text(encoding='utf-8'))
paths = [
    'src/palmprint/training/palmnet.py',
    'src/palmprint/training/palmnet_classifiers.py',
    'src/palmprint/features/palmnet.py',
    'src/palmprint/models/palmnet_matching.py',
]
for path in paths:
    source = (root / path).read_text(encoding='utf-8-sig')
    digest = hashlib.sha256(ast.dump(ast.parse(source), include_attributes=False).encode()).hexdigest()
    if digest != original[path]:
        raise RuntimeError(f'Parsed program differs from benchmark source: {path}')
print(f'AST verified: {len(paths)} benchmark modules')
