# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Minsoo Seok
"""Check release hygiene and, optionally, semantic preservation of copied modules."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]
INTENTIONAL_CHANGES={'gui.py','ml_image_audit.py','run_plena_batch.py'}


class RemoveDocstrings(ast.NodeTransformer):
    def generic_visit(self,node):
        if isinstance(node,(ast.Module,ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)):
            if node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):
                node.body=node.body[1:]
        return super().generic_visit(node)


def normalized_ast(path):
    return ast.dump(RemoveDocstrings().visit(ast.parse(path.read_text(encoding='utf-8-sig'))))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,help='Optional original pipenet6 directory, read-only.')
    args=parser.parse_args()
    errors=[]
    checked=0
    for path in ROOT.glob('*.py'):
        text=path.read_text(encoding='utf-8-sig')
        if not text.startswith('# SPDX-License-Identifier: MIT'):
            errors.append(f'Missing source license header: {path.name}')
        if re.search(r'[A-Za-z]:\\Users\\|/Users/|/home/[^/ ]+/',text):
            errors.append(f'Personal path in source: {path.name}')
        ast.parse(text)
        checked+=1
    inventory=json.loads((ROOT/'third_party/dependency_inventory.json').read_text(encoding='utf-8'))
    notices=0
    for package in inventory['packages']:
        if not package['license_files']:
            errors.append(f'Missing license text: {package["name"]}')
        for entry in package['license_files']:
            path=ROOT/entry['path']
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:
                errors.append(f'Changed upstream notice: {entry["path"]}')
            notices+=1
    preserved=[]
    if args.source:
        for original in sorted(args.source.glob('*.py')):
            if original.name in INTENTIONAL_CHANGES:
                continue
            copy=ROOT/original.name
            if not copy.exists() or normalized_ast(copy)!=normalized_ast(original):
                errors.append(f'Unexpected algorithm change: {original.name}')
            else:
                preserved.append(original.name)
    tracked=[]
    if (ROOT/'.git').is_dir():
        result=subprocess.run(['git','ls-files','-z'],cwd=ROOT,check=True,capture_output=True)
        tracked=[x for x in result.stdout.decode().split('\0') if x]
        for package in inventory['packages']:
            for entry in package['license_files']:
                if entry['path'] not in tracked:
                    errors.append(f'License omitted from Git: {entry["path"]}')
        forbidden={'.exe','.dll','.pt','.pth','.npz','.shp','.shx','.dbf','.xlsx','.docx','.pyc'}
        for name in tracked:
            if Path(name).suffix.lower() in forbidden or name.startswith(('output/','gold_labels/','build/','dist/')):
                errors.append(f'Private/generated file tracked: {name}')
    print(json.dumps({'source_modules':checked,'dependency_packages':len(inventory['packages']),
                      'verified_notice_files':notices,'algorithm_modules_preserved':preserved,
                      'tracked_files':len(tracked),'errors':errors},indent=2))
    raise SystemExit(bool(errors))


if __name__=='__main__':
    main()
