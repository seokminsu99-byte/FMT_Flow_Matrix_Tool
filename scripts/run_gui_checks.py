# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Minsoo Seok
"""Run real-Tk synthetic checks without importing or changing user settings."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]


def main():
    checks=[['qa_workspace.py'],['qa_workspace.py','--external-pipe'],
            ['qa_workspace.py','--reverse-branch','--bfs-after-fill'],['qa_gis_editing.py']]
    for args in checks:
        with tempfile.TemporaryDirectory(prefix='fmt-qa-') as tmp:
            env={**os.environ,'FMT_DATA_DIR':tmp,'PYTHONUTF8':'1'}
            subprocess.run([sys.executable,*args],cwd=ROOT,env=env,check=True,timeout=120)
    print('All isolated real-Tk checks passed.')


if __name__=='__main__':
    main()
