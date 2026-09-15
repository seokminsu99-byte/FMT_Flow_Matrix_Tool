# External PLENA engine

Canonical source: https://github.com/seokminsu99-byte/FMT_plug_PLENA

Inspected revision: `cc489c5bc3f365af52b931370ceb27bd57e16838`.
The upstream license is MIT, copyright 2026 Minsoo Seok, Changmin Park and Yongwon Seo.
See the upstream LICENSE, CITATION.cff and PROVENANCE.md; retain those notices
when redistributing PLENA itself. FMT's own LICENSE does not replace them.

Build the public engine separately with a C++17 compiler:

```powershell
git clone https://github.com/seokminsu99-byte/FMT_plug_PLENA.git
cd FMT_plug_PLENA
git checkout cc489c5bc3f365af52b931370ceb27bd57e16838
g++ -O2 -std=c++17 -pthread -Wall -Wextra -pedantic src/plena.cpp -o PLENA.exe
./PLENA.exe --self-test
$env:FMT_PLENA_EXECUTABLE = (Resolve-Path ./PLENA.exe).Path
```

Then return to the FMT directory and launch `python fmt.py` in the same shell.
The default resource lookup also accepts a locally supplied `PLENA.exe` beside
`gui.py`; executables are deliberately excluded from Git.

The pre-existing private-folder executable was not assumed to be byte-identical
to this public source. FMT matrix export is tested independently of the engine;
the public engine build and full scientific batch reproduction are separate
checks. Consult the upstream documentation for stochastic-run settings and seeds.
