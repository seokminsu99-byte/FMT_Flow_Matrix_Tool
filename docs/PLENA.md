# Included PLENA C++ engine

Canonical source: https://github.com/seokminsu99-byte/FMT_plug_PLENA

Inspected revision: `cc489c5bc3f365af52b931370ceb27bd57e16838`.
The upstream license is MIT, copyright 2026 Minsoo Seok, Changmin Park and Yongwon Seo.
See the upstream LICENSE, CITATION.cff and PROVENANCE.md; retain those notices
when redistributing PLENA itself. FMT's own LICENSE does not replace them.

The unchanged source is included in `external/plena/src/plena.cpp`, together
with its original notices. Exact revision and file hashes are recorded in
`external/plena/UPSTREAM.json`. No prebuilt executable or additional build
configuration is included. With an existing C++17 GCC/MinGW compiler, build
from the FMT repository root:

```powershell
g++ -O2 -std=c++17 -pthread -Wall -Wextra -pedantic external/plena/src/plena.cpp -o PLENA.exe
./PLENA.exe --self-test
$env:FMT_PLENA_EXECUTABLE = (Resolve-Path ./PLENA.exe).Path
```

Then launch `python fmt.py` in the same shell.
The default resource lookup also accepts a locally supplied `PLENA.exe` beside
`gui.py`; executables are deliberately excluded from Git.

The pre-existing private-folder executable was not assumed to be byte-identical
to this public source. FMT matrix export is tested independently of the engine;
the public engine build and full scientific batch reproduction are separate
checks. Consult the upstream documentation for stochastic-run settings and seeds.
