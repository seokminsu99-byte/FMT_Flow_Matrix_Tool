# Software provenance

## Public implementation

The canonical source is [`src/plena.cpp`](src/plena.cpp).

```text
SHA-256: F03926B9FC6DCB8DEFDB78151251967195D77B35961228C003D9AF48BEF3CBF7
```

The source includes a `--self-test` diagnostic that enumerates valid states of
a small network and checks transition-row sums, preservation of the Gibbs
target distribution, and detailed balance to a numerical tolerance of
`1e-12`.

## Scientific and software lineage

PLENA implements the one-parameter Gibbs drainage-network model described by
Troutman and Karlinger (1992) and follows the urban-drainage application and
width-function workflow developed by Seo and Schmidt (2014) and Seo et al.
(2015).

The core direction-change, loop-checking, and Gibbs-update workflow was ported
with permission from a MATLAB reference implementation provided by Yongwon Seo
and A. R. Schmidt. Minsoo Seok performed the C++ porting and implementation and
led the software experiments, analysis, and verification. Minsoo Seok,
Changmin Park, and Yongwon Seo retain responsibility for the public software and
associated scientific claims.

Relevant publications:

- Troutman, B. M., and Karlinger, M. R. (1992). Gibbs' distribution on
  drainage networks. *Water Resources Research*, 28, 563-577.
  <https://doi.org/10.1029/91WR02648>
- Seo, Y., and Schmidt, A. R. (2014). Application of Gibbs' model to urban
  drainage networks: A case study in southwestern Chicago, USA.
  *Hydrological Processes*, 28, 1148-1158.
  <https://doi.org/10.1002/hyp.9657>
- Seo, Y., Hwang, J., and Noh, S. J. (2015). Analysis of urban drainage
  networks using Gibbs' Model: A case study in Seoul, South Korea. *Water*, 7,
  4129-4143. <https://doi.org/10.3390/w7084129>

## Third-party technical components

- The pseudorandom-number generator follows the `xoshiro256**` reference
  implementation by David Blackman and Sebastiano Vigna:
  <https://prng.di.unimi.it/xoshiro256starstar.c>.
- Seed mixing uses an `fmix64`-style finalizer pattern and constants associated
  with MurmurHash3. A reference implementation is available at
  <https://github.com/rurban/smhasher/blob/master/MurmurHash3.cpp>.

## Input responsibility

The research workflow uses FMT to prepare PLENA inputs. The public PLENA source
also validates matrix dimensions, direction codes, outlet bounds, outlet
activity, and initial outlet reachability. During stochastic generation, it
rejects every proposed state that breaks outlet reachability.
