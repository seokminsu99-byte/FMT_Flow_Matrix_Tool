/*
PLENA: Program for Large-scale Evaluation of Network Analysis

RESEARCH PROVENANCE AND RESPONSIBILITY

Research authors: Minsoo Seok, Changmin Park, and Yongwon Seo
Lead C++ implementation: Minsoo Seok

- The core changedirect2, loopcheck2, and Gibbs-update workflow was ported
  with permission from a MATLAB reference implementation provided by
  Yongwon Seo and A. R. Schmidt. Yongwon Seo is a co-author of the study.
- Minsoo Seok performed the C++ porting, implementation, experiments,
  analysis, and verification. The authors reviewed the public software and
  retain responsibility for the code, results, and interpretations.
- Author-developed additions include batch scheduling, progress reporting,
  file I/O and selection, input validation, candidate screening,
  outlet-reachability checks, scale-proportional updates, result writing,
  and the parallel execution workflow.

THIRD-PARTY TECHNICAL SOURCES

- PRNG: xoshiro256** reference implementation by David Blackman and
  Sebastiano Vigna, https://prng.di.unimi.it/xoshiro256starstar.c
- Seed mixing: fmix64-style finalizer pattern and constants associated with
  MurmurHash3, https://github.com/rurban/smhasher/blob/master/MurmurHash3.cpp

IMPLEMENTATION OVERVIEW

State space
- D is a four-direction drainage tree on the occupied mask AD = (D != 0).
- Every occupied cell must reach the fixed outlet.
- The transition energy is sum(T). The shortest-path term in the manuscript
  Hamiltonian is constant for a fixed mask and cancels in Delta H.

One random-scan Metropolis update
1. Select one non-outlet occupied cell uniformly.
2. Select one of its three directions other than the current direction uniformly.
3. Screen boundary, inactive-cell, and immediate two-cycle proposals.
4. Reject a proposal as a self-transition if it fails outlet reachability.
5. Otherwise accept with min(1, exp(-beta * Delta H)).

With N_u update-eligible cells, each directed proposal slot has probability
1/(3*N_u). This proposal factor is supplied by steps 1 and 2 and is not applied
again in the conditional acceptance test.
*/

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace std;
namespace fs = filesystem;

using Clock = chrono::steady_clock;

struct Cell {
    int i = 0;
    int j = 0;
};

struct Matrix {
    int n = 0;
    int m = 0;
    vector<int> data;

    Matrix() = default;
    Matrix(int rows, int cols, int value = 0)
        : n(rows), m(cols), data(static_cast<size_t>(rows) * static_cast<size_t>(cols), value) {}

    int& operator()(int i, int j) {
        return data[static_cast<size_t>(i) * static_cast<size_t>(m) + static_cast<size_t>(j)];
    }

    const int& operator()(int i, int j) const {
        return data[static_cast<size_t>(i) * static_cast<size_t>(m) + static_cast<size_t>(j)];
    }

    void assign(int value) {
        fill(data.begin(), data.end(), value);
    }
};

struct QResult {
    Matrix FA;
    Matrix T;
    vector<double> q;
};

struct GibbsStats {
    uint64_t iterations = 0;
    uint64_t accepted = 0;
    uint64_t rejectedEnergy = 0;
    uint64_t screenedLocal = 0;
    uint64_t screenedReachability = 0;
};

struct BatchTask {
    int k = 0;
    int run = 0;
    double beta = 0.0;
    size_t outIndex = 0;
};

struct BatchResult {
    bool ok = true;
    string err;
    vector<double> q;
};

static constexpr double DEFAULT_ALPHA = 10.0;

static inline Clock::time_point tic() {
    return Clock::now();
}

static inline double toc_sec(const Clock::time_point& t0) {
    return chrono::duration<double>(Clock::now() - t0).count();
}

static bool ends_with(const string& s, const string& suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static bool in_bounds(const Matrix& M, int i, int j) {
    return i >= 0 && i < M.n && j >= 0 && j < M.m;
}

static Cell step_from_direction(Cell c, int dir) {
    if (dir == 1) return { c.i, c.j + 1 };
    if (dir == 2) return { c.i + 1, c.j };
    if (dir == 3) return { c.i, c.j - 1 };
    if (dir == 4) return { c.i - 1, c.j };
    return { c.i, c.j };
}

static vector<fs::path> listTxtFilesInCwd(bool excludeResultFiles = true) {
    vector<fs::path> files;
    error_code ec;
    for (const auto& entry : fs::directory_iterator(fs::current_path(), ec)) {
        if (ec) break;
        if (!entry.is_regular_file()) continue;
        fs::path p = entry.path();
        if (p.extension() != ".txt") continue;
        string name = p.filename().string();
        if (excludeResultFiles && (ends_with(name, "_result.txt") || ends_with(name, "_NSE.txt"))) continue;
        files.push_back(p);
    }
    sort(files.begin(), files.end(), [](const fs::path& a, const fs::path& b) {
        error_code ec1, ec2;
        auto ta = fs::last_write_time(a, ec1);
        auto tb = fs::last_write_time(b, ec2);
        if (ec1 || ec2) return a.filename().string() < b.filename().string();
        return ta > tb;
    });
    return files;
}

static string chooseTxtFileInteractive() {
    auto files = listTxtFilesInCwd(true);
    cout << "CWD = " << fs::current_path().string() << "\n";
    if (files.empty()) {
        cout << "No selectable text-format files were found in the current folder.\n";
        return "";
    }

    cout << "\n//------ Input file list //------\n";
    for (size_t i = 0; i < files.size(); ++i) {
        cout << "[" << (i + 1) << "] " << files[i].filename().string() << "\n";
    }
    cout << "[0] Exit\n";

    while (true) {
        cout << "\nSelect a file number: ";
        int sel = 0;
        if (!(cin >> sel)) {
            cin.clear();
            cin.ignore(numeric_limits<streamsize>::max(), '\n');
            continue;
        }
        if (sel == 0) return "";
        if (sel >= 1 && static_cast<size_t>(sel) <= files.size()) {
            return files[static_cast<size_t>(sel - 1)].string();
        }
    }
}

struct Xoshiro256ss {
    uint64_t s[4]{};

    static uint64_t rotl(uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }

    static uint64_t splitmix64(uint64_t& x) {
        uint64_t z = (x += 0x9E3779B97f4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }

    explicit Xoshiro256ss(uint64_t seed) {
        uint64_t x = seed;
        for (uint64_t& v : s) v = splitmix64(x);
    }

    uint64_t next() {
        uint64_t result = rotl(s[1] * 5ULL, 7) * 9ULL;
        uint64_t t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = rotl(s[3], 45);
        return result;
    }

    double uniform01() {
        return static_cast<double>(next() >> 11) * (1.0 / static_cast<double>(1ULL << 53));
    }

    int uniformInt(int a, int b) {
        if (b < a) throw invalid_argument("invalid uniformInt range");
        const uint64_t range = static_cast<uint64_t>(b - a) + 1ULL;
        const uint64_t threshold = (0ULL - range) % range;
        uint64_t value = 0;
        do {
            value = next();
        } while (value < threshold);
        return a + static_cast<int>(value % range);
    }
};

static uint64_t mix_seed(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

static uint64_t makeAutomaticSeed() {
    uint64_t seed = static_cast<uint64_t>(
        chrono::high_resolution_clock::now().time_since_epoch().count()
    );

    // random_device uses the platform entropy source when one is available.
    try {
        random_device rd;
        for (int i = 0; i < 4; ++i) {
            const uint64_t entropy = (static_cast<uint64_t>(rd()) << 32) ^ rd();
            seed ^= mix_seed(entropy + 0x9E3779B97F4A7C15ULL * static_cast<uint64_t>(i + 1));
        }
    }
    catch (...) {
        // The high-resolution timestamp remains as the fallback source.
    }

    seed = mix_seed(seed ^ 0xD1B54A32D192ED03ULL);
    return (seed == 0) ? 0xA0761D6478BD642FULL : seed;
}

static Matrix occupiedMaskFromD(const Matrix& D) {
    Matrix AD(D.n, D.m, 0);
    for (int i = 0; i < D.n; ++i) {
        for (int j = 0; j < D.m; ++j) {
            AD(i, j) = (D(i, j) != 0) ? 1 : 0;
        }
    }
    return AD;
}

static vector<Cell> occupiedCells(const Matrix& AD, Cell outlet, bool includeOutlet) {
    vector<Cell> cells;
    for (int i = 0; i < AD.n; ++i) {
        for (int j = 0; j < AD.m; ++j) {
            if (AD(i, j) == 0) continue;
            if (!includeOutlet && i == outlet.i && j == outlet.j) continue;
            cells.push_back({ i, j });
        }
    }
    return cells;
}

static void computeTravelTime(const Matrix& D, const Matrix& AD, Matrix& T, Cell outlet) {
    T.assign(0);
    queue<Cell> q;
    T(outlet.i, outlet.j) = 1;
    q.push(outlet);

    while (!q.empty()) {
        Cell c = q.front();
        q.pop();
        int curT = T(c.i, c.j);

        array<pair<Cell, int>, 4> incoming = {
            pair<Cell, int>{{c.i, c.j - 1}, 1},
            pair<Cell, int>{{c.i - 1, c.j}, 2},
            pair<Cell, int>{{c.i, c.j + 1}, 3},
            pair<Cell, int>{{c.i + 1, c.j}, 4},
        };

        for (const auto& item : incoming) {
            Cell nb = item.first;
            int requiredDir = item.second;
            if (!in_bounds(D, nb.i, nb.j)) continue;
            if (AD(nb.i, nb.j) == 0 || T(nb.i, nb.j) != 0) continue;
            if (D(nb.i, nb.j) != requiredDir) continue;
            T(nb.i, nb.j) = curT + 1;
            q.push(nb);
        }
    }
}

static bool allReachOutlet(const Matrix& D, const Matrix& AD, Cell outlet) {
    Matrix T(D.n, D.m, 0);
    computeTravelTime(D, AD, T, outlet);
    for (int i = 0; i < D.n; ++i) {
        for (int j = 0; j < D.m; ++j) {
            if (AD(i, j) != 0 && T(i, j) == 0) return false;
        }
    }
    return true;
}

static double energyH(const Matrix& D, const Matrix& AD, Cell outlet) {
    Matrix T(D.n, D.m, 0);
    computeTravelTime(D, AD, T, outlet);
    long long sumT = 0;
    for (int i = 0; i < D.n; ++i) {
        for (int j = 0; j < D.m; ++j) {
            if (AD(i, j) == 0) continue;
            if (T(i, j) == 0) return numeric_limits<double>::infinity();
            sumT += T(i, j);
        }
    }
    return static_cast<double>(sumT);
}

static void validateInputMatrix(const Matrix& D, const Matrix& AD, Cell outlet) {
    if (!in_bounds(D, outlet.i, outlet.j)) {
        throw runtime_error("Input range error: outlet is outside the matrix");
    }
    if (AD(outlet.i, outlet.j) == 0) {
        throw runtime_error("Input error: outlet cell must be non-zero");
    }

    for (int i = 0; i < D.n; ++i) {
        for (int j = 0; j < D.m; ++j) {
            int v = D(i, j);
            if (v < 0 || v > 4) {
                throw runtime_error("Input error: direction values must be in 0..4");
            }
            if (AD(i, j) == 0) continue;
            if (i == outlet.i && j == outlet.j) continue;
            if (v < 1 || v > 4) {
                throw runtime_error("Input error: every non-outlet occupied cell must have direction 1..4");
            }
            Cell nb = step_from_direction({ i, j }, v);
            if (!in_bounds(D, nb.i, nb.j) || AD(nb.i, nb.j) == 0) {
                throw runtime_error("Input error: an occupied cell points outside the occupied pipe mask");
            }
        }
    }

    if (!allReachOutlet(D, AD, outlet)) {
        throw runtime_error("Input error: initial matrix is not outlet-reachable; correct the input before PLENA");
    }
}

static int oppositeDirection(int dir) {
    if (dir < 1 || dir > 4) throw invalid_argument("direction must be in 1..4");
    return ((dir + 1) % 4) + 1;
}

static uint64_t iterationBudget(double alpha, int n, int m) {
    if (!(alpha > 0.0) || !isfinite(alpha)) {
        throw invalid_argument("alpha must be a finite positive number");
    }
    const long double raw = static_cast<long double>(alpha)
        * static_cast<long double>(n)
        * static_cast<long double>(m);
    if (raw > static_cast<long double>(numeric_limits<uint64_t>::max())) {
        throw overflow_error("alpha*n*m exceeds the supported iteration range");
    }
    return max<uint64_t>(1ULL, static_cast<uint64_t>(ceil(raw)));
}

static int proposeAlternativeDirection(int currentDirection, Xoshiro256ss& gen) {
    if (currentDirection < 1 || currentDirection > 4) {
        throw invalid_argument("current direction must be in 1..4");
    }
    const int offset = gen.uniformInt(1, 3);
    return ((currentDirection - 1 + offset) % 4) + 1;
}

static bool failsLocalScreen(
    const Matrix& D,
    const Matrix& AD,
    Cell outlet,
    Cell c,
    int proposedDirection
) {
    const Cell neighbor = step_from_direction(c, proposedDirection);
    if (!in_bounds(D, neighbor.i, neighbor.j)) return true;
    if (AD(neighbor.i, neighbor.j) == 0) return true;

    // The outlet direction is a storage marker, not an outgoing drainage edge.
    if (neighbor.i == outlet.i && neighbor.j == outlet.j) return false;
    return D(neighbor.i, neighbor.j) == oppositeDirection(proposedDirection);
}

static Matrix gibbsMetropolisSampler(
    const Matrix& D0,
    const Matrix& AD,
    Cell outlet,
    double beta,
    double alpha,
    uint64_t seed,
    GibbsStats* stats = nullptr
) {
    Matrix D = D0;
    vector<Cell> cells = occupiedCells(AD, outlet, false);
    if (cells.empty()) return D;

    Xoshiro256ss gen(seed);
    const uint64_t iterations = iterationBudget(alpha, D.n, D.m);
    double currentEnergy = energyH(D, AD, outlet);
    if (!isfinite(currentEnergy)) {
        throw runtime_error("initial network is outside the outlet-reachable state space");
    }

    GibbsStats local;
    local.iterations = iterations;

    for (uint64_t iter = 0; iter < iterations; ++iter) {
        Cell c = cells[static_cast<size_t>(gen.uniformInt(0, static_cast<int>(cells.size()) - 1))];
        const int oldDirection = D(c.i, c.j);
        const int proposedDirection = proposeAlternativeDirection(oldDirection, gen);

        if (failsLocalScreen(D, AD, outlet, c, proposedDirection)) {
            local.screenedLocal++;
            continue;
        }

        Matrix candidate = D;
        candidate(c.i, c.j) = proposedDirection;
        const double candidateEnergy = energyH(candidate, AD, outlet);
        if (!isfinite(candidateEnergy)) {
            local.screenedReachability++;
            continue;
        }

        const double deltaH = candidateEnergy - currentEnergy;
        const double acceptance = (deltaH <= 0.0) ? 1.0 : exp(-beta * deltaH);
        if (gen.uniform01() < acceptance) {
            D = move(candidate);
            currentEnergy = candidateEnergy;
            local.accepted++;
        }
        else {
            local.rejectedEnergy++;
        }
    }

    if (!allReachOutlet(D, AD, outlet)) {
        throw runtime_error("internal error: Gibbs sampler left outlet-reachable state space");
    }

    if (stats != nullptr) *stats = local;
    return D;
}

static void computeFlowAccum(const Matrix& D, const Matrix& AD, Matrix& FA) {
    for (int i = 0; i < D.n; ++i) {
        for (int j = 0; j < D.m; ++j) {
            if (AD(i, j) == 0) {
                FA(i, j) = 0;
                continue;
            }
            int val = 0;
            if (j > 0 && AD(i, j - 1) != 0 && D(i, j - 1) == 1) val += 1;
            if (i > 0 && AD(i - 1, j) != 0 && D(i - 1, j) == 2) val += 1;
            if (j < D.m - 1 && AD(i, j + 1) != 0 && D(i, j + 1) == 3) val += 1;
            if (i < D.n - 1 && AD(i + 1, j) != 0 && D(i + 1, j) == 4) val += 1;
            FA(i, j) = val;
        }
    }
}

static void computeDischarge(const Matrix& AD, const Matrix& T, const Matrix& FA, Cell outlet, vector<double>& q) {
    int maxT = 0;
    for (int i = 0; i < T.n; ++i) {
        for (int j = 0; j < T.m; ++j) {
            if (AD(i, j) != 0 && T(i, j) > maxT) maxT = T(i, j);
        }
    }

    q.clear();
    if (maxT <= 0) return;

    vector<double> qk(static_cast<size_t>(maxT) + 1, 0.0);
    for (int i = 0; i < T.n; ++i) {
        for (int j = 0; j < T.m; ++j) {
            int tt = T(i, j);
            if (AD(i, j) != 0 && tt >= 1 && tt <= maxT) {
                qk[static_cast<size_t>(tt)] += static_cast<double>(FA(i, j));
            }
        }
    }

    q.reserve(static_cast<size_t>(maxT) + 2);
    q.push_back(0.0);
    q.push_back(static_cast<double>(AD(outlet.i, outlet.j) != 0 ? 1 : 0));
    for (int k = 1; k <= maxT - 1; ++k) q.push_back(qk[static_cast<size_t>(k)]);
    q.push_back(0.0);
}

static QResult calculateQ(const Matrix& D, const Matrix& AD, Cell outlet) {
    Matrix FA(D.n, D.m, 0);
    computeFlowAccum(D, AD, FA);
    Matrix T(D.n, D.m, 0);
    computeTravelTime(D, AD, T, outlet);
    vector<double> q;
    computeDischarge(AD, T, FA, outlet, q);
    return { FA, T, q };
}

static string makeResultFilename(const string& inputName) {
    size_t pos = inputName.rfind('.');
    if (pos == string::npos) return inputName + "_result.txt";
    return inputName.substr(0, pos) + "_result" + inputName.substr(pos);
}

static void writeResultFileFromRes(
    const string& inputName,
    const Matrix& D,
    const QResult& res,
    const GibbsStats& stats,
    double beta,
    double alpha,
    size_t eligibleCells,
    uint64_t baseSeed,
    uint64_t runSeed
) {
    string outName = makeResultFilename(inputName);
    ofstream ofs(outName);
    if (!ofs) {
        cerr << "Cannot open result file: " << outName << "\n";
        return;
    }

    ofs << "//------ PLENA metadata //------\n";
    ofs << "sampler random-scan Metropolis with pre-screened self-transitions\n";
    ofs << "target_probability proportional_to exp(-beta * H(D))\n";
    ofs << "energy H(D)=sum(T over occupied cells)\n";
    ofs << "beta " << setprecision(17) << beta << "\n";
    ofs << "alpha " << alpha << "\n";
    ofs << "base_seed " << baseSeed << "\n";
    ofs << "run_seed " << runSeed << "\n";
    ofs << "iterations " << stats.iterations << "\n";
    ofs << "update_eligible_cells " << eligibleCells << "\n";
    ofs << "proposal_degree_bound_r_star " << (3ULL * static_cast<uint64_t>(eligibleCells)) << "\n";
    ofs << "accepted " << stats.accepted << "\n";
    ofs << "rejected_energy " << stats.rejectedEnergy << "\n";
    ofs << "screened_local " << stats.screenedLocal << "\n";
    ofs << "screened_reachability " << stats.screenedReachability << "\n\n";

    ofs << "//------ Final Direction Matrix D //------\n";
    for (int i = 0; i < D.n; ++i) {
        for (int j = 0; j < D.m; ++j) ofs << D(i, j) << ' ';
        ofs << '\n';
    }

    ofs << "\n//------ Travel Time Matrix T //------\n";
    for (int i = 0; i < res.T.n; ++i) {
        for (int j = 0; j < res.T.m; ++j) ofs << res.T(i, j) << ' ';
        ofs << '\n';
    }

    ofs << "\n//------ Flow Accumulation Matrix FA //------\n";
    for (int i = 0; i < res.FA.n; ++i) {
        for (int j = 0; j < res.FA.m; ++j) ofs << res.FA(i, j) << ' ';
        ofs << '\n';
    }

    ofs << "\n//------ q(t) (index value) //------\n";
    for (size_t k = 0; k < res.q.size(); ++k) {
        ofs << k << ' ' << res.q[k] << '\n';
    }
}

static int findLastNonzeroIndex(const vector<double>& q) {
    for (int idx = static_cast<int>(q.size()) - 1; idx >= 0; --idx) {
        if (q[static_cast<size_t>(idx)] != 0.0) return idx;
    }
    return 0;
}

static void computeWidthDistribution(const Matrix& D, const Matrix& AD, Matrix& FD, Matrix& LS) {
    FD = Matrix(D.n, D.m, 0);
    LS = Matrix(D.n, D.m, 0);

    for (int i = 0; i < D.n; ++i) {
        for (int j = 0; j < D.m; ++j) {
            if (AD(i, j) == 0) continue;
            QResult res = calculateQ(D, AD, { i, j });
            double maxq = 0.0;
            for (double v : res.q) maxq = max(maxq, v);
            FD(i, j) = static_cast<int>(round(maxq));
            LS(i, j) = findLastNonzeroIndex(res.q);
        }
    }
}

static vector<double> computeQVector(const Matrix& D, const Matrix& AD, Cell outlet) {
    return calculateQ(D, AD, outlet).q;
}

static void writeCsvMatrix(const Matrix& M, const string& filename) {
    ofstream ofs(filename);
    if (!ofs) {
        cerr << "Cannot open CSV file: " << filename << "\n";
        return;
    }
    for (int i = 0; i < M.n; ++i) {
        for (int j = 0; j < M.m; ++j) {
            ofs << M(i, j);
            if (j < M.m - 1) ofs << ',';
        }
        ofs << '\n';
    }
}

static void writeWidthFunctionsCsv(const vector<string>& headers, const vector<vector<double>>& all_q, const string& filename) {
    size_t maxLen = 0;
    for (const auto& v : all_q) maxLen = max(maxLen, v.size());

    ofstream ofs(filename);
    if (!ofs) {
        cerr << "Cannot open CSV file: " << filename << "\n";
        return;
    }

    ofs << "distance";
    for (const auto& h : headers) ofs << ',' << h;
    ofs << '\n';

    for (size_t i = 0; i < maxLen; ++i) {
        ofs << i;
        for (const auto& col : all_q) {
            ofs << ',' << ((i < col.size()) ? col[i] : 0.0);
        }
        ofs << '\n';
    }
}

static double computeNSE(const vector<double>& obs, const vector<double>& sim) {
    const size_t L = max(obs.size(), sim.size());
    if (L == 0) return numeric_limits<double>::quiet_NaN();

    double meanObs = 0.0;
    for (size_t i = 0; i < L; ++i) meanObs += (i < obs.size()) ? obs[i] : 0.0;
    meanObs /= static_cast<double>(L);

    double denom = 0.0;
    double numer = 0.0;
    for (size_t i = 0; i < L; ++i) {
        double o = (i < obs.size()) ? obs[i] : 0.0;
        double s = (i < sim.size()) ? sim[i] : 0.0;
        denom += (o - meanObs) * (o - meanObs);
        numer += (o - s) * (o - s);
    }
    if (denom <= 0.0) return (numer <= 0.0) ? 1.0 : numeric_limits<double>::quiet_NaN();
    return 1.0 - numer / denom;
}

static string makeNseFilename(const string& inputName) {
    size_t pos = inputName.rfind('.');
    if (pos == string::npos) return inputName + "_NSE.txt";
    return inputName.substr(0, pos) + "_NSE" + inputName.substr(pos);
}

static int resolveThreadCount(int max_threads_user) {
    unsigned hc = thread::hardware_concurrency();
    int t = (hc == 0) ? 4 : static_cast<int>(hc);
    if (max_threads_user > 0) t = min(t, max_threads_user);
    return max(1, t);
}

static void reportNSE_andSave(
    const string& inputName,
    const vector<double>& q_original,
    const vector<BatchTask>& tasks,
    const vector<BatchResult>& results,
    const vector<vector<double>>& all_q,
    double alpha,
    uint64_t iterations,
    uint64_t baseSeed
) {
    struct BetaAgg {
        int okRuns = 0;
        int totalRuns = 0;
        double sumNSE = 0.0;
        int finiteNseRuns = 0;
        vector<double> sumQ;
    };

    map<int, BetaAgg> agg;
    for (size_t ti = 0; ti < tasks.size(); ++ti) {
        const BatchTask& task = tasks[ti];
        BetaAgg& A = agg[task.k];
        A.totalRuns++;
        if (!results[ti].ok || all_q[task.outIndex].empty()) continue;

        const vector<double>& q_sim = all_q[task.outIndex];
        double nse = computeNSE(q_original, q_sim);
        A.okRuns++;
        if (!isnan(nse)) {
            A.sumNSE += nse;
            A.finiteNseRuns++;
        }
        if (A.sumQ.size() < q_sim.size()) A.sumQ.resize(q_sim.size(), 0.0);
        for (size_t i = 0; i < q_sim.size(); ++i) A.sumQ[i] += q_sim[i];
    }

    string nseFile = makeNseFilename(inputName);
    ofstream nfs(nseFile);

    cout << "\n[Summary by beta]\n";
    cout << "k\t\tbeta\t\tok/total\t\tmeanNSE(run)\t\tNSE(mean_q)\n";
    if (nfs) {
        nfs << "sampler random-scan Metropolis with pre-screened self-transitions\n";
        nfs << "alpha " << setprecision(17) << alpha << "\n";
        nfs << "iterations_per_run " << iterations << "\n";
        nfs << "base_seed " << baseSeed << "\n\n";
        nfs << "[Summary by beta]\n";
        nfs << "k\t\tbeta\t\tok/total\t\tmeanNSE(run)\t\tNSE(mean_q)\n";
    }

    cout << fixed << setprecision(6);
    for (const auto& kv : agg) {
        int k = kv.first;
        const BetaAgg& A = kv.second;
        double betaVal = pow(10.0, k);

        double meanNSE_run = numeric_limits<double>::quiet_NaN();
        if (A.finiteNseRuns > 0) meanNSE_run = A.sumNSE / static_cast<double>(A.finiteNseRuns);

        vector<double> meanQ = A.sumQ;
        if (A.okRuns > 0) {
            for (double& v : meanQ) v /= static_cast<double>(A.okRuns);
        } else {
            meanQ.clear();
        }
        double nse_meanQ = computeNSE(q_original, meanQ);

        cout << k << "\t"
             << scientific << setprecision(3) << betaVal
             << fixed << setprecision(6) << "\t"
             << A.okRuns << "/" << A.totalRuns << "\t\t";
        if (isnan(meanNSE_run)) cout << "NaN\t\t";
        else cout << meanNSE_run << "\t\t";
        if (isnan(nse_meanQ)) cout << "NaN\n";
        else cout << nse_meanQ << "\n";

        if (nfs) {
            nfs << k << "\t"
                << scientific << setprecision(10) << betaVal
                << fixed << setprecision(10) << "\t"
                << A.okRuns << "/" << A.totalRuns << "\t";
            if (isnan(meanNSE_run)) nfs << "NaN\t";
            else nfs << meanNSE_run << "\t";
            if (isnan(nse_meanQ)) nfs << "NaN\n";
            else nfs << nse_meanQ << "\n";
        }
    }

    cout << "\nNSE saved: " << nseFile << "\n";
}

static Matrix readInputMatrix(const string& inputName, Cell& outlet) {
    ifstream ifs(inputName);
    if (!ifs) {
        throw runtime_error("Cannot open input file: " + inputName);
    }

    int n = 0;
    int m = 0;
    if (!(ifs >> n >> m)) throw runtime_error("File format error: failed to read n and m");

    int n0_1 = 0;
    int m0_1 = 0;
    if (!(ifs >> n0_1 >> m0_1)) throw runtime_error("File format error: failed to read outlet (n0 m0)");

    outlet = { n0_1 - 1, m0_1 - 1 };
    if (n <= 0 || m <= 0) throw runtime_error("Input range error: n and m must be positive");
    if (static_cast<unsigned long long>(n) * static_cast<unsigned long long>(m) > 50'000'000ULL) {
        throw runtime_error("Input range error: matrix is too large");
    }

    Matrix D(n, m, 0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            int v = 0;
            if (!(ifs >> v)) throw runtime_error("File format error: insufficient data for D matrix");
            D(i, j) = v;
        }
    }
    return D;
}

static bool verifyTinyStateKernel(double beta, double tolerance) {
    const Cell outlet{ 1, 1 };
    const Matrix AD(2, 2, 1);
    const vector<Cell> cells = occupiedCells(AD, outlet, false);

    vector<Matrix> states;
    for (int code = 0; code < 64; ++code) {
        Matrix D(2, 2, 1);
        int value = code;
        bool locallyValid = true;
        for (const Cell c : cells) {
            const int dir = (value % 4) + 1;
            value /= 4;
            D(c.i, c.j) = dir;
            const Cell neighbor = step_from_direction(c, dir);
            if (!in_bounds(D, neighbor.i, neighbor.j)) {
                locallyValid = false;
                break;
            }
        }
        if (locallyValid && allReachOutlet(D, AD, outlet)) states.push_back(move(D));
    }

    map<vector<int>, size_t> stateIndex;
    for (size_t i = 0; i < states.size(); ++i) stateIndex[states[i].data] = i;

    const size_t count = states.size();
    vector<vector<double>> transition(count, vector<double>(count, 0.0));
    vector<double> energy(count, 0.0);
    for (size_t i = 0; i < count; ++i) energy[i] = energyH(states[i], AD, outlet);

    const double slotProbability = 1.0 / (3.0 * static_cast<double>(cells.size()));
    double maxRowResidual = 0.0;
    for (size_t i = 0; i < count; ++i) {
        const Matrix& current = states[i];
        for (const Cell c : cells) {
            const int oldDirection = current(c.i, c.j);
            for (int offset = 1; offset <= 3; ++offset) {
                const int proposedDirection = ((oldDirection - 1 + offset) % 4) + 1;
                if (failsLocalScreen(current, AD, outlet, c, proposedDirection)) {
                    transition[i][i] += slotProbability;
                    continue;
                }

                Matrix candidate = current;
                candidate(c.i, c.j) = proposedDirection;
                const double candidateEnergy = energyH(candidate, AD, outlet);
                if (!isfinite(candidateEnergy)) {
                    transition[i][i] += slotProbability;
                    continue;
                }

                const auto found = stateIndex.find(candidate.data);
                if (found == stateIndex.end()) {
                    throw runtime_error("self-test could not locate a feasible candidate state");
                }
                const double deltaH = candidateEnergy - energy[i];
                const double acceptance = (deltaH <= 0.0) ? 1.0 : exp(-beta * deltaH);
                transition[i][found->second] += slotProbability * acceptance;
                transition[i][i] += slotProbability * (1.0 - acceptance);
            }
        }

        double rowSum = 0.0;
        for (double probability : transition[i]) rowSum += probability;
        maxRowResidual = max(maxRowResidual, abs(rowSum - 1.0));
    }

    const double minEnergy = *min_element(energy.begin(), energy.end());
    vector<double> target(count, 0.0);
    double targetSum = 0.0;
    for (size_t i = 0; i < count; ++i) {
        target[i] = exp(-beta * (energy[i] - minEnergy));
        targetSum += target[i];
    }
    for (double& probability : target) probability /= targetSum;

    double maxStationaryResidual = 0.0;
    double maxDetailedBalanceResidual = 0.0;
    for (size_t j = 0; j < count; ++j) {
        double propagated = 0.0;
        for (size_t i = 0; i < count; ++i) {
            propagated += target[i] * transition[i][j];
            maxDetailedBalanceResidual = max(
                maxDetailedBalanceResidual,
                abs(target[i] * transition[i][j] - target[j] * transition[j][i])
            );
        }
        maxStationaryResidual = max(maxStationaryResidual, abs(propagated - target[j]));
    }

    cout << scientific << setprecision(3)
         << "beta=" << beta
         << ", states=" << count
         << ", row_residual=" << maxRowResidual
         << ", stationary_residual=" << maxStationaryResidual
         << ", detailed_balance_residual=" << maxDetailedBalanceResidual << "\n";

    return maxRowResidual <= tolerance
        && maxStationaryResidual <= tolerance
        && maxDetailedBalanceResidual <= tolerance;
}

static int runSelfTest() {
    cout << "PLENA transition-kernel self-test (2x2 full mask)\n";
    bool ok = true;
    for (double beta : { 0.0, 0.1, 1.0, 10.0 }) {
        ok = verifyTinyStateKernel(beta, 1e-12) && ok;
    }
    cout << (ok ? "SELF-TEST PASSED\n" : "SELF-TEST FAILED\n");
    return ok ? 0 : 2;
}

int main(int argc, char* argv[]) {
    try {
        if (argc == 2 && string(argv[1]) == "--self-test") return runSelfTest();
        if (argc > 3) {
            throw runtime_error("Usage: PLENA.exe [input.txt] [optional_seed]");
        }

        string inputName;
        if (argc > 1) inputName = argv[1];
        else {
            inputName = chooseTxtFileInteractive();
            if (inputName.empty()) return 0;
        }

        Cell outlet;
        Matrix D1 = readInputMatrix(inputName, outlet);
        Matrix AD = occupiedMaskFromD(D1);
        validateInputMatrix(D1, AD, outlet);

        double a = 0.0;
        cout << "\nBeta is defined as beta = 10^a.\n";
        cout << "Enter a (example: 6 -> 1e6): ";
        if (!(cin >> a)) throw runtime_error("Failed to read a");
        if (a > 308.0 || a < -308.0) throw runtime_error("a is out of range: -308 ~ 308");
        double beta = pow(10.0, a);
        cout << "Configured beta = " << beta << " (10^" << a << ")\n";

        double alpha = DEFAULT_ALPHA;
        cout << "Iteration coefficient alpha in I = alpha*n*m (0 = default 10): ";
        if (!(cin >> alpha)) throw runtime_error("Failed to read alpha");
        if (alpha == 0.0) alpha = DEFAULT_ALPHA;
        const uint64_t iterations = iterationBudget(alpha, D1.n, D1.m);
        const size_t eligibleCellCount = occupiedCells(AD, outlet, false).size();
        cout << "Configured alpha = " << alpha << "\n";
        cout << "Sampler = random-scan Metropolis with candidate screening\n";
        cout << "Iterations = " << iterations << " (alpha*n*m), proposal slots r = "
             << (3ULL * static_cast<uint64_t>(eligibleCellCount)) << "\n";

        uint64_t baseSeed = makeAutomaticSeed();
        bool userSeed = false;
        if (argc == 3) {
            size_t parsed = 0;
            const string seedText = argv[2];
            baseSeed = stoull(seedText, &parsed, 0);
            if (parsed != seedText.size()) throw runtime_error("optional_seed must be an unsigned integer");
            userSeed = true;
        }
        cout << (userSeed ? "User-specified seed = " : "Automatic seed = ") << baseSeed << "\n";

        uint64_t singleSeed = mix_seed(baseSeed ^ 0xC0DEC0DEULL);
        auto t0 = tic();
        GibbsStats stats;
        Matrix Dgibbs = gibbsMetropolisSampler(
            D1,
            AD,
            outlet,
            beta,
            alpha,
            singleSeed,
            &stats
        );
        QResult finalRes = calculateQ(Dgibbs, AD, outlet);
        double elapsed = toc_sec(t0);

        writeResultFileFromRes(
            inputName,
            Dgibbs,
            finalRes,
            stats,
            beta,
            alpha,
            eligibleCellCount,
            baseSeed,
            singleSeed
        );

        cout << "\n[Sampler time] " << elapsed << " sec"
             << "  (single realization, excluding input parsing and file save time)\n";
        cout << "Gibbs updates: accepted=" << stats.accepted
             << ", rejected_energy=" << stats.rejectedEnergy
             << ", screened_local=" << stats.screenedLocal
             << ", screened_reachability=" << stats.screenedReachability << "\n";
        cout << "Done: " << makeResultFilename(inputName) << "\n";

        char doWidth = 'n';
        cout << "\nDo you want to compute the width function? (y/n): ";
        cin >> doWidth;

        if (doWidth == 'y' || doWidth == 'Y') {
            Matrix FD, LS;
            cout << "Computing width-function distributions (FD, LS)...\n";
            computeWidthDistribution(D1, AD, FD, LS);

            string baseName = fs::path(inputName).filename().string();
            string fdName = baseName + "_FD.csv";
            string lsName = baseName + "_LS.csv";
            writeCsvMatrix(FD, fdName);
            writeCsvMatrix(LS, lsName);
            cout << "Saved: " << fdName << ", " << lsName << "\n";

            vector<vector<double>> all_q;
            vector<string> headers;
            all_q.push_back(computeQVector(D1, AD, outlet));
            headers.push_back("original");

            char doBatch = 'n';
            cout << "Do you want to save width functions for multiple beta values (10^k)? (y/n): ";
            cin >> doBatch;

            if (doBatch == 'y' || doBatch == 'Y') {
                vector<int> k_vals = { -4, -3, -2, -1, 0, 1, 2, 3 };
                int runsPerBeta = 100;

                char customRange = 'n';
                cout << "Default: k=-4~3, run=100. Change settings? (y/n): ";
                cin >> customRange;
                if (customRange == 'y' || customRange == 'Y') {
                    int kStart = 0;
                    int kEnd = 0;
                    cout << "Start k (integer): ";
                    cin >> kStart;
                    cout << "End k (integer): ";
                    cin >> kEnd;
                    if (kStart > kEnd) swap(kStart, kEnd);
                    k_vals.clear();
                    for (int k = kStart; k <= kEnd; ++k) k_vals.push_back(k);
                    cout << "Number of runs: ";
                    cin >> runsPerBeta;
                    if (runsPerBeta <= 0) runsPerBeta = 1;
                }

                int max_threads_user = 0;
                cout << "Thread cap (max_threads). Enter 0 for automatic selection (hardware_concurrency): ";
                cin >> max_threads_user;
                int nThreads = resolveThreadCount(max_threads_user);
                cout << "Running parallel batch: threads = " << nThreads
                     << "  (hc=" << thread::hardware_concurrency() << ")\n";

                vector<BatchTask> tasks;
                tasks.reserve(static_cast<size_t>(k_vals.size()) * static_cast<size_t>(runsPerBeta));
                size_t baseIndex = all_q.size();
                size_t totalTasks = static_cast<size_t>(k_vals.size()) * static_cast<size_t>(runsPerBeta);
                all_q.resize(baseIndex + totalTasks);
                headers.resize(baseIndex + totalTasks);

                size_t outIndex = baseIndex;
                for (int k : k_vals) {
                    double bval = pow(10.0, k);
                    for (int run = 1; run <= runsPerBeta; ++run) {
                        BatchTask task;
                        task.k = k;
                        task.run = run;
                        task.beta = bval;
                        task.outIndex = outIndex++;
                        tasks.push_back(task);
                        headers[task.outIndex] = "10^" + to_string(k) + "_run" + to_string(run);
                    }
                }

                vector<BatchResult> results(tasks.size());
                atomic<size_t> nextTask{ 0 };
                atomic<size_t> doneCount{ 0 };
                mutex ioMutex;

                auto worker = [&]() {
                    while (true) {
                        size_t taskIndex = nextTask.fetch_add(1);
                        if (taskIndex >= tasks.size()) break;
                        const BatchTask& task = tasks[taskIndex];

                        try {
                            uint64_t taskSeed = 0x9E3779B97f4A7C15ULL * static_cast<uint64_t>(taskIndex + 1);
                            uint64_t seed = mix_seed(baseSeed ^ taskSeed);
                            Matrix Dg = gibbsMetropolisSampler(D1, AD, outlet, task.beta, alpha, seed);
                            results[taskIndex].q = computeQVector(Dg, AD, outlet);
                            results[taskIndex].ok = true;
                        }
                        catch (const exception& e) {
                            results[taskIndex].ok = false;
                            results[taskIndex].err = e.what();
                        }
                        catch (...) {
                            results[taskIndex].ok = false;
                            results[taskIndex].err = "unknown error";
                        }

                        size_t done = doneCount.fetch_add(1) + 1;
                        if ((done % 10) == 0 || done == tasks.size()) {
                            lock_guard<mutex> lk(ioMutex);
                            cout << "\rProgress: " << done << " / " << tasks.size() << flush;
                        }
                    }
                };

                auto tb0 = tic();
                vector<thread> pool;
                pool.reserve(static_cast<size_t>(nThreads));
                for (int t = 0; t < nThreads; ++t) pool.emplace_back(worker);
                for (thread& th : pool) th.join();
                double batchSec = toc_sec(tb0);
                cout << "\nBatch completed. batch time = " << batchSec << " sec\n";

                size_t failCount = 0;
                for (size_t ti = 0; ti < tasks.size(); ++ti) {
                    const BatchTask& task = tasks[ti];
                    if (!results[ti].ok) {
                        failCount++;
                        all_q[task.outIndex] = {};
                    }
                    else {
                        all_q[task.outIndex] = move(results[ti].q);
                    }
                }
                if (failCount > 0) {
                    cout << "Warning: failed tasks = " << failCount << " / " << tasks.size() << "\n";
                }

                reportNSE_andSave(
                    inputName,
                    all_q[0],
                    tasks,
                    results,
                    all_q,
                    alpha,
                    iterations,
                    baseSeed
                );
            }

            string wfname = fs::path(inputName).filename().string() + "_width_functions.csv";
            writeWidthFunctionsCsv(headers, all_q, wfname);
            cout << "Saved: " << wfname << "\n";
        }

        cout << "\nEnd.\nEnter any value and press Enter to exit. MADE BY Yeungnam University, Department of Civil Engineering\n";
        int dummy = 0;
        if (!(cin >> dummy)) {
            cin.clear();
            cout << "(Input stream is closed) Press Enter to exit...\n";
            cin.get();
        }
        return 0;
    }
    catch (const exception& e) {
        cerr << "\n[Fatal exception] " << e.what() << "\n";
        cout << "Press Enter to exit...\n";
        cin.get();
        cin.get();
        return 1;
    }
    catch (...) {
        cerr << "\n[Fatal exception] unknown\n";
        cout << "Press Enter to exit...\n";
        cin.get();
        cin.get();
        return 1;
    }
}
