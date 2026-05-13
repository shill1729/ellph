#include "LPClarkson.hpp"
#include <random>
#include <algorithm>
#include <numeric>

static std::vector<int> sample_weighted(const std::vector<int>& S,
                                        const std::vector<double>& w,
                                        int ksam,
                                        std::mt19937_64& rng)
{
    std::discrete_distribution<int> pick(w.begin(), w.end());
    std::vector<int> out; out.reserve(ksam);
    for (int t = 0; t < ksam; ++t) out.push_back(S[pick(rng)]);
    return out;
}

static std::vector<int> union_sorted(const std::vector<int>& a, const std::vector<int>& b)
{
    std::vector<int> c = a;
    c.insert(c.end(), b.begin(), b.end());
    std::sort(c.begin(), c.end());
    c.erase(std::unique(c.begin(), c.end()), c.end());
    return c;
}

ClarksonResult clarkson_iterative(const EllipsoidLPOracle& O,
                                  const std::vector<int>& S,
                                  ClarksonOptions opt)
{
    const int n = (int)S.size();
    const int d = O.d();
    const int D = d + 1;
    const int ksam0 = (opt.sample_size > 0) ? opt.sample_size : 8 * D * D;
    const int ksam = std::min(std::max(ksam0, D + 1), std::max(n, 1));

    std::vector<double> w(n, 1.0);
    std::mt19937_64 rng(opt.seed);

    LPBasis B{{}, 0.0};
    std::vector<int> core;  // light violators accumulated permanently
    int vt = 0, doublings = 0;
    bool found = false;

    auto scan_violators = [&](const LPBasis& Bcur) -> std::vector<int> {
        LPEval evB = O.evaluate(Bcur.idx);
        std::vector<int> viol;
        for (int t = 0; t < n; ++t) {
            ++vt;
            if (O.is_violator(Bcur, S[t], evB)) viol.push_back(t);
        }
        return viol;
    };

    for (int outer = 0; outer < opt.rounds && !found; ++outer) {
        auto Rset = sample_weighted(S, w, ksam, rng);
        auto subset = union_sorted(Rset, core);
        B = O.compute_basis(subset);

        auto violators = scan_violators(B);
        if (violators.empty()) { found = true; break; }

        double Wviol = 0.0, Wall = 0.0;
        for (int t = 0; t < n; ++t) Wall += w[t];
        for (int id : violators) Wviol += w[id];

        if (Wviol / std::max(Wall, 1e-300) >= opt.weight_bad_threshold) {
            for (int id : violators) w[id] *= 2.0;
            ++doublings;
        } else {
            // Add violators to core (mirrors Python reference)
            for (int id : violators) core.push_back(S[id]);
            std::sort(core.begin(), core.end());
            core.erase(std::unique(core.begin(), core.end()), core.end());

            for (int inner = 0; inner < opt.max_inner && !found; ++inner) {
                auto Rset2 = sample_weighted(S, w, ksam, rng);
                subset = union_sorted(Rset2, core);
                B = O.compute_basis(subset);

                violators = scan_violators(B);
                if (violators.empty()) { found = true; break; }

                Wviol = 0.0; Wall = 0.0;
                for (int t = 0; t < n; ++t) Wall += w[t];
                for (int id : violators) Wviol += w[id];

                if (Wviol / std::max(Wall, 1e-300) >= opt.weight_bad_threshold) {
                    for (int id : violators) w[id] *= 2.0;
                    ++doublings;
                } else {
                    for (int id : violators) core.push_back(S[id]);
                    std::sort(core.begin(), core.end());
                    core.erase(std::unique(core.begin(), core.end()), core.end());
                    break;
                }
            }
        }
    }

    // Certification: verify no violators before returning; fall back to full set if needed
    if (n > 0) {
        LPEval evB = O.evaluate(B.idx);
        for (int t = 0; t < n; ++t) {
            ++vt;
            if (O.is_violator(B, S[t], evB)) { B = O.compute_basis(S); break; }
        }
    }

    return {B, vt, doublings};
}
