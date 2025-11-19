#include "LPClarkson.hpp"
#include <random>
#include <algorithm>
#include <numeric>

ClarksonResult clarkson_iterative(const EllipsoidLPOracle& O,
                                  const std::vector<int>& S,
                                  ClarksonOptions opt)
{
    const int n = (int)S.size();
    const int d = O.d();
    const int ksam = (opt.sample_size > 0) ? opt.sample_size : 4*(d+1)*(d+1);

    std::vector<double> w(n, 1.0);
    std::mt19937_64 rng(opt.seed);

    LPBasis B{{}, 0.0};
    int vt = 0, doublings = 0;

    for (int round = 0; round < opt.rounds; ++round) {
        // sample ksam indices with probability proportional to weight
        std::discrete_distribution<int> pick(w.begin(), w.end());
        std::vector<int> R; R.reserve(ksam);
        for (int t = 0; t < ksam && n>0; ++t) R.push_back(S[pick(rng)]);

        // Build candidate set C = R ∪ B
        std::vector<int> C = R;
        C.insert(C.end(), B.idx.begin(), B.idx.end());
        std::sort(C.begin(), C.end());
        C.erase(std::unique(C.begin(), C.end()), C.end());

        // // Compute basis of C
        // B = O.compute_basis(C);

        // // Scan violators in S; also accumulate their total weight
        // double Wviol = 0.0, Wall = 0.0;
        // std::vector<int> violators;
        // violators.reserve(n);
        // for (int t = 0; t < n; ++t) {
        //     ++vt;
        //     if (O.is_violator(B, S[t])) {
        //         violators.push_back(t);
        //         Wviol += w[t];
        //     }
        //     Wall += w[t];
        // }
        // Compute basis of C
        B = O.compute_basis(C);

        // Evaluate once on B
        LPEval evB = O.evaluate(B.idx);

        // Scan violators using cached evB
        double Wviol = 0.0, Wall = 0.0;
        std::vector<int> violators; violators.reserve(n);
        for (int t = 0; t < n; ++t) {
            ++vt;
            const bool v = O.is_violator(B, S[t], evB);
            if (v) { violators.push_back(t); Wviol += w[t]; }
            Wall += w[t];
        }

        if (Wviol / std::max(Wall, 1e-300) > opt.weight_bad_threshold) {
            for (int id : violators) w[id] *= 2.0; // double weights of bad guys
            ++doublings;
            continue; // next round
        }

        // Success if no violators (or negligible)
        if (violators.empty()) break;
    }
    return {B, vt, doublings};
}