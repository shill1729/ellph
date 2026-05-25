#include "LPSeidel.hpp"
#include <random>
#include <algorithm>


static void seidel_inner(const EllipsoidLPOracle& O,
                         const std::vector<int>& perm,
                         int prefix_end,
                         const std::vector<int>& R,
                         int D,
                         LPBasis& out,
                         int& vt_count)
{
    if ((int)R.size() > D || prefix_end <= 0) {
        std::vector<int> subset(perm.begin(), perm.begin() + prefix_end);
        subset.insert(subset.end(), R.begin(), R.end());
        std::sort(subset.begin(), subset.end());
        subset.erase(std::unique(subset.begin(), subset.end()), subset.end());
        out = subset.empty() ? LPBasis{{}, 0.0} : O.compute_basis(subset);
        return;
    }

    seidel_inner(O, perm, prefix_end - 1, R, D, out, vt_count);

    int j = perm[prefix_end - 1];
    LPEval evB = O.evaluate(out.idx);
    ++vt_count;
    if (!O.is_violator(out, j, evB)) return;

    std::vector<int> R2 = R;
    R2.push_back(j);
    seidel_inner(O, perm, prefix_end - 1, R2, D, out, vt_count);
}


SeidelResult seidel_incremental(const EllipsoidLPOracle& O,
                                const std::vector<int>& S,
                                SeidelOptions opt)
{
    std::vector<int> perm = S;
    std::mt19937_64 rng(opt.seed);
    std::shuffle(perm.begin(), perm.end(), rng);

    int vt = 0;
    const int D = (opt.max_depth < 0 ? O.d() + 1 : opt.max_depth);
    LPBasis B{{}, 0.0};
    seidel_inner(O, perm, (int)perm.size(), {}, D, B, vt);
    return {B, vt};
}
