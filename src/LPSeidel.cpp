#include "LPSeidel.hpp"
#include <random>
#include <algorithm>

// static SeidelResult seidel_inner(const EllipsoidLPOracle& O,
//                                  const std::vector<int>& perm,
//                                  int upto,
//                                  LPBasis B,
//                                  int depth,
//                                  int& vt_count)
// {
//     if (upto < 0) return {B, vt_count};
//     if (depth <= 0) return {B, vt_count}; // safety guard
//     // Solve with first 'upto' elements processed
//     SeidelResult r = seidel_inner(O, perm, upto-1, B, depth, vt_count);

//     int x = perm[upto];
//     ++vt_count;
//     if (!O.is_violator(r.basis, x)) {
//         return r; // nothing to do
//     }
//     // x violates => it must be in the basis; recurse with x added to prefix
//     std::vector<int> C = r.basis.idx;
//     C.push_back(x);
//     LPBasis Bnew = O.compute_basis(C);
//     return seidel_inner(O, perm, upto-1, Bnew, depth-1, vt_count);
// }
static SeidelResult seidel_inner(const EllipsoidLPOracle& O,
                                 const std::vector<int>& perm,
                                 int upto,
                                 LPBasis B,
                                 int depth,
                                 int& vt_count)
{
    if (upto < 0 || depth <= 0) return {B, vt_count};

    // Recurse on prefix
    SeidelResult r = seidel_inner(O, perm, upto-1, B, depth, vt_count);

    // Evaluate **once** on the current basis
    LPEval evB = O.evaluate(r.basis.idx);

    // Check violator with cached evB
    int x = perm[upto];
    ++vt_count;
    if (!O.is_violator(r.basis, x, evB)) return r;

    // Violation: grow basis and recompute
    std::vector<int> C = r.basis.idx; C.push_back(x);
    LPBasis Bnew = O.compute_basis(C);

    // Tail recurse with the **new** basis
    return seidel_inner(O, perm, upto-1, Bnew, depth-1, vt_count);
}

// SeidelResult seidel_incremental(const EllipsoidLPOracle& O,
//                                 const std::vector<int>& S,
//                                 SeidelOptions opt)
// {
//     std::vector<int> perm = S;
//     std::mt19937_64 rng(opt.seed);
//     std::shuffle(perm.begin(), perm.end(), rng);

//     int vt = 0;
//     LPBasis B0{{}, 0.0};
//     auto out = seidel_inner(O, perm, (int)perm.size()-1, B0, opt.max_depth, vt);
//     out.violation_tests = vt;
//     return out;
// }
SeidelResult seidel_incremental(const EllipsoidLPOracle& O,
                                const std::vector<int>& S,
                                SeidelOptions opt)
{
    std::vector<int> perm = S;
    std::mt19937_64 rng(opt.seed);
    std::shuffle(perm.begin(), perm.end(), rng);

    int vt = 0;
    LPBasis B0{{}, 0.0};
    const int depth0 = (opt.max_depth < 0 ? O.d() + 1 : opt.max_depth);
    auto out = seidel_inner(O, perm, (int)perm.size()-1, B0, depth0, vt);
    out.violation_tests = vt;
    return out;
}