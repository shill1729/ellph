// LPType.cpp
#include "LPType.hpp"
#include <algorithm>
#include <stdexcept>
#include <limits>
#include <cmath>
#include <numeric>

EllipsoidLPOracle::EllipsoidLPOracle(const std::vector<Ellipsoid>& all, int ambient_dim, LPParams p)
: all_(all), d_(ambient_dim), P_(p) {
    if (all_.empty()) throw std::invalid_argument("Oracle: empty ellipsoid set");
}

KObjective EllipsoidLPOracle::make_K_for_subset(const std::vector<int>& subset) const {
    std::vector<Ellipsoid> Es; Es.reserve(subset.size());
    for (int idx : subset) Es.push_back(all_[idx]);
    return make_Kobjective_from_ellipsoids(/*epsilon*/1.0, Es);
}


// uint64_t EllipsoidLPOracle::key_from_indices(const std::vector<int>& idx) {
//     // order-insensitive hash: 64-bit mix of sorted indices
//     uint64_t h = 1469598103934665603ULL;
//     for (int v : idx) { h ^= (uint64_t)(v + 0x9e3779b97f4a7c15ULL); h *= 1099511628211ULL; }
//     return h;
// }
static inline uint64_t fnv_mix(uint64_t h, uint64_t x){
    h ^= x + 0x9e3779b97f4a7c15ULL;
    h *= 1099511628211ULL;
    return h;
}

uint64_t EllipsoidLPOracle::key_from_set(const std::vector<int>& idx) {
    std::vector<int> s = idx;
    std::sort(s.begin(), s.end());
    uint64_t h = 1469598103934665603ULL;
    for (int v : s) h = fnv_mix(h, (uint64_t)v);
    return h;
}

// Changing!
// LPEval EllipsoidLPOracle::evaluate(const std::vector<int>& B) const {
//     if (B.empty()) {
//         // f(∅)=0; m doesn't matter. Return sane defaults.
//         LPEval z; z.eps_star = 0.0;
//         z.m = Eigen::VectorXd::Zero(d_);
//         z.dists = Eigen::VectorXd::Zero(0);
//         z.lambda = Eigen::VectorXd::Zero(0);
//         return z;
//     }
//     // auto K = make_K_for_subset(B);
//     // auto res = optimal_radius(K, P_.inner);
//     // // Bring λ* back to a vector aligned to B
//     // Eigen::VectorXd lam = res.lambda_star;        // size |B|
//     // Eigen::VectorXd d   = res.dists;              // size |B|
//     // return {res.eps_star, K.centroid(), d, lam};
//     auto key = key_from_set(B);
//     auto it = cache_.find(key);
//     if (it != cache_.end()) return it->second;

//     auto K = make_K_for_subset(B);
//     auto res = optimal_radius(K, P_.inner);
//     LPEval ev{res.eps_star, K.centroid(), res.dists, res.lambda_star};
//     cache_.emplace(key, ev);
//     return ev;
// }
LPEval EllipsoidLPOracle::evaluate(const std::vector<int>& B) const {
    if (B.empty()) {
        LPEval z; z.eps_star = 0.0;
        z.m = Eigen::VectorXd::Zero(d_);
        z.dists = Eigen::VectorXd::Zero(0);
        z.lambda = Eigen::VectorXd::Zero(0);
        return z;
    }

    const uint64_t key = key_from_set(B);
    CacheVal cv;
    auto it = cache_.find(key);
    if (it == cache_.end()) {
        // Solve on a canonical order (sorted), but cache only (eps, m)
        std::vector<int> Bsorted = B;
        std::sort(Bsorted.begin(), Bsorted.end());
        auto K = make_K_for_subset(Bsorted);
        auto res = optimal_radius(K, P_.inner);  // returns eps_star, dists (in that order), lambda_star
        cv = CacheVal{res.eps_star, K.centroid()};
        cache_.emplace(key, cv);
    } else {
        cv = it->second;
    }

    // Recompute per-constraint distances in the **caller’s order**
    Eigen::VectorXd d(B.size());
    for (int t = 0; t < (int)B.size(); ++t) {
        const Ellipsoid& Ei = all_[ B[t] ];
        const Eigen::VectorXd diff = cv.m - Ei.center();
        const double d2 = diff.transpose() * (Ei.precision() * diff);
        d[t] = std::sqrt(d2);  // distances (not squared), to match your convention
    }

    // λ* not needed by Seidel; leave empty unless you really need it
    return LPEval{cv.eps_star, cv.m, d, Eigen::VectorXd()};
}


// bool EllipsoidLPOracle::is_violator(const LPBasis& B, int i) const {
//     // Seed: empty basis cannot certify anything; force-add the first constraint.
//     if (B.idx.empty()) return true;

//     // Get the current solution (centroid m(B), radius eps*(B)) for the basis.
//     LPEval evB = evaluate(B.idx);

//     const Ellipsoid& Ei = all_[i];
//     const Eigen::VectorXd diff = evB.m - Ei.center();
//     const double di = diff.transpose() * (Ei.precision() * diff); // ||diff||^2_{A_i^{-1}}

//     // Violates if outside the certified ball
//     return (std::sqrt(di) > evB.eps_star + P_.tight_tol);
// }

bool EllipsoidLPOracle::is_violator(const LPBasis& B, int i, const LPEval& evB) const {
    if (B.idx.empty()) return true; // seed the first constraint
    const Ellipsoid& Ei = all_[i];
    const Eigen::VectorXd diff = evB.m - Ei.center();
    const double di_sq = diff.transpose() * (Ei.precision() * diff);
    return (std::sqrt(di_sq) > evB.eps_star + P_.tight_tol);
}

// Keep the old is_violator(B,i) as a slow fallback that just calls evaluate(B.idx) once:
bool EllipsoidLPOracle::is_violator(const LPBasis& B, int i) const {
    LPEval evB = evaluate(B.idx);
    return is_violator(B, i, evB);
}

// std::vector<int> EllipsoidLPOracle::shrink_tight(const std::vector<int>& tight,
//                                                  const std::vector<int>& superset,
//                                                  const LPEval& ev) const {
//     // Generic position => |tight|<=D_+1. In degeneracy, pick the (D_+1) most "tight".
//     if ((int)tight.size() <= d_ + 1) return tight;
//     // Build pair (gap, global_idx), smaller gap preferred
//     struct Item{ double gap; int gidx; };
//     std::vector<Item> items; items.reserve(tight.size());
//     for (int t = 0; t < (int)tight.size(); ++t) {
//         int gidx = tight[t];
//         // locate gidx inside superset to get its local position in ev.dists
//         int pos = int(std::find(superset.begin(), superset.end(), gidx) - superset.begin());
//         // const double gap = std::abs(std::sqrt(ev.dists[pos]) - ev.eps_star);
//         const double gap = std::abs(ev.dists[pos] - ev.eps_star);
//         items.push_back({gap, gidx});
//     }
//     std::nth_element(items.begin(), items.begin() + (d_+1), items.end(),
//                      [](const Item& a, const Item& b){ return a.gap < b.gap; });
//     std::vector<int> picked; picked.reserve(d_+1);
//     for (int j = 0; j < d_+1; ++j) picked.push_back(items[j].gidx);
//     return picked;
// }
std::vector<int> EllipsoidLPOracle::shrink_tight(const std::vector<int>& tight,
                                                 const std::vector<int>& superset,
                                                 const LPEval& ev) const {
    if ((int)tight.size() <= d_ + 1) return tight;

    std::unordered_map<int,int> pos; pos.reserve(superset.size()*2);
    for (int j = 0; j < (int)superset.size(); ++j) pos[superset[j]] = j;

    struct Item{ double gap; int gidx; };
    std::vector<Item> items; items.reserve(tight.size());
    for (int gidx : tight) {
        int j = pos[gidx];
        const double gap = std::abs(ev.dists[j] - ev.eps_star); // distances, not squared
        items.push_back({gap, gidx});
    }
    const int need = d_ + 1;
    std::nth_element(items.begin(), items.begin()+need, items.end(),
                     [](const Item& a, const Item& b){ return a.gap < b.gap; });

    std::vector<int> picked; picked.reserve(need);
    for (int j = 0; j < need; ++j) picked.push_back(items[j].gidx);
    // optional determinism
    std::sort(picked.begin(), picked.end());
    return picked;
}


LPBasis EllipsoidLPOracle::compute_basis(const std::vector<int>& C) const {
    if (C.empty()) return LPBasis{{}, 0.0};

    // Solve on C
    LPEval ev = evaluate(C);
    // Tight set T := { j in C : |d_j - eps*| <= tol }
    std::vector<int> T;
    for (int t = 0; t < (int)C.size(); ++t) {
        // if (std::abs(std::sqrt(ev.dists[t]) - ev.eps_star) <= P_.tight_tol) T.push_back(C[t]);
        if (std::abs(ev.dists[t] - ev.eps_star) <= P_.tight_tol) T.push_back(C[t]);
    }
    // if (T.empty()) {
    //     // Numerical fallback: pick the argmax distance as tight
    //     int argmax = 0; double best = -1.0;

    //     // Need to change this:
    //     // for (int t = 0; t < (int)C.size(); ++t) {
    //     //     double val = ev.dists[t];
    //     //     if (val > best) { best = val; argmax = t; }
    //     // }
    //     // T.push_back(C[argmax]);
    //     for (int t = 0; t < (int)C.size(); ++t) 
    //     {
    //         if (std::abs(ev.dists[t] - ev.eps_star) <= P_.tight_tol)
    //         {
    //             T.push_back(C[t]);
    //         }
    //     }
    // }
    if (T.empty()) {
        int argmax = 0; double best = -1.0;
        for (int t = 0; t < (int)C.size(); ++t) {
            double val = ev.dists[t];
            if (val > best) { best = val; argmax = t; }
        }
        T.push_back(C[argmax]);
    }


    auto Bidx = shrink_tight(T, C, ev);
    // Recompute eps* on the basis itself (cheap, usually unchanged)
    LPEval evB = evaluate(Bidx);
    return LPBasis{Bidx, evB.eps_star};
}