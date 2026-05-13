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

static std::vector<Ellipsoid> ellipsoid_subset(const std::vector<Ellipsoid>& all,
                                               const std::vector<int>& subset) {
    std::vector<Ellipsoid> Es;
    Es.reserve(subset.size());
    for (int idx : subset) Es.push_back(all[idx]);
    return Es;
}

static SolverKind dual_solver_kind(BasisSolverKind solver) {
    switch (solver) {
        case BasisSolverKind::DualPGD:
            return SolverKind::PGD;
        case BasisSolverKind::DualCauchy:
            return SolverKind::Cauchy;
        case BasisSolverKind::DualSLSQP:
        case BasisSolverKind::PrimalSOCP:
            return SolverKind::SLSQP;
    }
    return SolverKind::SLSQP;
}

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
        std::vector<int> Bsorted = B;
        std::sort(Bsorted.begin(), Bsorted.end());
        if (P_.inner == BasisSolverKind::PrimalSOCP) {
            auto Es = ellipsoid_subset(all_, Bsorted);
            auto res = solve_socp_alglib(Es);
            cv = CacheVal{res.eps_star, res.m, Eigen::VectorXd()};
        } else {
            auto K = make_K_for_subset(Bsorted);
            auto res = optimal_radius(K, dual_solver_kind(P_.inner));
            cv = CacheVal{res.eps_star, K.centroid(), res.lambda_star};
        }
        cache_.emplace(key, cv);
    } else {
        cv = it->second;
    }

    const int sz = (int)B.size();
    Eigen::VectorXd d(sz);
    for (int t = 0; t < sz; ++t) {
        const Ellipsoid& Ei = all_[B[t]];
        const Eigen::VectorXd diff = cv.m - Ei.center();
        d[t] = std::sqrt(diff.transpose() * (Ei.precision() * diff));
    }

    // Align lambda to caller’s order (cv.lambda is aligned to sorted order)
    Eigen::VectorXd lam_caller;
    if (cv.lambda.size() == sz) {
        std::vector<int> Bsorted = B;
        std::sort(Bsorted.begin(), Bsorted.end());
        lam_caller.resize(sz);
        for (int t = 0; t < sz; ++t) {
            auto pos = std::lower_bound(Bsorted.begin(), Bsorted.end(), B[t]) - Bsorted.begin();
            lam_caller[t] = cv.lambda[(int)pos];
        }
    }

    return LPEval{cv.eps_star, cv.m, d, lam_caller};
}

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

    LPEval ev = evaluate(C);
    std::vector<int> Bidx;

    if (ev.lambda.size() == (int)C.size()) {
        // Prefer lambda-support basis (matches Python reference)
        for (int t = 0; t < (int)C.size(); ++t) {
            if (ev.lambda[t] > P_.support_tol) Bidx.push_back(C[t]);
        }
        if ((int)Bidx.size() > d_ + 1) {
            Bidx = shrink_tight(Bidx, C, ev);
        }
    }

    if (Bidx.empty()) {
        // Fall back: tight-distance approach (used for SOCP or empty support)
        std::vector<int> T;
        for (int t = 0; t < (int)C.size(); ++t) {
            if (std::abs(ev.dists[t] - ev.eps_star) <= P_.tight_tol) T.push_back(C[t]);
        }
        if (T.empty()) {
            int argmax = 0; double best = -1.0;
            for (int t = 0; t < (int)C.size(); ++t) {
                if (ev.dists[t] > best) { best = ev.dists[t]; argmax = t; }
            }
            T.push_back(C[argmax]);
        }
        Bidx = shrink_tight(T, C, ev);
    }

    std::sort(Bidx.begin(), Bidx.end());
    LPEval evB = evaluate(Bidx);
    return LPBasis{Bidx, evB.eps_star};
}
