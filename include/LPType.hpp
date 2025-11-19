// LPType.hpp
#pragma once
#include "Ellipsoid.hpp"
#include "KFromEllipsoids.hpp"
#include "OptimalRadius.hpp"
#include <vector>
#include <optional>
#include <random>

struct LPBasis {
    std::vector<int> idx;      // indices into the global array S = {0..n-1}
    double eps_star = 0.0;     // f(B)
};

struct LPEval {
    double eps_star;           // f(B)
    Eigen::VectorXd m;         // centroid at λ*
    Eigen::VectorXd dists;     // per-ellipse distances at m
    Eigen::VectorXd lambda;    // λ* on B (size |B|)
};

struct CacheVal {
    double eps_star;
    Eigen::VectorXd m;
};

struct LPParams {
    SolverKind inner = SolverKind::SLSQP; // your 3 options
    double tight_tol = 1e-5;              // d_j within tol of eps* => tight
};

class EllipsoidLPOracle {
    public:
        EllipsoidLPOracle(const std::vector<Ellipsoid>& all, int ambient_dim, LPParams p);

        

        // Evaluate f(B) and related quantities (over B only)
        LPEval evaluate(const std::vector<int>& B) const;

        // Violation test: does i violate the basis B?
        bool is_violator(const LPBasis& B, int i) const;

        bool is_violator(const LPBasis& B, int i, const LPEval& evB) const;

        // Compute (a) tight set for C, (b) reduced basis <= d+1 indices
        LPBasis compute_basis(const std::vector<int>& C) const;

        int d() const noexcept { return d_; }
        int n() const noexcept { return static_cast<int>(all_.size()); }

    private:
        const std::vector<Ellipsoid>& all_;
        int d_;
        LPParams P_;
        // mutable std::unordered_map<uint64_t, LPEval> cache_;
        // static uint64_t key_from_indices(const std::vector<int>& idx);

        mutable std::unordered_map<uint64_t, CacheVal> cache_;
        static uint64_t key_from_set(const std::vector<int>& idx); // sorts internally

        // helper to build KObjective from a subset
        KObjective make_K_for_subset(const std::vector<int>& subset) const;

        // shrink a tight set deterministically to <= d_+1 indices
        std::vector<int> shrink_tight(const std::vector<int>& tight,
                                    const std::vector<int>& superset,
                                    const LPEval& ev) const;
};