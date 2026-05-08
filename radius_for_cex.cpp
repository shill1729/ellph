// main_three_ellipses.cpp
//
// Three hard-coded ellipses in R^2. Run all solvers and print the optimal
// intersection radius ε* returned by each method.

#include "Ellipsoid.hpp"
#include "KFromEllipsoids.hpp"
#include "OptimalRadius.hpp"
#include "LPType.hpp"
#include "LPSeidel.hpp"
#include "LPClarkson.hpp"

#include <Eigen/Dense>
#include <iostream>
#include <numeric>
#include <optional>
#include <vector>

int main() {
    std::cout.setf(std::ios::fixed);
    std::cout.precision(9);

    using Mat = Ellipsoid::Mat;   // typically Eigen::MatrixXd
    using Vec = Ellipsoid::Vec;   // typically Eigen::VectorXd

    const int d = 2;

    // --------- define 3 ellipses in precision form (x-c)^T A (x-c) ≤ r ---------

    // E1: center (0,0), axes-aligned
    Vec c1(d); c1 << 0.0, 0.0;
    Mat A1(d, d);
    A1 << 8.5, 7.5,
          7.5, 8.5;
    double r1 = 1.0;

    // E2: center (1.1, 0.1), axes-aligned, tighter in y
    Vec c2(d); c2 << 1., 0.;
    Mat A2(d, d);
    A2 << 16, 0.,
          0.0, 1;
    double r2 = 1.0;

    Vec c3(d); c3 << 0., 1.;
    Mat A3(d, d);
    A3 << 1, 0.,
          0.0, 16;
    double r3 = 1.0;

    // Construct Ellipsoid objects using precision (leave covariance null)
    std::vector<Ellipsoid> Es;
    Es.reserve(3);
    // Es.emplace_back(c1, std::nullopt, std::optional<Mat>{A1}, r1);
    // Es.emplace_back(c2, std::nullopt, std::optional<Mat>{A2}, r2);
    // Es.emplace_back(c3, std::nullopt, std::optional<Mat>{A3}, r3);
    Es.emplace_back(c1, std::optional<Mat>{A1}, std::nullopt, r1);
    Es.emplace_back(c2, std::optional<Mat>{A2}, std::nullopt, r2);
    Es.emplace_back(c3, std::optional<Mat>{A3}, std::nullopt, r3);

    // K-objective and LP oracle
    auto K = make_Kobjective_from_ellipsoids(1.0, Es);
    EllipsoidLPOracle O(Es, d, LPParams{SolverKind::SLSQP, 1e-8});

    // Index set for LP-type methods
    std::vector<int> S(Es.size());
    std::iota(S.begin(), S.end(), 0);

    // --------- raw solvers on the full set ---------
    {
        auto res = optimal_radius(K, SolverKind::SLSQP);
        std::cout << "Raw-SLSQP   eps_star=" << res.eps_star << "\n";
    }
    {
        auto res = optimal_radius(K, SolverKind::PGD);
        std::cout << "Raw-PGD     eps_star=" << res.eps_star << "\n";
    }
    {
        auto res = optimal_radius(K, SolverKind::Cauchy);
        std::cout << "Raw-Cauchy  eps_star=" << res.eps_star << "\n";
    }

    // --------- LP-type methods (basis algorithms) ---------
    {
        SeidelOptions so;
        so.seed = 42;
        so.max_depth = -1;
        auto out = seidel_incremental(O, S, so);
        std::cout << "LP-Seidel   eps_star=" << out.basis.eps_star;
    }
    {
        ClarksonOptions co;
        co.rounds = 25;
        co.seed = 123;
        auto out = clarkson_iterative(O, S, co);
        std::cout << "LP-Clarkson eps_star=" << out.basis.eps_star;
    }

    return 0;
}
