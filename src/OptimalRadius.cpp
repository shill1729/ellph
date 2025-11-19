#include "OptimalRadius.hpp"
#include "Simplex.hpp"
#include <cmath>

EpsStar optimal_radius(KObjective& obj, SolverKind solver) {
    const int k = obj.k();
    auto lam0 = Simplex::uniform_start(k);

    Eigen::VectorXd lam_star;
    double fval;

    switch (solver) {
        case SolverKind::PGD: {
            PGDOptions o; o.max_iters=2000; o.tol=1e-10;
            auto res = minimize_pgd(obj, lam0, o);
            lam_star = res.lambda; fval = res.fval; break;
        }
        case SolverKind::Cauchy: {
            CSOptions o; o.max_iters=4000; o.tol=1e-10;
            auto res = minimize_cauchy_simplex(obj, lam0, o);
            lam_star = res.lambda; fval = res.fval; break;
        }
        case SolverKind::SLSQP: {
            NloptOptions o; o.max_evals=5000; o.rel_tol=1e-10; o.abs_tol=1e-12;
            auto res = minimize_slsqp(obj, lam0, o);
            lam_star = res.lambda; fval = res.fval; break;
        }
    }

    // Ensure internal state is consistent with lam_star (value_grad fills centroid + d2)
    Eigen::VectorXd g;
    g.resize(lam_star.size());
    obj.value_grad(lam_star, g);
    const auto& m = obj.centroid();
    const auto& d2 = obj.mahalanobis_d2();
    Eigen::VectorXd d = d2.array().sqrt();

    double eps_star = d.maxCoeff();
    return {eps_star, lam_star, d};
}