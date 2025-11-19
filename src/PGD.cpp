#include "PGD.hpp"
#include "Simplex.hpp"
#include <cmath>

PGDResult minimize_pgd(KObjective& obj, const Eigen::VectorXd& lambda0, const PGDOptions& opt) {
    using Vec = Eigen::VectorXd;
    Vec lam = Simplex::project_to_simplex(lambda0);
    Vec g; g.resize(lam.size());
    double f = obj.value_grad(lam, g);

    for (int it = 0; it < opt.max_iters; ++it) {
        // Feasible descent direction via projected step
        const Vec z = lam - opt.step0 * g;
        Vec cand = Simplex::project_to_simplex(z);

        // Armijo backtracking on the segment lam -> cand
        double alpha = 1.0;
        const double gTd = g.dot(cand - lam);
        Vec lam_new = cand;
        double f_new = obj.value(lam_new);
        while (f_new > f + opt.armijo_c * alpha * gTd) {
            if (alpha < 1e-12) break;
            alpha *= opt.armijo_beta;
            lam_new = Simplex::project_to_simplex(lam + alpha * (cand - lam));
            f_new = obj.value(lam_new);
        }

        if ((lam_new - lam).norm() < opt.tol * std::max(1.0, lam.norm()) &&
            std::abs(f_new - f) < opt.tol * std::max(1.0, std::abs(f))) {
            return {lam_new, f_new, it+1, true};
        }

        lam.swap(lam_new);
        f = obj.value_grad(lam, g);
    }
    return {lam, f, opt.max_iters, false};
}