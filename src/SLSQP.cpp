#include "SLSQP.hpp"
#include "Simplex.hpp"
#include <stdexcept>

namespace {
double wrapper(unsigned n, const double* x, double* grad, void* data) {
    KObjective* obj = static_cast<KObjective*>(data);
    Eigen::Map<const Eigen::VectorXd> lam(x, n);
    if (grad) {
        Eigen::Map<Eigen::VectorXd> g(grad, n);
        return obj->value_grad(lam, g);
    } else {
        return obj->value(lam);
    }
}
}

NloptResult minimize_slsqp(KObjective& obj, const Eigen::VectorXd& lambda0, const NloptOptions& opt) {
    const int k = static_cast<int>(lambda0.size());
    nlopt::opt opti(nlopt::LD_SLSQP, k);

    // Bounds λ_i ≥ 0
    std::vector<double> lb(k, 0.0), ub(k, 1.0);
    opti.set_lower_bounds(lb);
    opti.set_upper_bounds(ub);

    // Equality: sum λ_i = 1
    opti.add_equality_mconstraint(
        [](unsigned m, double* result, unsigned n, const double* x, double* grad, void*){
            double s = 0.0;
            for (unsigned i=0;i<n;++i) s += x[i];
            result[0] = s - 1.0;
            if (grad) for (unsigned i=0;i<n;++i) grad[i] = 1.0;
        },
        nullptr, std::vector<double>{1e-10}
    );

    opti.set_min_objective(wrapper, &obj);
    opti.set_maxeval(opt.max_evals);
    opti.set_xtol_rel(opt.rel_tol);
    opti.set_xtol_abs(opt.abs_tol);

    std::vector<double> x(k);
    Eigen::VectorXd lam0 = Simplex::project_to_simplex(lambda0);
    for (int i=0;i<k;++i) x[i] = lam0[i];

    double minf;
    nlopt::result status = opti.optimize(x, minf);

    Eigen::VectorXd lam(k);
    for (int i=0;i<k;++i) lam[i] = x[i];
    return {lam, minf, status};
}