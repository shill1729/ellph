#include "CauchySimplex.hpp"
#include <algorithm>
#include <numeric>

static inline double dot(const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
    return a.dot(b);
}

// Compute centered gradient c = g - (w·g) * 1
static inline void centered_grad(const Eigen::VectorXd& w,
                                 const Eigen::VectorXd& g,
                                 Eigen::VectorXd& c)
{
    const double wg = dot(w, g);
    c = g.array() - wg;
}

// eta_max = 1 / max_{i in active} c_i, with active = {i: w_i > 0}
static inline double eta_max_cap(const Eigen::VectorXd& w,
                                 const Eigen::VectorXd& c)
{
    double maxci = 0.0;
    for (int i = 0; i < w.size(); ++i) {
        if (w[i] > 0.0) maxci = std::max(maxci, c[i]);
    }
    if (maxci <= 0.0) return std::numeric_limits<double>::infinity();
    return 1.0 / maxci;
}

static inline void zero_clip_and_renorm(Eigen::VectorXd& w,
                                        double eps_clip,
                                        bool renorm)
{
    double s = 0.0;
    for (int i = 0; i < w.size(); ++i) {
        if (w[i] < eps_clip) w[i] = 0.0;
        s += w[i];
    }
    if (renorm) {
        if (s <= 0.0) {
            // fallback to uniform if all got clipped (pathological)
            w.setConstant(1.0 / double(w.size()));
        } else {
            w.array() /= s;
        }
    }
}

CSResult minimize_cauchy_simplex(KObjective& obj,
                                 const Eigen::VectorXd& w0,
                                 const CSOptions& opt)
{
    using Vec = Eigen::VectorXd;
    Vec w = w0;
    // Ensure feasibility: strictly interior start is preferred
    double sumw = w.sum();
    if (sumw <= 0) w.setConstant(1.0 / double(w.size()));
    else w.array() /= sumw;
    for (int i = 0; i < w.size(); ++i) if (w[i] < opt.eps_clip) w[i] = std::max(w[i], 1e-6);

    Vec g; g.resize(w.size());
    double f = obj.value_grad(w, g);

    Vec c(g.size()), d(g.size());
    for (int it = 0; it < opt.max_iters; ++it) {
        centered_grad(w, g, c);            // c = g - (w·g)1
        d = w.array() * c.array();         // d_i = w_i * c_i

        // Check first-order stationarity: projected grad Π_w g -> small
        const double pg_norm = (c.array().square() * w.array()).sqrt().matrix().norm(); // ||W^(1/2) c||
        if (pg_norm < opt.tol * std::max(1.0, g.norm())) {
            return {w, f, it, true};
        }

        // Step-size cap
        double eta_cap = eta_max_cap(w, c);
        if (!std::isfinite(eta_cap)) {
            // All c_i <= 0 ⇒ already optimal
            return {w, f, it, true};
        }
        eta_cap = std::max(0.0, eta_cap - opt.eta_shrink);

        // Line search on [0, eta_cap]
        double eta = eta_cap;
        Vec w_new;
        double f_new;

        if (opt.armijo) {
            double gTd = dot(g, d); // note: descent uses w+ = w - eta d
            // start at full eta, backtrack
            while (true) {
                w_new = w - eta * d;
                // positivity guaranteed if eta <= eta_cap, but we still clip tiny negatives
                zero_clip_and_renorm(w_new, opt.eps_clip, opt.renormalize);
                f_new = obj.value(w_new);
                if (f_new <= f - opt.armijo_c * eta * gTd || eta <= 1e-16) break;
                eta *= opt.armijo_beta;
            }
        } else {
            // No Armijo: just take the capped step
            w_new = w - eta * d;
            zero_clip_and_renorm(w_new, opt.eps_clip, opt.renormalize);
            f_new = obj.value(w_new);
        }

        // Convergence check
        if ((w_new - w).norm() < opt.tol * std::max(1.0, w.norm()) &&
            std::abs(f_new - f)   < opt.tol * std::max(1.0, std::abs(f))) {
            return {w_new, f_new, it+1, true};
        }

        w.swap(w_new);
        f = obj.value_grad(w, g);
    }
    return {w, f, opt.max_iters, false};
}