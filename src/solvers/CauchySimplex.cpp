// ========================= src/solvers/CauchySimplex.cpp =========================
#include "solvers/CauchySimplex.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

using Vec = Eigen::VectorXd;

static inline double dot(const Vec& a, const Vec& b)
{
    return a.dot(b);
}

static inline void normalize_or_uniform(Vec& w)
{
    const double s = w.sum();

    if (s <= 0.0 || !std::isfinite(s)) {
        w.setConstant(1.0 / static_cast<double>(w.size()));
        return;
    }

    w.array() /= s;
}

static inline void make_simplex_feasible(Vec& w, double eps_zero)
{
    for (int i = 0; i < w.size(); ++i) {
        if (!std::isfinite(w[i]) || w[i] < 0.0) {
            w[i] = 0.0;
        }
    }

    normalize_or_uniform(w);

    // The paper initializes in relint(Delta_n). If the caller supplies boundary data,
    // push it slightly into the simplex before the iteration begins.
    for (int i = 0; i < w.size(); ++i) {
        if (w[i] <= eps_zero) {
            w[i] = eps_zero;
        }
    }

    normalize_or_uniform(w);
}

// c_i = grad_i f(w) - w · grad f(w)
static inline void centered_gradient(const Vec& w,
                                     const Vec& grad,
                                     Vec& c)
{
    const double mean_grad = dot(w, grad);
    c = grad.array() - mean_grad;
}

// Algorithm 1:
// S = { i : w_i > epsilon }
// eta_max = 1 / max_{i in S} (grad_i f(w) - w · grad f(w))
static inline double compute_eta_max(const Vec& w,
                                     const Vec& c,
                                     double eps_zero)
{
    double max_active_c = -std::numeric_limits<double>::infinity();

    for (int i = 0; i < w.size(); ++i) {
        if (w[i] > eps_zero) {
            max_active_c = std::max(max_active_c, c[i]);
        }
    }

    if (!std::isfinite(max_active_c)) {
        return 0.0;
    }

    // If max_i c_i <= 0, the paper remarks that eta_max = infinity.
    // For a finite implementation, this means the user-chosen eta is not capped here.
    if (max_active_c <= 0.0) {
        return std::numeric_limits<double>::infinity();
    }

    return 1.0 / max_active_c;
}

// Algorithm 1 update:
//   S = {i : w_i > eps}
//   Q = {i : w_i <= eps}
//   \hat w_i = w_i - eta w_i c_i for i in S
//   \hat w_j = 0 for j in Q
//   w_new = \hat w / sum_j \hat w_j
static inline Vec cauchy_simplex_candidate(const Vec& w,
                                           const Vec& c,
                                           double eta,
                                           double eps_zero)
{
    Vec candidate(w.size());

    for (int i = 0; i < w.size(); ++i) {
        if (w[i] <= eps_zero) {
            candidate[i] = 0.0;
        } else {
            candidate[i] = w[i] - eta * w[i] * c[i];

            // Guard only against roundoff. The eta cap is what enforces positivity.
            if (candidate[i] < 0.0 && candidate[i] > -100.0 * eps_zero) {
                candidate[i] = 0.0;
            }
        }
    }

    normalize_or_uniform(candidate);
    return candidate;
}

CSResult minimize_cauchy_simplex(KObjective& obj,
                                 const Vec& lambda0,
                                 const CSOptions& opt)
{
    Vec w = lambda0;
    make_simplex_feasible(w, opt.eps_zero);

    Vec grad(w.size());
    double f = obj.value_grad(w, grad);

    Vec c(w.size());

    for (int it = 1; it <= opt.max_iters; ++it) {
        centered_gradient(w, grad, c);

        // Paper direction d_i = w_i (grad_i f - w · grad f).
        // The actual minimization update is w_new = w - eta d.
        Vec direction = w.array() * c.array();

        if (direction.lpNorm<Eigen::Infinity>() <= opt.tol) {
            return {w, f, it - 1, true};
        }

        double eta_max = compute_eta_max(w, c, opt.eps_zero);

        if (eta_max <= 0.0) {
            return {w, f, it - 1, false};
        }

        double eta = opt.initial_eta;

        if (std::isfinite(eta_max)) {
            eta = std::min(eta, eta_max);

            // Section 3.2: taking eta = eta_max may incorrectly set an index to zero.
            eta *= opt.boundary_shrink;
        }

        if (eta <= 0.0 || !std::isfinite(eta)) {
            return {w, f, it - 1, false};
        }

        Vec candidate = w;
        double candidate_f = f;
        bool accepted = false;

        if (opt.armijo) {
            // Directional derivative along update w - eta d is -grad · d.
            const double grad_dot_direction = dot(grad, direction);

            double trial_eta = eta;

            for (int ls = 0; ls < opt.max_line_search_steps; ++ls) {
                candidate = cauchy_simplex_candidate(
                    w,
                    c,
                    trial_eta,
                    opt.eps_zero
                );

                candidate_f = obj.value(candidate);

                // Armijo decrease for minimization:
                // f(w - eta d) <= f(w) - c eta grad·d.
                if (candidate_f <= f - opt.armijo_c * trial_eta * grad_dot_direction) {
                    accepted = true;
                    eta = trial_eta;
                    break;
                }

                trial_eta *= opt.armijo_beta;

                if (trial_eta <= 0.0 || trial_eta < std::numeric_limits<double>::epsilon()) {
                    break;
                }
            }
        } else {
            candidate = cauchy_simplex_candidate(w, c, eta, opt.eps_zero);
            candidate_f = obj.value(candidate);
            accepted = std::isfinite(candidate_f);
        }

        if (!accepted) {
            return {w, f, it - 1, false};
        }

        if ((candidate - w).lpNorm<Eigen::Infinity>() <= opt.tol &&
            std::abs(candidate_f - f) <= opt.tol * std::max(1.0, std::abs(f))) {
            return {candidate, candidate_f, it, true};
        }

        w.swap(candidate);
        f = obj.value_grad(w, grad);
    }

    return {w, f, opt.max_iters, false};
}