// ========================= include/solvers/CauchySimplex.hpp =========================
#pragma once

#include <Eigen/Dense>

#include "solvers/KObjective.hpp"

struct CSOptions {
    int max_iters = 2000;

    // Termination tolerance for the Cauchy-Simplex direction.
    double tol = 1e-9;

    // Algorithm 1 uses epsilon = 1e-10 as "Tolerance for the zero set".
    // See Chok--Vasil, Algorithm 1 and Section 3.2.
    double eps_zero = 1e-10;

    // The Cauchy-Simplex Algorithm 1 says: choose eta^t > 0, then eta^t <- min(eta^t, eta_max).
    // This is the user-chosen initial eta^t before capping.
    double initial_eta = 1.0;

    // Section 3.2 warns that taking eta = eta_max may incorrectly set an index to zero.
    // Therefore we use eta <- boundary_shrink * min(initial_eta, eta_max),
    // with boundary_shrink < 1.
    double boundary_shrink = 0.99;

    // Practical line search. Section 6 says the experiments use a line search; Lemma 10
    // states asymptotic convergence when eta is chosen by line search over [0, eta_max - eps].
    bool armijo = true;
    double armijo_c = 1e-4;
    double armijo_beta = 0.5;
    int max_line_search_steps = 30;
};

struct CSResult {
    Eigen::VectorXd lambda;
    double fval = 0.0;
    int iterations = 0;
    bool converged = false;
};

CSResult minimize_cauchy_simplex(
    KObjective& obj,
    const Eigen::VectorXd& lambda0,
    const CSOptions& opt = {}
);