// ========================= include/solvers/CauchySimplex.hpp =========================
#pragma once

#include <Eigen/Dense>

#include "solvers/KObjective.hpp"

struct CSOptions {
    int max_iters = 2000;

    // Termination tolerance for the Cauchy-Simplex direction.
    double tol = 1e-9;
    double eps_zero = 1e-10;
    double initial_eta = 1.0;
    double boundary_shrink = 0.99;

    //Line search
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