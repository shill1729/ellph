#pragma once
#include "KObjective.hpp"

struct CSOptions {
    int max_iters = 4000;
    double tol = 1e-9;
    double eps_clip = 1e-12;   // threshold to zero-out tiny weights
    double eta_shrink = 1e-12; // use eta_max - eta_shrink as the upper bound
    bool renormalize = true;   // normalize sum to 1 after zero-clipping
    bool armijo = true;        // Armijo line-search inside [0, eta_max - eps]
    double armijo_beta = 0.5;
    double armijo_c = 1e-4;
};

struct CSResult {
    Eigen::VectorXd lambda; // w_T
    double fval;
    int iters;
    bool converged;
};

CSResult minimize_cauchy_simplex(KObjective& obj,
                                 const Eigen::VectorXd& w0,
                                 const CSOptions& opt);