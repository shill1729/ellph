#pragma once
#include "KObjective.hpp"

struct PGDOptions {
    int max_iters = 500;
    double tol = 1e-8;
    double step0 = 1.0;
    double armijo_beta = 0.5;
    double armijo_c = 1e-4;
    bool use_hessian_safeguard = false; // optional
};

struct PGDResult {
    Eigen::VectorXd lambda;
    double fval;
    int iters;
    bool converged;
};

PGDResult minimize_pgd(KObjective& obj, const Eigen::VectorXd& lambda0, const PGDOptions& opt);