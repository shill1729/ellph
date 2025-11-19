#pragma once
#include "KObjective.hpp"
#include <nlopt.hpp>

struct NloptOptions {
    int max_evals = 2000;
    double rel_tol = 1e-8;
    double abs_tol = 1e-10;
};

struct NloptResult {
    Eigen::VectorXd lambda;
    double fval;
    nlopt::result status;
};

NloptResult minimize_slsqp(KObjective& obj, const Eigen::VectorXd& lambda0, const NloptOptions& opt);