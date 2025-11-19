#pragma once
#include "KObjective.hpp"
#include "PGD.hpp"
#include "SLSQP.hpp"
#include "CauchySimplex.hpp"


struct EpsStar {
    double eps_star;
    Eigen::VectorXd lambda_star;
    Eigen::VectorXd dists; // per-ellipse distances at m(λ*)
};

enum class SolverKind { PGD, Cauchy, SLSQP };

EpsStar optimal_radius(KObjective& obj, SolverKind solver);