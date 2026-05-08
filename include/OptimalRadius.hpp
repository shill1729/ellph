#pragma once
#include "solvers/KObjective.hpp"
#include "solvers/PGD.hpp"
#include "solvers/SLSQP.hpp"
#include "solvers/CauchySimplex.hpp"


struct EpsStar {
    double eps_star;
    Eigen::VectorXd lambda_star;
    Eigen::VectorXd dists; // per-ellipse distances at m(λ*)
};

enum class SolverKind { PGD, Cauchy, SLSQP };

EpsStar optimal_radius(KObjective& obj, SolverKind solver);