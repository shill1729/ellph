#pragma once

#include "geometry/Ellipsoid.hpp"
#include <Eigen/Dense>
#include <vector>

enum class SOCPAlgorithm {
    DenseGENIPM,
    SparseGENIPM
};

struct SOCPOptions {
    double eps = 1e-8;
    SOCPAlgorithm algorithm = SOCPAlgorithm::DenseGENIPM;
    bool throw_on_failure = true;
};

struct SOCPResult {
    double eps_star = 0.0;
    Eigen::VectorXd m;
    Eigen::VectorXd dists;
    int termination_type = 0;
    int inner_iterations = 0;
    int outer_iterations = 0;
};

SOCPResult solve_socp_alglib(const std::vector<Ellipsoid>& ellipsoids,
                             const SOCPOptions& opt = {});
