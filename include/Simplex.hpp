#pragma once
#include <Eigen/Dense>

namespace Simplex {

// Project z onto the probability simplex Δ_k = {λ ≥ 0, 1^T λ = 1}.
Eigen::VectorXd project_to_simplex(const Eigen::VectorXd& z);

// Make a strictly interior start (optional).
Eigen::VectorXd uniform_start(int k) ;

// Backtracking Armijo on the simplex along a feasible direction dir with projection.
struct ArmijoParams { double alpha0=1.0, beta=0.5, c=1e-4, min_alpha=1e-12; };

}