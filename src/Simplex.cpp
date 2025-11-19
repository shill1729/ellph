#include "Simplex.hpp"
#include <algorithm>
#include <vector>

Eigen::VectorXd Simplex::project_to_simplex(const Eigen::VectorXd& z) {
    // Condat (2016)-style O(k log k)
    const int k = static_cast<int>(z.size());
    std::vector<double> u(k);
    for (int i = 0; i < k; ++i) u[i] = z[i];
    std::sort(u.begin(), u.end(), std::greater<double>());

    double css = 0.0;
    double theta = 0.0;
    for (int i = 0; i < k; ++i) {
        css += u[i];
        const double t = (css - 1.0) / (i + 1);
        if (i == k-1 || u[i+1] <= t) { theta = t; break; }
    }
    Eigen::VectorXd x = z.array() - theta;
    for (int i = 0; i < k; ++i) if (x[i] < 0.0) x[i] = 0.0;
    return x;
}

Eigen::VectorXd Simplex::uniform_start(int k) {
    return Eigen::VectorXd::Constant(k, 1.0 / double(k));
}