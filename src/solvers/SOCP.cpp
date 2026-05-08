#include "SOCP.hpp"

#include "optimization.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace {

using Mat = Eigen::MatrixXd;
using Vec = Eigen::VectorXd;

int y_offset(int d, int ellipsoid_index) {
    return d + ellipsoid_index * d;
}

std::vector<Mat> cholesky_factors(const std::vector<Ellipsoid>& ellipsoids) {
    std::vector<Mat> factors;
    factors.reserve(ellipsoids.size());

    for (const auto& ellipsoid : ellipsoids) {
        Eigen::LLT<Mat> llt(ellipsoid.precision());
        if (llt.info() != Eigen::Success) {
            throw std::runtime_error("SOCP: ellipsoid precision is not SPD.");
        }
        factors.push_back(Mat(llt.matrixL().transpose()));
    }

    return factors;
}

Vec average_center(const std::vector<Ellipsoid>& ellipsoids, int d) {
    Vec center = Vec::Zero(d);
    for (const auto& ellipsoid : ellipsoids) {
        center.noalias() += ellipsoid.center();
    }
    center /= static_cast<double>(ellipsoids.size());
    return center;
}

} // namespace

SOCPResult solve_socp_alglib(const std::vector<Ellipsoid>& ellipsoids,
                             const SOCPOptions& opt) {
    if (ellipsoids.empty()) {
        return {};
    }

    const int n = static_cast<int>(ellipsoids.size());
    const int d = ellipsoids.front().dim();
    for (const auto& ellipsoid : ellipsoids) {
        if (ellipsoid.dim() != d) {
            throw std::invalid_argument("SOCP: all ellipsoids must have the same dimension.");
        }
    }

    const auto B = cholesky_factors(ellipsoids);
    const int eps_idx = d + n * d;
    const int num_vars = eps_idx + 1;
    const int num_equalities = n * d;

    alglib::minqpstate state;
    alglib::minqpreport report;
    alglib::real_1d_array x;

    alglib::minqpcreate(num_vars, state);

    alglib::real_1d_array linear;
    linear.setlength(num_vars);
    for (int j = 0; j < num_vars; ++j) {
        linear[j] = 0.0;
    }
    linear[eps_idx] = 1.0;
    alglib::minqpsetlinearterm(state, linear);

    alglib::real_1d_array lower;
    alglib::real_1d_array upper;
    lower.setlength(num_vars);
    upper.setlength(num_vars);
    for (int j = 0; j < num_vars; ++j) {
        lower[j] = alglib::fp_neginf;
        upper[j] = alglib::fp_posinf;
    }
    lower[eps_idx] = 0.0;
    alglib::minqpsetbc(state, lower, upper);

    alglib::real_2d_array lc;
    alglib::real_1d_array lc_lower;
    alglib::real_1d_array lc_upper;
    lc.setlength(num_equalities, num_vars);
    lc_lower.setlength(num_equalities);
    lc_upper.setlength(num_equalities);
    for (int r = 0; r < num_equalities; ++r) {
        for (int j = 0; j < num_vars; ++j) {
            lc[r][j] = 0.0;
        }
    }

    for (int i = 0; i < n; ++i) {
        const Vec bc = B[i] * ellipsoids[i].center();
        const int y0 = y_offset(d, i);
        for (int row = 0; row < d; ++row) {
            const int eq = i * d + row;
            for (int col = 0; col < d; ++col) {
                lc[eq][col] = -B[i](row, col);
            }
            lc[eq][y0 + row] = 1.0;
            lc_lower[eq] = -bc[row];
            lc_upper[eq] = -bc[row];
        }
        alglib::minqpaddsoccprimitive(state, y0, y0 + d, eps_idx, false);
    }
    alglib::minqpsetlc2dense(state, lc, lc_lower, lc_upper, num_equalities);

    const Vec m0 = average_center(ellipsoids, d);
    alglib::real_1d_array start;
    start.setlength(num_vars);
    for (int j = 0; j < num_vars; ++j) {
        start[j] = 0.0;
    }
    for (int j = 0; j < d; ++j) {
        start[j] = m0[j];
    }

    double eps0 = 0.0;
    for (int i = 0; i < n; ++i) {
        const Vec yi = B[i] * (m0 - ellipsoids[i].center());
        const int y0 = y_offset(d, i);
        for (int row = 0; row < d; ++row) {
            start[y0 + row] = yi[row];
        }
        eps0 = std::max(eps0, yi.norm());
    }
    start[eps_idx] = std::max(1.0, eps0 + 1.0);
    alglib::minqpsetstartingpoint(state, start);

    alglib::real_1d_array scale;
    scale.setlength(num_vars);
    for (int j = 0; j < num_vars; ++j) {
        scale[j] = 1.0;
    }
    alglib::minqpsetscale(state, scale);

    switch (opt.algorithm) {
        case SOCPAlgorithm::DenseGENIPM:
            alglib::minqpsetalgodensegenipm(state, opt.eps);
            break;
        case SOCPAlgorithm::SparseGENIPM:
            alglib::minqpsetalgosparsegenipm(state, opt.eps);
            break;
    }

    alglib::minqpoptimize(state);
    alglib::minqpresults(state, x, report);

    if (report.terminationtype <= 0 && opt.throw_on_failure) {
        throw std::runtime_error(
            "SOCP: ALGLIB minqp failed with termination type " +
            std::to_string(report.terminationtype));
    }

    SOCPResult result;
    result.m.resize(d);
    for (int j = 0; j < d; ++j) {
        result.m[j] = x[j];
    }

    result.dists.resize(n);
    for (int i = 0; i < n; ++i) {
        const Vec diff = result.m - ellipsoids[i].center();
        const double d2 = diff.transpose() * (ellipsoids[i].precision() * diff);
        result.dists[i] = std::sqrt(std::max(0.0, d2));
    }

    result.eps_star = result.dists.size() > 0 ? result.dists.maxCoeff() : 0.0;
    result.termination_type = static_cast<int>(report.terminationtype);
    result.inner_iterations = static_cast<int>(report.inneriterationscount);
    result.outer_iterations = static_cast<int>(report.outeriterationscount);
    return result;
}
