#include "RandomEllipsoidGenerator.hpp"
#include <Eigen/QR>
#include <Eigen/Cholesky>
#include <stdexcept>
#include <cmath>

RandomEllipsoidGenerator::RandomEllipsoidGenerator(Options opts)
    : opts_(std::move(opts)), rng_(opts_.seed)
{
    if (opts_.n <= 0 || opts_.d <= 0) {
        throw std::invalid_argument("RandomEllipsoidGenerator: n and d must be positive.");
    }
    if (opts_.spd_mode == SPDMode::LogUniformSpectrum) {
        if (!(opts_.lambda_min > 0.0 && opts_.lambda_max > opts_.lambda_min)) {
            throw std::invalid_argument("LogUniformSpectrum: require 0 < lambda_min < lambda_max.");
        }
    } else if (opts_.spd_mode == SPDMode::Wishart) {
        if (opts_.wishart_df == 0) opts_.wishart_df = opts_.d + 2;
        if (opts_.wishart_df < opts_.d) {
            throw std::invalid_argument("Wishart: df must be >= dimension.");
        }
    }
    if (opts_.radius <= 0.0) {
        throw std::invalid_argument("radius must be positive.");
    }
}

std::vector<Ellipsoid> RandomEllipsoidGenerator::generate() {
    std::vector<Ellipsoid> out;
    out.reserve(static_cast<size_t>(opts_.n));

    for (int i = 0; i < opts_.n; ++i) {
        Vec c = sample_center();

        // Build SPD as covariance by default; if user wants precision at storage time,
        // we invert once (dimension is usually moderate; for large d you could switch to lazy).
        Mat cov;
        if (opts_.spd_mode == SPDMode::LogUniformSpectrum) {
            cov = spd_from_loguniform_spectrum();
        } else {
            cov = spd_from_wishart();
        }

        if (opts_.store_covariance) {
            out.emplace_back(std::move(c), cov, std::nullopt, opts_.radius);
        } else {
            // store precision, not covariance
            Eigen::LLT<Mat> llt(cov);
            if (llt.info() != Eigen::Success) {
                throw std::runtime_error("Generated covariance is not SPD (LLT failed).");
            }
            Mat L = llt.matrixL();
            Mat Linv = L.inverse();
            Mat prec = Linv.transpose() * Linv; // Σ^{-1}
            out.emplace_back(std::move(c), std::nullopt, prec, opts_.radius);
        }
    }
    return out;
}

RandomEllipsoidGenerator::Vec RandomEllipsoidGenerator::sample_center() {
    Vec v(opts_.d);
    if (opts_.center_mode == CenterMode::UniformHypercube) {
        std::uniform_real_distribution<double> U(-opts_.center_scale, opts_.center_scale);
        for (int j = 0; j < opts_.d; ++j) v[j] = U(rng_);
    } else {
        std::normal_distribution<double> N(0.0, opts_.center_std);
        for (int j = 0; j < opts_.d; ++j) v[j] = N(rng_);
    }
    return v;
}

RandomEllipsoidGenerator::Mat RandomEllipsoidGenerator::random_orthonormal(int d) {
    Mat G = Mat::NullaryExpr(d, d, [this](){ std::normal_distribution<double> N(0.0, 1.0); return N(rng_); });
    Eigen::HouseholderQR<Mat> qr(G);
    Mat Q = qr.householderQ() * Mat::Identity(d, d);
    return Q; // columns orthonormal
}

RandomEllipsoidGenerator::Mat RandomEllipsoidGenerator::make_cov_from_spectrum(const Mat& Q, const Vec& evals) {
    // Ensure strictly positive evals
    if ((evals.array() <= 0.0).any()) {
        throw std::invalid_argument("Eigenvalues must be strictly positive.");
    }
    Mat D = evals.asDiagonal();
    return Q * D * Q.transpose();
}

RandomEllipsoidGenerator::Mat RandomEllipsoidGenerator::spd_from_loguniform_spectrum() {
    const int d = opts_.d;
    Mat Q = random_orthonormal(d);

    // sample λ ~ LogUniform([lambda_min, lambda_max])
    std::uniform_real_distribution<double> U(std::log(opts_.lambda_min), std::log(opts_.lambda_max));
    Vec evals(d);
    for (int i = 0; i < d; ++i) evals[i] = std::exp(U(rng_));
    // cov = Q diag(λ) Q^T
    return make_cov_from_spectrum(Q, evals);
}

RandomEllipsoidGenerator::Mat RandomEllipsoidGenerator::spd_from_wishart() {
    // Wishart W_d(df, I): cov = (1/df) * G G^T with G ~ N(0,1)^{d x df}
    // Scaling by 1/df keeps eigenvalues centered near 1 as df grows.
    const int d = opts_.d;
    const int df = opts_.wishart_df;
    Mat G = Mat::NullaryExpr(d, df, [this](){ std::normal_distribution<double> N(0.0, 1.0); return N(rng_); });
    Mat S = (G * G.transpose()) / static_cast<double>(df);

    // Small regularization for numerical stability (optional; comment if you prefer pure Wishart)
    // S += 1e-12 * Mat::Identity(d, d);

    Eigen::LLT<Mat> llt(S);
    if (llt.info() != Eigen::Success) {
        // In the extremely rare case numerical SPD is lost, nudge the diagonal
        Mat I = Mat::Identity(d, d);
        double eps = 1e-10;
        Mat Sreg = S + eps * I;
        Eigen::LLT<Mat> llt2(Sreg);
        if (llt2.info() != Eigen::Success) {
            throw std::runtime_error("Wishart draw failed to be SPD even after regularization.");
        }
        return Sreg;
    }
    return S;
}