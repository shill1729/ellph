#include "Ellipsoid.hpp"
#include <stdexcept>

Ellipsoid::Ellipsoid(Vec center, std::optional<Mat> cov, std::optional<Mat> prec, double radius)
    : center_(std::move(center)), cov_(std::move(cov)), prec_(std::move(prec)), radius_(radius)
{
    if (!cov_ && !prec_) {
        throw std::invalid_argument("Ellipsoid: need covariance or precision.");
    }
    if (cov_ && prec_) {
        // Optional: verify consistency (skip for speed)
    }
    if (radius_ <= 0.0) {
        throw std::invalid_argument("Ellipsoid: radius must be positive.");
    }
}

const Ellipsoid::Mat& Ellipsoid::covariance() const {
    if (cov_) return *cov_;
    // Compute Σ = (A^{-1})^{-1} by robust Cholesky
    Eigen::LLT<Mat> llt(*prec_);
    if (llt.info() != Eigen::Success) {
        throw std::runtime_error("Ellipsoid: precision not SPD (LLT failed).");
    }
    // Invert via Cholesky: inv(A) = L^{-T} L^{-1}
    Mat L = llt.matrixL();
    Mat Linv = L.inverse();   // For d<=~100 this is fine; for larger d prefer triangular solves on demand
    cov_.emplace(Linv.transpose() * Linv);
    return *cov_;
}

const Ellipsoid::Mat& Ellipsoid::precision() const {
    if (prec_) return *prec_;
    Eigen::LLT<Mat> llt(*cov_);
    if (llt.info() != Eigen::Success) {
        throw std::runtime_error("Ellipsoid: covariance not SPD (LLT failed).");
    }
    Mat L = llt.matrixL();
    Mat Linv = L.inverse();
    prec_.emplace(Linv.transpose() * Linv);
    return *prec_;
}