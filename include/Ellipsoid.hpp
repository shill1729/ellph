#pragma once
#include <Eigen/Dense>
#include <optional>

class Ellipsoid {
public:
    using Vec = Eigen::VectorXd;
    using Mat = Eigen::MatrixXd;

    Ellipsoid() = default;

    // Construct with center and either covariance or precision.
    // Provide exactly one of (cov, prec); the other will be computed lazily.
    Ellipsoid(Vec center, std::optional<Mat> cov, std::optional<Mat> prec, double radius = 1.0);

    // Accessors
    const Vec& center() const noexcept { return center_; }
    double radius() const noexcept { return radius_; }

    // Guaranteed SPD (throws std::runtime_error if inversion fails, which shouldn't happen if SPD)
    const Mat& covariance() const;
    const Mat& precision()  const;

    // Dimension
    int dim() const noexcept { return static_cast<int>(center_.size()); }

private:
    Vec center_;
    mutable std::optional<Mat> cov_;   // Σ
    mutable std::optional<Mat> prec_;  // A^{-1}
    double radius_{1.0};
};