#pragma once
#include <Eigen/Dense>
#include <vector>

// K_epsilon(λ) = ε^2 - C(λ) on the probability simplex
// Data: centers x_i (d-vectors) and precision matrices A_i^{-1} (d×d, SPD).

class KObjective {
public:
    using Vec = Eigen::VectorXd;
    using Mat = Eigen::MatrixXd;

    KObjective(double epsilon,
               const std::vector<Vec>& centers,
               const std::vector<Mat>& precisions); // A_i^{-1}

    int k() const noexcept { return static_cast<int>(centers_.size()); }
    int d() const noexcept { return dim_; }

    // Evaluate K(λ), gradient g, and optionally Hessian H.
    // λ is size k, simplex-feasible (nonnegative, sum=1).
    // double value(const Vec& lambda);
    // double value_grad(const Vec& lambda, Vec& grad);
    // double value_grad_hess(const Vec& lambda, Vec& grad, Mat& hess);

    // ?
    double value(const Eigen::Ref<const Vec>& lambda);
    double value_grad(const Eigen::Ref<const Vec>& lambda, Eigen::Ref<Vec> grad);
    double value_grad_hess(const Eigen::Ref<const Vec>& lambda,
                        Eigen::Ref<Vec> grad,
                        Eigen::Ref<Mat> hess);

    // Accessors for downstream use (distances, m(λ))
    const Vec& centroid() const noexcept { return m_; }
    const Vec& mahalanobis_d2() const noexcept { return d2_; } // d_j^2 = (m-x_j)^T A_j^{-1} (m-x_j)

private:
    double eps_;
    int dim_;
    std::vector<Vec> centers_;
    std::vector<Mat> Ainv_;      // A_i^{-1}
    std::vector<double> q_;      // q_i = x_i^T A_i^{-1} x_i

    // Scratch (reused to avoid allocs)
    Mat S_;            // S(λ) = sum λ_i A_i^{-1}, SPD
    Eigen::LLT<Mat> lltS_;
    Vec mu_;           // mu(λ) = sum λ_i A_i^{-1} x_i
    Vec m_;            // centroid m(λ): solves S m = mu
    Vec Sm_;           // S*m == mu (cheap to keep)
    Vec d2_;           // per-index squared Mahalanobis to m(λ)

    void assemble_S_mu(const Vec& lambda); // builds S_, mu_, lltS_
    void solve_centroid();                  // m_ from S m = mu
    double C_value() const;                 // sum λ q_i - m^T S m (but S m = mu -> m^T mu)
    void distances_squared();               // fill d2_[j]
};