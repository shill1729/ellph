#include "KObjective.hpp"
#include <stdexcept>

KObjective::KObjective(double epsilon,
                       const std::vector<Vec>& centers,
                       const std::vector<Mat>& precisions)
: eps_(epsilon), dim_(0), centers_(centers), Ainv_(precisions)
{
    const int k = static_cast<int>(centers_.size());
    if (k == 0 || centers_.size() != Ainv_.size())
        throw std::invalid_argument("centers and precisions must be nonempty and same length.");

    dim_ = static_cast<int>(centers_[0].size());
    for (int i = 0; i < k; ++i) {
        if (centers_[i].size() != dim_ || Ainv_[i].rows() != dim_ || Ainv_[i].cols() != dim_)
            throw std::invalid_argument("dimension mismatch in centers/precisions.");
    }

    q_.resize(k);
    for (int i = 0; i < k; ++i) {
        q_[i] = centers_[i].transpose() * (Ainv_[i] * centers_[i]);
    }

    S_.resize(dim_, dim_);
    mu_.resize(dim_);
    m_.resize(dim_);
    Sm_.resize(dim_);
    d2_.setZero(k);
}

void KObjective::assemble_S_mu(const Vec& lambda) {
    const int k = static_cast<int>(centers_.size());
    S_.setZero();
    mu_.setZero();
    for (int i = 0; i < k; ++i) {
        const double w = lambda[i];
        if (w == 0.0) continue;
        S_.noalias() += w * Ainv_[i];
        mu_.noalias() += w * (Ainv_[i] * centers_[i]);
    }
    lltS_.compute(S_);
    if (lltS_.info() != Eigen::Success) {
        throw std::runtime_error("LLT failed: S(λ) must be SPD.");
    }
    // Sm_ = S*m = mu_; but m unknown yet
    Sm_ = mu_;
}

void KObjective::solve_centroid() {
    // Solve S m = mu via LLT
    m_ = lltS_.solve(mu_);
    if (lltS_.info() != Eigen::Success) {
        throw std::runtime_error("LLT solve failed for centroid.");
    }
}

double KObjective::C_value() const {
    // C(λ) = sum λ q_i - m^T S m ; but S m = mu => m^T S m = m^T mu
    // We don't have λ here; caller should accumulate sum λ q_i externally if needed.
    // Provide only the second term contribution:
    return 0.0; // not used directly in this form
}

void KObjective::distances_squared() {
    const int k = static_cast<int>(centers_.size());
    for (int j = 0; j < k; ++j) {
        const Vec diff = m_ - centers_[j];
        d2_[j] = diff.transpose() * (Ainv_[j] * diff);
    }
}

double KObjective::value(const Eigen::Ref<const Vec>& lambda) {
    assemble_S_mu(lambda);
    solve_centroid();
    double sum_lq = 0.0;
    for (int i = 0; i < lambda.size(); ++i) sum_lq += lambda[i] * q_[i];
    const double mSm = m_.dot(Sm_); // == m^T mu
    const double C = sum_lq - mSm;
    return eps_*eps_ - C;
}

double KObjective::value_grad(const Eigen::Ref<const Vec>& lambda,
                              Eigen::Ref<Vec> grad) {
    const double val = value(lambda);
    distances_squared();
    // NO RESIZE on Ref:
    if (grad.size() != lambda.size())
        throw std::invalid_argument("value_grad: grad has wrong size");
    for (int j = 0; j < grad.size(); ++j) grad[j] = -d2_[j];
    return val;
}

double KObjective::value_grad_hess(const Eigen::Ref<const Vec>& lambda,
                                   Eigen::Ref<Vec> grad,
                                   Eigen::Ref<Mat> hess) {
    const int k = static_cast<int>(lambda.size());
    if (grad.size() != k)  throw std::invalid_argument("value_grad_hess: grad wrong size");
    if (hess.rows() != k || hess.cols() != k)
        throw std::invalid_argument("value_grad_hess: hess wrong shape");

    const double val = value_grad(lambda, grad); // will fill grad

    // Build Hessian into provided matrix (no resize)
    std::vector<Vec> y(k, Vec::Zero(d()));
    for (int j = 0; j < k; ++j) {
        const Vec rhs = Ainv_[j] * (m_ - centers_[j]);
        y[j] = lltS_.solve(rhs);
        if (lltS_.info() != Eigen::Success)
            throw std::runtime_error("LLT solve failed in Hessian.");
    }
    for (int i = 0; i < k; ++i) {
        const Vec left = Ainv_[i] * (m_ - centers_[i]);
        for (int j = 0; j < k; ++j)
            hess(i,j) = 2.0 * left.dot(y[j]);
    }
    return val;
}