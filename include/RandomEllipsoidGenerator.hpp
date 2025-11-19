#pragma once
#include "Ellipsoid.hpp"
#include <random>
#include <vector>

class RandomEllipsoidGenerator {
public:
    enum class CenterMode { UniformHypercube, Gaussian };
    enum class SPDMode    { LogUniformSpectrum, Wishart };

    struct Options {
        int n = 10;                    // number of ellipsoids
        int d = 2;                     // dimension
        CenterMode center_mode = CenterMode::UniformHypercube;
        double center_scale = 1.0;     // for Uniform: sample in [-center_scale, center_scale]^d
        double center_std   = 1.0;     // for Gaussian: N(0, center_std^2 I)

        SPDMode spd_mode = SPDMode::LogUniformSpectrum;

        // LogUniformSpectrum parameters: eigenvalues ~ logU([lambda_min, lambda_max])
        double lambda_min = 0.25;
        double lambda_max = 4.0;

        // Wishart parameters: W_d(df, S). Here S = I (scale), df >= d
        int wishart_df = 0; // if 0, defaults to d + 2

        // Whether to store Σ (covariance) or A^{-1} (precision) at construction
        bool store_covariance = true;

        // Radius parameter for the geometric ellipsoid { x : (x-c)^T A (x-c) <= radius^2 }
        // If store_covariance==true, A = Σ^{-1}; if false, A is the provided precision itself.
        double radius = 1.0;

        // RNG seed
        uint64_t seed = 42ULL;
    };

    explicit RandomEllipsoidGenerator(Options opts);

    // Main API
    std::vector<Ellipsoid> generate();

private:
    using Vec = Eigen::VectorXd;
    using Mat = Eigen::MatrixXd;

    Options opts_;
    std::mt19937_64 rng_;

    // Centers
    Vec sample_center();
    // Orthonormal Q via QR of Gaussian matrix
    Mat random_orthonormal(int d);
    // SPD construction
    Mat spd_from_loguniform_spectrum();
    Mat spd_from_wishart();

    // Helpers
    static Mat make_cov_from_spectrum(const Mat& Q, const Vec& evals);
};