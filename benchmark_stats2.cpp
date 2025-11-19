#include "RandomEllipsoidGenerator.hpp"
#include "KFromEllipsoids.hpp"
#include "OptimalRadius.hpp"

#include "LPType.hpp"
#include "LPSeidel.hpp"
#include "LPClarkson.hpp"

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using Clock = std::chrono::high_resolution_clock;

template <class F>
double time_ms(F&& f) {
    const auto t0 = Clock::now();
    f();
    const auto t1 = Clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// Online mean / variance (Welford)
struct RunningStats {
    int n = 0;
    double mean = 0.0;
    double M2 = 0.0;

    void push(double x) {
        ++n;
        double delta = x - mean;
        mean += delta / static_cast<double>(n);
        double delta2 = x - mean;
        M2 += delta * delta2;
    }

    int count() const { return n; }

    double variance() const {
        return (n > 1) ? M2 / static_cast<double>(n - 1) : 0.0;
    }

    double stddev() const {
        return std::sqrt(variance());
    }
};

int main(int argc, char** argv) {
    std::cout.setf(std::ios::fixed);
    std::cout.precision(9);

    // Number of random instances per (n,d) per method.
    // You can override from the command line: ./prog 100
    int num_trials = 50;
    if (argc >= 2) {
        num_trials = std::stoi(argv[1]);
    }

    // Grid in (n,d)
    const int d_values[] = {2, 3, 4};
    const int n_values[] = {2, 3, 4};

    // Open CSV output
    const std::string filename = "benchmark_results.csv";
    std::ofstream ofs(filename);
    if (!ofs) {
        std::cerr << "Error: could not open " << filename << " for writing.\n";
        return 1;
    }

    // CSV header
    ofs << "d,n,method,mean_ms,std_ms,num_trials\n";

    // Sweep over d, n
    for (int d : d_values) {
        for (int n : n_values) {

            // Running stats for each of the 5 methods at this (n,d)
            RunningStats stats_raw_slsqp;
            RunningStats stats_raw_pgd;
            RunningStats stats_raw_cauchy;
            RunningStats stats_lp_seidel;
            RunningStats stats_lp_clarkson;

            // Base seed; perturbed by trial index to vary instances
            const unsigned long long base_seed = 12345ull
                                                 + 1000ull * static_cast<unsigned long long>(d)
                                                 + 10ull * static_cast<unsigned long long>(n);

            for (int trial = 0; trial < num_trials; ++trial) {
                // --- Generate random ellipsoids for this trial ---
                RandomEllipsoidGenerator::Options opt;
                opt.n = n;
                opt.d = d;
                opt.center_mode = RandomEllipsoidGenerator::CenterMode::UniformHypercube;
                opt.center_scale = 1.0;
                opt.spd_mode = RandomEllipsoidGenerator::SPDMode::LogUniformSpectrum;
                opt.lambda_min = 0.25;
                opt.lambda_max = 4.0;
                opt.store_covariance = false; // directly store precision if you prefer
                opt.radius = 1.;
                opt.seed = static_cast<unsigned long>(
                    base_seed + static_cast<unsigned long long>(trial)
                );

                RandomEllipsoidGenerator gen(opt);
                auto Es = gen.generate();

                // Build objective and LP oracle for this instance
                auto K = make_Kobjective_from_ellipsoids(1.0, Es);
                EllipsoidLPOracle O(Es, d, LPParams{SolverKind::SLSQP, 1e-8});
                std::vector<int> S(n);
                std::iota(S.begin(), S.end(), 0);

                // --- Raw: solve once on full set with three inner solvers ---

                {
                    double eps_star = 0.0;
                    double ms = time_ms([&]() {
                        auto res = optimal_radius(K, SolverKind::SLSQP);
                        eps_star = res.eps_star;
                    });
                    (void)eps_star; // eps_star is computed for sanity; unused here
                    stats_raw_slsqp.push(ms);
                }

                {
                    double eps_star = 0.0;
                    double ms = time_ms([&]() {
                        auto res = optimal_radius(K, SolverKind::PGD);
                        eps_star = res.eps_star;
                    });
                    (void)eps_star;
                    stats_raw_pgd.push(ms);
                }

                {
                    double eps_star = 0.0;
                    double ms = time_ms([&]() {
                        auto res = optimal_radius(K, SolverKind::Cauchy);
                        eps_star = res.eps_star;
                    });
                    (void)eps_star;
                    stats_raw_cauchy.push(ms);
                }

                // --- LP-type: Seidel + Clarkson (inner = SLSQP here) ---

                {
                    SeidelOptions so;
                    so.seed = 42;      // can also vary with trial if desired
                    so.max_depth = -1; // unlimited depth

                    SeidelResult out;
                    double ms = time_ms([&]() {
                        out = seidel_incremental(O, S, so);
                    });
                    (void)out; // could check out.basis.eps_star vs eps_star if desired
                    stats_lp_seidel.push(ms);
                }

                {
                    ClarksonOptions co;
                    co.rounds = 25;
                    co.seed = 123;

                    ClarksonResult out;
                    double ms = time_ms([&]() {
                        out = clarkson_iterative(O, S, co);
                    });
                    (void)out;
                    stats_lp_clarkson.push(ms);
                }
            }

            // Write one row per method for this (n,d)
            auto write_row = [&](const std::string& method, const RunningStats& st) {
                ofs << d << ","
                    << n << ","
                    << method << ","
                    << st.mean << ","
                    << st.stddev() << ","
                    << st.count() << "\n";
            };

            write_row("Raw-SLSQP",    stats_raw_slsqp);
            write_row("Raw-PGD",      stats_raw_pgd);
            write_row("Raw-Cauchy",   stats_raw_cauchy);
            write_row("LP-Seidel",    stats_lp_seidel);
            write_row("LP-Clarkson",  stats_lp_clarkson);
        }
    }

    ofs.close();
    std::cerr << "Wrote CSV to " << filename << "\n";
    return 0;
}
