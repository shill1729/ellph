#include "RandomEllipsoidGenerator.hpp"
#include "KFromEllipsoids.hpp"
#include "OptimalRadius.hpp"
#include "solvers/SOCP.hpp"

#include "include/solvers/LPType.hpp"
#include "include/solvers/LPSeidel.hpp"
#include "include/solvers/LPClarkson.hpp"

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

struct MethodStats {
    RunningStats time_ms;
    RunningStats radius;

    void push(double ms, double eps_star) {
        time_ms.push(ms);
        radius.push(eps_star);
    }
};

double radius_at(const std::vector<Ellipsoid>& Es, const Eigen::VectorXd& m) {
    double radius = 0.0;
    for (const auto& E : Es) {
        const Eigen::VectorXd diff = m - E.center();
        const double d2 = diff.transpose() * (E.precision() * diff);
        radius = std::max(radius, std::sqrt(std::max(0.0, d2)));
    }
    return radius;
}

double full_radius_for_basis(const EllipsoidLPOracle& O,
                             const std::vector<int>& basis,
                             const std::vector<Ellipsoid>& Es) {
    const LPEval ev = O.evaluate(basis);
    return radius_at(Es, ev.m);
}

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
    // TODO: make these passable arguments?
    const int d_values[] = {2, 3, 4, 5, 10, 20, 30, 50};
    const int n_values[] = {2, 3, 4, 5, 10, 15};

    // Open CSV output
    const std::string filename = "build/benchmark_results.csv";
    // std::filesystem::create_directories("build");
    std::ofstream ofs(filename);
    if (!ofs) {
        std::cerr << "Error: could not open " << filename << " for writing.\n";
        return 1;
    }

    // CSV header
    ofs << "d,n,method,mean_ms,std_ms,moe,mean_radius,num_trials\n";

    // Sweep over d, n
    for (int d : d_values) {
        for (int n : n_values) {
            std::cout << "Running trial for (n,d) = (" << n << "," << d << ")\n";
            // Running stats for each method at this (n,d)
            MethodStats stats_raw_slsqp;
            MethodStats stats_raw_pgd;
            MethodStats stats_raw_cauchy;
            MethodStats stats_raw_socp;
            MethodStats stats_lp_seidel;
            MethodStats stats_lp_clarkson;
            MethodStats stats_lp_seidel_socp;
            MethodStats stats_lp_clarkson_socp;
        
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
                EllipsoidLPOracle O(Es, d, LPParams{BasisSolverKind::DualSLSQP, 1e-8});
                EllipsoidLPOracle O_socp(Es, d, LPParams{BasisSolverKind::PrimalSOCP, 1e-5});
                std::vector<int> S(n);
                std::iota(S.begin(), S.end(), 0);

                // --- Raw: solve once on full set ---

                {
                    double eps_star = 0.0;
                    bool ok = true;
                    double ms = time_ms([&]() {
                        try {
                            auto res = optimal_radius(K, SolverKind::SLSQP);
                            eps_star = res.eps_star;
                        } catch (const nlopt::roundoff_limited&) {
                            ok = false;
                            std::cerr << "[d=" << d << " n=" << n << " trial=" << trial
                                      << " Raw-SLSQP] roundoff_limited\n";
                        }
                    });
                    if (ok) stats_raw_slsqp.push(ms, eps_star);
                }

                {
                    double eps_star = 0.0;
                    double ms = time_ms([&]() {
                        auto res = optimal_radius(K, SolverKind::PGD);
                        eps_star = res.eps_star;
                    });
                    stats_raw_pgd.push(ms, eps_star);
                }

                {
                    double eps_star = 0.0;
                    double ms = time_ms([&]() {
                        auto res = optimal_radius(K, SolverKind::Cauchy);
                        eps_star = res.eps_star;
                    });
                    stats_raw_cauchy.push(ms, eps_star);
                }

                {
                    double eps_star = 0.0;
                    bool ok = true;
                    double ms = time_ms([&]() {
                        try {
                            auto res = solve_socp_alglib(Es);
                            eps_star = res.eps_star;
                        } catch (const std::exception& e) {
                            ok = false;
                            std::cerr << "SOCP failed: " << e.what() << "\n";
                        }
                    });
                    if (ok) stats_raw_socp.push(ms, eps_star);
                }

                // --- LP-type: Seidel + Clarkson with dual and primal basis solvers ---

                {
                    SeidelOptions so;
                    so.seed = 42;
                    so.max_depth = -1;

                    SeidelResult out;
                    bool ok = true;
                    double ms = time_ms([&]() {
                        try {
                            out = seidel_incremental(O, S, so);
                        } catch (const nlopt::roundoff_limited&) {
                            ok = false;
                            std::cerr << "[d=" << d << " n=" << n << " trial=" << trial
                                      << " LP-Seidel] roundoff_limited\n";
                        }
                    });
                    if (ok) stats_lp_seidel.push(ms, full_radius_for_basis(O, out.basis.idx, Es));
                }

                {
                    SeidelOptions so;
                    so.seed = 42;
                    so.max_depth = -1;

                    SeidelResult out;
                    bool ok = true;
                    double ms = time_ms([&]() {
                        try {
                            out = seidel_incremental(O_socp, S, so);
                        } catch (const std::exception& e) {
                            ok = false;
                            std::cerr << "LP-Seidel-SOCP failed: " << e.what() << "\n";
                        }
                    });
                    if (ok) {
                        stats_lp_seidel_socp.push(ms, full_radius_for_basis(O_socp, out.basis.idx, Es));
                    }
                }

                {
                    ClarksonOptions co;
                    co.rounds = 25;
                    co.seed = 123;

                    ClarksonResult out;
                    bool ok = true;
                    double ms = time_ms([&]() {
                        
                        try {
                            out = clarkson_iterative(O, S, co);
                        } catch (const nlopt::roundoff_limited&) {
                            ok = false;
                            std::cerr << "[d=" << d << " n=" << n << " trial=" << trial
                                      << " LP-Clarkson] roundoff_limited\n";
                        }
                    });
                    if (ok) stats_lp_clarkson.push(ms, full_radius_for_basis(O, out.basis.idx, Es));
                }

                {
                    ClarksonOptions co;
                    co.rounds = 25;
                    co.seed = 123;

                    ClarksonResult out;
                    bool ok = true;
                    double ms = time_ms([&]() {
                        try {
                            out = clarkson_iterative(O_socp, S, co);
                        } catch (const std::exception& e) {
                            ok = false;
                            std::cerr << "LP-Clarkson-SOCP failed: " << e.what() << "\n";
                        }
                    });
                    if (ok) {
                        stats_lp_clarkson_socp.push(ms, full_radius_for_basis(O_socp, out.basis.idx, Es));
                    }
                }
            }

            // Write one row per method for this (n,d)
            auto write_row = [&](const std::string& method, const MethodStats& st) {
                ofs << d << ","
                    << n << ","
                    << method << ","
                    << st.time_ms.mean << ","
                    << st.time_ms.stddev() << ","
                    << 1.959964 * st.time_ms.stddev() / std::sqrt(st.time_ms.count()) << ","
                    << st.radius.mean << ","
                    << st.time_ms.count() << "\n";
            };

            write_row("Raw-SLSQP",    stats_raw_slsqp);
            write_row("Raw-PGD",      stats_raw_pgd);
            write_row("Raw-Cauchy",   stats_raw_cauchy);
            write_row("Raw-SOCP",     stats_raw_socp);
            write_row("LP-Seidel",    stats_lp_seidel);
            write_row("LP-Clarkson",  stats_lp_clarkson);
            write_row("LP-Seidel-SOCP",   stats_lp_seidel_socp);
            write_row("LP-Clarkson-SOCP", stats_lp_clarkson_socp);

        }
    }

    ofs.close();
    std::cerr << "Wrote CSV to " << filename << "\n";
    return 0;
}
