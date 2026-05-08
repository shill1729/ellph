// LPSeidel.hpp
#pragma once
#include "LPType.hpp"

struct SeidelOptions {
    uint64_t seed = 42;
    int max_depth = -1;     // safety
};

struct SeidelResult {
    LPBasis basis;
    int violation_tests = 0;
};

SeidelResult seidel_incremental(const EllipsoidLPOracle& oracle,
                                const std::vector<int>& S,
                                SeidelOptions opt = {});