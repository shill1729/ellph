#pragma once
#include "LPType.hpp"

struct ClarksonOptions {
    int rounds = 20;          // outer rounds
    int sample_size = -1;     // if <0, default to 4*(d+1)*(d+1)
    double weight_bad_threshold = 0.5; // if violators carry >50% sampled weight, double
    uint64_t seed = 123;
};

struct ClarksonResult {
    LPBasis basis;
    int violation_tests = 0;
    int doublings = 0;
};

ClarksonResult clarkson_iterative(const EllipsoidLPOracle& oracle,
                                  const std::vector<int>& S,
                                  ClarksonOptions opt = {});