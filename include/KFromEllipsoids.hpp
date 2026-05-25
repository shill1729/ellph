#pragma once
#include "geometry/Ellipsoid.hpp"
#include "solvers/KObjective.hpp"
#include <vector>

inline KObjective make_Kobjective_from_ellipsoids(
        double epsilon,
        const std::vector<Ellipsoid>& Es)
{
    std::vector<KObjective::Vec> xs; xs.reserve(Es.size());
    std::vector<KObjective::Mat> A; A.reserve(Es.size());
    for (const auto& E : Es) {
        xs.push_back(E.center());
        A.push_back(E.precision());
    }
    return KObjective(epsilon, xs, A);
}