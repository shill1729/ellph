#pragma once
#include "Ellipsoid.hpp"
#include "KObjective.hpp"
#include <vector>

inline KObjective make_Kobjective_from_ellipsoids(
        double epsilon,
        const std::vector<Ellipsoid>& Es)
{
    std::vector<KObjective::Vec> xs; xs.reserve(Es.size());
    std::vector<KObjective::Mat> Ainv; Ainv.reserve(Es.size());
    for (const auto& E : Es) {
        xs.push_back(E.center());
        Ainv.push_back(E.precision()); // you guaranteed SPD precision
    }
    return KObjective(epsilon, xs, Ainv);
}