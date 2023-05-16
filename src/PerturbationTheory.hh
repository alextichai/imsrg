// Copyright (c) 2023 Matthias Heinz
// 
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef PERTURBATIONTHEORY_H_
#define PERTURBATIONTHEORY_H_

#include "Operator.hh"

// Get correction for <H_1 H_2> using H_0 as unperturbed Hamiltonian.
double GetSecondOrderCorrection(
    Operator& H_0,
    Operator& H_1,
    Operator& H_2
);



#endif  // PERTURBATIONTHEORY_H_
