// Copyright (c) 2023 Matthias Heinz
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#include "PerturbationTheory.hh"

double GetSecondOrderCorrection(Operator& H_0, Operator& H_1, Operator& H_2) {
  double Emp2 = 0;
  int nparticles = H_0.modelspace->particles.size();
  std::vector<index_t> particles_vec(
      H_0.modelspace->particles.begin(),
      H_0.modelspace->particles
          .end());  // convert set to vector for OMP looping
                    //   for ( auto& i : modelspace->particles)
                    //   #pragma omp parallel for reduction(+:Emp2)
  for (int ii = 0; ii < nparticles; ++ii) {
    //     std::cout << " i = " << i << std::endl;
    //     index_t i = modelspace->particles[ii];
    index_t i = particles_vec[ii];
    double ei = H_0.OneBody(i, i);
    Orbit& oi = H_0.modelspace->GetOrbit(i);
    for (auto& a : H_0.modelspace->holes) {
      Orbit& oa = H_0.modelspace->GetOrbit(a);
      double ea = H_0.OneBody(a, a);
      Emp2 += (oa.j2 + 1) * oa.occ * H_1.OneBody(i, a) * H_2.OneBody(i, a) /
              (ea - ei);
      for (index_t j : H_0.modelspace->particles) {
        if (j < i) continue;
        double ej = H_0.OneBody(j, j);
        Orbit& oj = H_0.modelspace->GetOrbit(j);
        for (auto& b : H_0.modelspace->holes) {
          if (b < a) continue;
          Orbit& ob = H_0.modelspace->GetOrbit(b);
          if ((oi.l + oj.l + oa.l + ob.l) % 2 > 0) continue;
          if ((oi.tz2 + oj.tz2) != (oa.tz2 + ob.tz2)) continue;
          double eb = H_0.OneBody(b, b);
          double denom = ea + eb - ei - ej;
          int Jmin =
              std::max(std::abs(oi.j2 - oj.j2), std::abs(oa.j2 - ob.j2)) / 2;
          int Jmax = std::min(oi.j2 + oj.j2, oa.j2 + ob.j2) / 2;
          int dJ = 1;
          if ((a == b) || (i == j)) {
            Jmin += Jmin % 2;
            dJ = 2;
          }
          for (int J = Jmin; J <= Jmax; J += dJ) {
            double tbme_h1 = H_1.TwoBody.GetTBME_J_norm(J, a, b, i, j);
            double tbme_h2 = H_2.TwoBody.GetTBME_J_norm(J, a, b, i, j);
            // no factor 1/4 because of the restricted sum
            Emp2 += (2 * J + 1) * oa.occ * ob.occ * tbme_h1 * tbme_h2 / denom;
          }
        }
      }
    }
  }
  return Emp2;
}
