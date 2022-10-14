///////////////////////////////////////////////////////////////////////////////////
//    Commutator.hh, part of  imsrg++
//    Copyright (C) 2018  Ragnar Stroberg
//
//    This program is free software; you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation; either version 2 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License along
//    with this program; if not, write to the Free Software Foundation, Inc.,
//    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
///////////////////////////////////////////////////////////////////////////////////

#include "Commutator331.hh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "ModelSpace.hh"

#include <omp.h>

template <typename T> void Print(std::string prefix, const T &val) {
  std::cout << prefix << ": " << val << "\n";
}

template <typename T>
void Print(std::string prefix, const T &val, const T &val2) {
  std::cout << prefix << ": " << val << "," << val2 << "\n";
}

namespace comm331 {

void comm331ss_expand_impl(const Operator &X, const Operator &Y, Operator &Z) {
  std::cout << "In comm331_expand\n";
  double tstart = omp_get_wtime();
  Z.modelspace->PreCalculateSixJ();

  int hX = 1;
  if (X.IsAntiHermitian())
    hX = -1;
  int hY = 1;
  if (Y.IsAntiHermitian())
    hY = -1;

  const int emax_3body = Z.modelspace->GetEMax3Body();
  const int e3max = Z.modelspace->GetE3max();
  const int jj1max = internal::ExtractJJ1Max(Z);

  // Case 1: AB=HH, CDE=PPP
  {
    std::size_t num_chans = 0;
    for (std::size_t i_ch_3b = 0;
         i_ch_3b < Y.modelspace->GetNumberThreeBodyChannels(); i_ch_3b += 1) {
      const ThreeBodyChannel &ch_3b =
          Z.modelspace->GetThreeBodyChannel(i_ch_3b);
      const std::vector<std::size_t> chans_2b =
          internal::Extract2BChannelsValidIn3BChannel(jj1max, i_ch_3b, Z);

      for (const auto &i_ch_2b_ab : chans_2b) {
        for (const auto &i_ch_2b_cd : chans_2b) {
          const TwoBodyChannel &ch_2b_ab =
              Z.modelspace->GetTwoBodyChannel(i_ch_2b_ab);
          const TwoBodyChannel &ch_2b_cd =
              Z.modelspace->GetTwoBodyChannel(i_ch_2b_cd);
          const internal::TwoBodyBasis basis_2b_ab_hh =
              internal::TwoBodyBasis::PQInTwoBodyChannelWithE3Max_HH(i_ch_2b_ab,
                                                                     Z, e3max);
          const internal::TwoBodyBasis basis_2b_cd_pp =
              internal::TwoBodyBasis::PQInTwoBodyChannelWithE3Max_PP(i_ch_2b_cd,
                                                                     Z, e3max);

          // final contracted index e constrained by being in state | (cd) J_cd
          // e>
          const int tz2_e = ch_3b.twoTz - 2 * ch_2b_cd.Tz;
          const int parity_e = (ch_3b.parity + ch_2b_cd.parity) % 2;
          const int jj_min_e = std::abs(ch_3b.twoJ - ch_2b_cd.J * 2);
          const int jj_max_e = std::min(ch_3b.twoJ + ch_2b_cd.J * 2, jj1max);
          const internal::OneBodyBasis basis_e_p =
              internal::OneBodyBasis::FromQuantumNumbers_P(
                  Z, jj_min_e, jj_max_e, parity_e, tz2_e);
          const internal::ThreeBodyBasis basis_cde =
              internal::ThreeBodyBasis::MinFrom2BAnd1BBasis(
                  i_ch_3b, i_ch_2b_cd, Z, basis_2b_cd_pp, basis_e_p, e3max);

          // exterenal indices alpha constrained by being in state | (ab) J_ab
          // alpha>
          const int tz2_alpha = ch_3b.twoTz - 2 * ch_2b_ab.Tz;
          const int parity_alpha = (ch_3b.parity + ch_2b_ab.parity) % 2;
          const int jj_min_alpha = std::abs(ch_3b.twoJ - ch_2b_ab.J * 2);
          const int jj_max_alpha =
              std::min(ch_3b.twoJ + ch_2b_ab.J * 2, jj1max);
          const internal::OneBodyBasis basis_1b_alpha =
              internal::OneBodyBasis::FromQuantumNumbers(
                  Z, jj_min_alpha, jj_max_alpha, parity_alpha, tz2_alpha);
          const internal::ThreeBodyBasis basis_abalpha =
              internal::ThreeBodyBasis::From2BAnd1BBasis(i_ch_3b, i_ch_2b_ab, Z,
                                                         basis_2b_ab_hh,
                                                         basis_1b_alpha, e3max);

          if ((basis_2b_ab_hh.BasisSize() == 0) ||
              (basis_2b_cd_pp.BasisSize() == 0) ||
              (basis_e_p.BasisSize() == 0) ||
              (basis_1b_alpha.BasisSize() == 0) ||
              (basis_cde.BasisSize() == 0) ||
              (basis_abalpha.BasisSize() == 0)) {
            continue;
          }

          num_chans += 1;

          Print("JJ_3B", ch_3b.twoJ);
          Print("P_3B", ch_3b.parity);
          Print("TTz_3B", ch_3b.twoTz);
          Print("JJ_2B_AB", ch_2b_ab.J * 2);
          Print("P_2B_AB", ch_2b_ab.parity);
          Print("TTz_2B_AB", ch_2b_ab.Tz * 2);
          Print("JJ_2B_CD", ch_2b_cd.J * 2);
          Print("P_2b_CD", ch_2b_cd.parity);
          Print("TTz_2b_CD", ch_2b_cd.Tz * 2);

          Print("DIM_AB_HH", basis_2b_ab_hh.BasisSize());
          Print("DIM_CD_PP", basis_2b_cd_pp.BasisSize());
          Print("DIM_E_P", basis_e_p.BasisSize());
          Print("DIM_I/J", basis_1b_alpha.BasisSize());
          Print("DIM_CDE_PPP", basis_cde.BasisSize());
          Print("DIM_ABALPHA_HHX", basis_abalpha.BasisSize());
        }
      }
    }
    Print("NUM_CHANS_HHPPP", num_chans);
  }

  Z.profiler.timer[__func__] += omp_get_wtime() - tstart;
}

namespace internal {

std::size_t ExtractWrapFactor(const Operator &Z) {
  // We don't actually need all orbits for 331, but this is safer for lookups
  // The memory cost is very low (~100 MB per 3B lookup table at emax=14)
  const auto &sp_indices = Z.modelspace->all_orbits;
  std::size_t wrap_factor = 0;
  for (const auto &p : sp_indices) {
    wrap_factor = std::max(wrap_factor, static_cast<std::size_t>(p));
  }
  return wrap_factor + 1;
}

int ExtractJJ1Max(const Operator &Z) {
  int jj1max = 1;
  for (const std::size_t &p : Z.modelspace->orbits_3body_space_) {
    const Orbit op = Z.modelspace->GetOrbit(p);
    jj1max = std::max(op.j2, jj1max);
  }
  return jj1max;
}

std::vector<std::size_t> Extract2BChannelsValidIn3BChannel(int jj1max,
                                                           std::size_t i_ch_3b,
                                                           const Operator &Z) {
  const ThreeBodyChannel &ch_3b = Z.modelspace->GetThreeBodyChannel(i_ch_3b);
  std::vector<std::size_t> valid_ch_2;
  for (std::size_t i_ch_2b = 0;
       i_ch_2b < Z.modelspace->GetNumberTwoBodyChannels(); i_ch_2b += 1) {
    const TwoBodyChannel &ch_2b = Z.modelspace->GetTwoBodyChannel(i_ch_2b);

    if ((ch_2b.J * 2 - jj1max <= ch_3b.twoJ) &&
        (ch_2b.J * 2 + jj1max >= ch_3b.twoJ)) {
      if (std::abs(ch_3b.twoTz - ch_2b.Tz * 2) == 1) {
        valid_ch_2.push_back((i_ch_2b));
      }
    }
  }

  return valid_ch_2;
}

static std::vector<std::size_t>
GetLookupIndices(const std::vector<std::size_t> &states,
                 std::size_t lookup_size) {
  std::vector<std::size_t> indices(lookup_size, 0);
  for (std::size_t i_p = 0; i_p < states.size(); i_p += 1) {
    const std::size_t p = states[i_p];
    indices[p] = i_p;
  }
  return indices;
}

static std::vector<int>
GetLookupValidities(const std::vector<std::size_t> &states,
                    std::size_t lookup_size) {
  std::vector<int> validities(lookup_size, 0);
  for (std::size_t i_p = 0; i_p < states.size(); i_p += 1) {
    const std::size_t p = states[i_p];
    validities[p] = 1;
  }
  return validities;
}

std::vector<std::size_t> Get1BPIndices(const std::vector<std::size_t> &p_states,
                                       std::size_t wrap_factor) {
  return GetLookupIndices(p_states, wrap_factor);
}

std::vector<int> Get1BPValidities(const std::vector<std::size_t> &p_states,
                                  std::size_t wrap_factor) {
  return GetLookupValidities(p_states, wrap_factor);
}

OneBodyBasis OneBodyBasis::FromQuantumNumbers(const Operator &Z, int j2min,
                                              int j2max, int parity, int tz2) {
  const std::size_t wrap_factor = ExtractWrapFactor(Z);

  std::vector<std::size_t> states_1b(Z.modelspace->orbits_3body_space_.begin(),
                                     Z.modelspace->orbits_3body_space_.end());
  std::sort(states_1b.begin(), states_1b.end());

  std::vector<std::size_t> p_states;
  for (const auto &p : states_1b) {
    const Orbit &op = Z.modelspace->GetOrbit(p);
    if ((op.tz2 == tz2) && (op.l % 2 == parity) && (op.j2 <= j2max) &&
        (op.j2 >= j2min)) {
      p_states.push_back(p);
    }
  }

  return OneBodyBasis(p_states, wrap_factor);
}

OneBodyBasis OneBodyBasis::FromQuantumNumbers_H(const Operator &Z, int j2min,
                                                int j2max, int parity,
                                                int tz2) {
  const std::size_t wrap_factor = ExtractWrapFactor(Z);

  std::vector<std::size_t> states_1b(Z.modelspace->orbits_3body_space_.begin(),
                                     Z.modelspace->orbits_3body_space_.end());
  std::sort(states_1b.begin(), states_1b.end());

  std::vector<std::size_t> p_states;
  for (const auto &p : states_1b) {
    const Orbit &op = Z.modelspace->GetOrbit(p);
    if ((op.tz2 == tz2) && (op.l % 2 == parity) && (op.j2 <= j2max) &&
        (op.j2 >= j2min) && (std::abs(op.occ) > 1e-12)) {
      p_states.push_back(p);
    }
  }

  return OneBodyBasis(p_states, wrap_factor);
}

OneBodyBasis OneBodyBasis::FromQuantumNumbers_P(const Operator &Z, int j2min,
                                                int j2max, int parity,
                                                int tz2) {
  const std::size_t wrap_factor = ExtractWrapFactor(Z);

  std::vector<std::size_t> states_1b(Z.modelspace->orbits_3body_space_.begin(),
                                     Z.modelspace->orbits_3body_space_.end());
  std::sort(states_1b.begin(), states_1b.end());

  std::vector<std::size_t> p_states;
  for (const auto &p : states_1b) {
    const Orbit &op = Z.modelspace->GetOrbit(p);
    if ((op.tz2 == tz2) && (op.l % 2 == parity) && (op.j2 <= j2max) &&
        (op.j2 >= j2min) && (std::abs(1 - op.occ) > 1e-12)) {
      p_states.push_back(p);
    }
  }

  return OneBodyBasis(p_states, wrap_factor);
}

std::vector<std::size_t> Get2BPStates(const std::vector<std::size_t> &pq_states,
                                      std::size_t wrap_factor) {
  std::vector<std::size_t> p_states(pq_states.size(), 0);
  std::transform(
      pq_states.begin(), pq_states.end(), p_states.begin(),
      [&wrap_factor](const std::size_t &pq) { return pq / wrap_factor; });
  return p_states;
}

std::vector<std::size_t> Get2BQStates(const std::vector<std::size_t> &pq_states,
                                      std::size_t wrap_factor) {
  std::vector<std::size_t> q_states(pq_states.size(), 0);
  std::transform(
      pq_states.begin(), pq_states.end(), q_states.begin(),
      [&wrap_factor](const std::size_t &pq) { return pq % wrap_factor; });
  return q_states;
}

std::vector<std::size_t>
Get2BPQIndices(const std::vector<std::size_t> &pq_states,
               std::size_t wrap_factor) {
  return GetLookupIndices(pq_states, wrap_factor * wrap_factor);
}

std::vector<int> Get2BPQValidities(const std::vector<std::size_t> &pq_states,
                                   std::size_t wrap_factor) {
  return GetLookupValidities(pq_states, wrap_factor * wrap_factor);
}

TwoBodyBasis TwoBodyBasis::PQInTwoBodyChannel(std::size_t i_ch_2b,
                                              const Operator &Z) {
  const std::size_t wrap_factor = ExtractWrapFactor(Z);
  const TwoBodyChannel &ch_2b = Z.modelspace->GetTwoBodyChannel(i_ch_2b);

  std::vector<std::size_t> states_1b(Z.modelspace->orbits_3body_space_.begin(),
                                     Z.modelspace->orbits_3body_space_.end());
  std::sort(states_1b.begin(), states_1b.end());

  std::vector<std::size_t> pq_states;
  std::vector<double> pq_factors;
  for (const auto &p : states_1b) {
    const Orbit &op = Z.modelspace->GetOrbit(p);
    for (const auto &q : states_1b) {
      const Orbit &oq = Z.modelspace->GetOrbit(q);
      if ((p <= q) && ((p != q) || (ch_2b.J % 2 == 0)) && // Pauli principle
          (op.tz2 + oq.tz2 == ch_2b.Tz * 2) &&
          ((op.l + oq.l) % 2 == ch_2b.parity) &&
          (std::abs(op.j2 - oq.j2) <= ch_2b.J * 2) &&
          (std::abs(op.j2 + oq.j2) >= ch_2b.J * 2)) {
        pq_states.push_back(p * wrap_factor + q);
        double norm_pq = 1.0;
        if (p == q) {
          norm_pq = 0.5;
        }
        pq_factors.push_back(norm_pq);
      }
    }
  }

  return TwoBodyBasis(pq_states, wrap_factor, pq_factors);
}

TwoBodyBasis TwoBodyBasis::PQInTwoBodyChannelWithE3Max(std::size_t i_ch_2b,
                                                       const Operator &Z,
                                                       int e3max) {
  const std::size_t wrap_factor = ExtractWrapFactor(Z);
  const TwoBodyChannel &ch_2b = Z.modelspace->GetTwoBodyChannel(i_ch_2b);

  std::vector<std::size_t> states_1b(Z.modelspace->orbits_3body_space_.begin(),
                                     Z.modelspace->orbits_3body_space_.end());
  std::sort(states_1b.begin(), states_1b.end());

  std::vector<std::size_t> pq_states;
  std::vector<double> pq_factors;
  for (const auto &p : states_1b) {
    const Orbit &op = Z.modelspace->GetOrbit(p);
    const int ep = op.n * 2 + op.l;
    for (const auto &q : states_1b) {
      const Orbit &oq = Z.modelspace->GetOrbit(q);
      const int eq = oq.n * 2 + oq.l;
      if ((p <= q) && ((p != q) || (ch_2b.J % 2 == 0)) && // Pauli principle
          (op.tz2 + oq.tz2 == ch_2b.Tz * 2) &&
          ((op.l + oq.l) % 2 == ch_2b.parity) &&
          (std::abs(op.j2 - oq.j2) <= ch_2b.J * 2) &&
          (std::abs(op.j2 + oq.j2) >= ch_2b.J * 2) && (ep + eq <= e3max)) {
        pq_states.push_back(p * wrap_factor + q);
        double norm_pq = 1.0;
        if (p == q) {
          norm_pq = 0.5;
        }
        pq_factors.push_back(norm_pq);
      }
    }
  }

  return TwoBodyBasis(pq_states, wrap_factor, pq_factors);
}

TwoBodyBasis TwoBodyBasis::PQInTwoBodyChannelWithE3Max_HH(std::size_t i_ch_2b,
                                                          const Operator &Z,
                                                          int e3max) {
  const std::size_t wrap_factor = ExtractWrapFactor(Z);
  const TwoBodyChannel &ch_2b = Z.modelspace->GetTwoBodyChannel(i_ch_2b);

  std::vector<std::size_t> states_1b(Z.modelspace->orbits_3body_space_.begin(),
                                     Z.modelspace->orbits_3body_space_.end());
  std::sort(states_1b.begin(), states_1b.end());

  std::vector<std::size_t> pq_states;
  std::vector<double> pq_factors;
  for (const auto &p : states_1b) {
    const Orbit &op = Z.modelspace->GetOrbit(p);
    const int ep = op.n * 2 + op.l;
    for (const auto &q : states_1b) {
      const Orbit &oq = Z.modelspace->GetOrbit(q);
      const int eq = oq.n * 2 + oq.l;
      if ((p <= q) && ((p != q) || (ch_2b.J % 2 == 0)) && // Pauli principle
          (op.tz2 + oq.tz2 == ch_2b.Tz * 2) &&
          ((op.l + oq.l) % 2 == ch_2b.parity) &&
          (std::abs(op.j2 - oq.j2) <= ch_2b.J * 2) &&
          (std::abs(op.j2 + oq.j2) >= ch_2b.J * 2) && (ep + eq <= e3max) &&
          (std::abs(op.occ * oq.occ) > 1e-12)) {
        pq_states.push_back(p * wrap_factor + q);
        double norm_pq = op.occ * oq.occ;
        if (p == q) {
          norm_pq *= 0.5;
        }
        pq_factors.push_back(norm_pq);
      }
    }
  }

  return TwoBodyBasis(pq_states, wrap_factor, pq_factors);
}

TwoBodyBasis TwoBodyBasis::PQInTwoBodyChannelWithE3Max_PP(std::size_t i_ch_2b,
                                                          const Operator &Z,
                                                          int e3max) {
  const std::size_t wrap_factor = ExtractWrapFactor(Z);
  const TwoBodyChannel &ch_2b = Z.modelspace->GetTwoBodyChannel(i_ch_2b);

  std::vector<std::size_t> states_1b(Z.modelspace->orbits_3body_space_.begin(),
                                     Z.modelspace->orbits_3body_space_.end());
  std::sort(states_1b.begin(), states_1b.end());

  std::vector<std::size_t> pq_states;
  std::vector<double> pq_factors;
  for (const auto &p : states_1b) {
    const Orbit &op = Z.modelspace->GetOrbit(p);
    const int ep = op.n * 2 + op.l;
    for (const auto &q : states_1b) {
      const Orbit &oq = Z.modelspace->GetOrbit(q);
      const int eq = oq.n * 2 + oq.l;
      if ((p <= q) && ((p != q) || (ch_2b.J % 2 == 0)) && // Pauli principle
          (op.tz2 + oq.tz2 == ch_2b.Tz * 2) &&
          ((op.l + oq.l) % 2 == ch_2b.parity) &&
          (std::abs(op.j2 - oq.j2) <= ch_2b.J * 2) &&
          (std::abs(op.j2 + oq.j2) >= ch_2b.J * 2) && (ep + eq <= e3max) &&
          (std::abs((1 - op.occ) * (1 - oq.occ)) > 1e-12)) {
        pq_states.push_back(p * wrap_factor + q);
        double norm_pq = (1 - op.occ) * (1 - oq.occ);
        if (p == q) {
          norm_pq *= 0.5;
        }
        pq_factors.push_back(norm_pq);
      }
    }
  }

  return TwoBodyBasis(pq_states, wrap_factor, pq_factors);
}

std::vector<std::size_t>
Get3BPStates(const std::vector<std::size_t> &pqr_states,
             std::size_t wrap_factor) {
  std::vector<std::size_t> p_states(pqr_states.size(), 0);
  std::transform(pqr_states.begin(), pqr_states.end(), p_states.begin(),
                 [&wrap_factor](const std::size_t &pqr) {
                   return (pqr / wrap_factor) / wrap_factor;
                 });
  return p_states;
}

std::vector<std::size_t>
Get3BQStates(const std::vector<std::size_t> &pqr_states,
             std::size_t wrap_factor) {
  std::vector<std::size_t> q_states(pqr_states.size(), 0);
  std::transform(pqr_states.begin(), pqr_states.end(), q_states.begin(),
                 [&wrap_factor](const std::size_t &pqr) {
                   return (pqr / wrap_factor) % wrap_factor;
                 });
  return q_states;
}

std::vector<std::size_t>
Get3BRStates(const std::vector<std::size_t> &pqr_states,
             std::size_t wrap_factor) {
  std::vector<std::size_t> r_states(pqr_states.size(), 0);
  std::transform(
      pqr_states.begin(), pqr_states.end(), r_states.begin(),
      [&wrap_factor](const std::size_t &pqr) { return pqr % wrap_factor; });
  return r_states;
}

std::vector<std::size_t>
Get3BPQRIndices(const std::vector<std::size_t> &pqr_states,
                std::size_t wrap_factor) {
  return GetLookupIndices(pqr_states, wrap_factor * wrap_factor * wrap_factor);
}

std::vector<int> Get3BPQRValidities(const std::vector<std::size_t> &pqr_states,
                                    std::size_t wrap_factor) {
  return GetLookupValidities(pqr_states,
                             wrap_factor * wrap_factor * wrap_factor);
}

void GetRecoupling(std::size_t i_ch_3b, std::size_t i_ch_2b, const Operator &Z,
                   const std::vector<std::size_t> &states_3b_p,
                   const std::vector<std::size_t> &states_3b_q,
                   const std::vector<std::size_t> &states_3b_r,
                   std::vector<std::vector<std::size_t>> &indices,
                   std::vector<std::vector<double>> &recoupling) {
  int twoJ = Z.modelspace->GetThreeBodyChannel(i_ch_3b).twoJ;
  int Jab = Z.modelspace->GetTwoBodyChannel(i_ch_2b).J;

  for (std::size_t i = 0; i < states_3b_p.size(); i += 1) {
    Z.ThreeBody.GetKetIndex_withRecoupling(Jab, twoJ, states_3b_p[i],
                                           states_3b_q[i], states_3b_r[i],
                                           indices[i], recoupling[i]);
  }
}

ThreeBodyBasis ThreeBodyBasis::From2BAnd1BBasis(
    std::size_t i_ch_3b, std::size_t i_ch_2b, const Operator &Z,
    const TwoBodyBasis &basis_pq, const OneBodyBasis &basis_r, int e3max) {
  const std::size_t wrap_factor = ExtractWrapFactor(Z);

  std::size_t num_states = 0;
  for (const std::size_t &pq : basis_pq.GetPQVals()) {
    const std::size_t p = pq / wrap_factor;
    const std::size_t q = pq % wrap_factor;
    const Orbit &op = Z.modelspace->GetOrbit(p);
    const Orbit &oq = Z.modelspace->GetOrbit(q);
    const int ep = op.n * 2 + op.l;
    const int eq = oq.n * 2 + oq.l;
    for (const std::size_t &r : basis_r.GetPVals()) {
      const Orbit &oR = Z.modelspace->GetOrbit(r);
      const int er = oR.n * 2 + oR.l;

      if (ep + eq + er <= e3max) {
        num_states += 1;
      }
    }
  }

  std::vector<std::size_t> pqr_states;
  std::vector<double> pqr_factors;
  pqr_states.reserve(num_states);
  pqr_factors.reserve(num_states);
  for (std::size_t i_pq = 0; i_pq < basis_pq.BasisSize(); i_pq += 1) {
    const std::size_t pq = basis_pq.GetPQFactors()[i_pq];
    const std::size_t p = basis_pq.GetPVals()[i_pq];
    const std::size_t q = basis_pq.GetQVals()[i_pq];
    const double pq_factor = basis_pq.GetPQFactors()[i_pq];
    const Orbit &op = Z.modelspace->GetOrbit(p);
    const Orbit &oq = Z.modelspace->GetOrbit(q);
    const int ep = op.n * 2 + op.l;
    const int eq = oq.n * 2 + oq.l;
    for (const std::size_t &r : basis_r.GetPVals()) {
      const Orbit &oR = Z.modelspace->GetOrbit(r);
      const int er = oR.n * 2 + oR.l;

      if (ep + eq + er <= e3max) {
        pqr_states.push_back(pq * wrap_factor + r);
        pqr_factors.push_back(pq_factor);
      }
    }
  }

  return ThreeBodyBasis(i_ch_3b, i_ch_2b, Z, pqr_states, wrap_factor,
                        pqr_factors);
}

ThreeBodyBasis ThreeBodyBasis::MinFrom2BAnd1BBasis(
    std::size_t i_ch_3b, std::size_t i_ch_2b, const Operator &Z,
    const TwoBodyBasis &basis_pq, const OneBodyBasis &basis_r, int e3max) {
  const std::size_t wrap_factor = ExtractWrapFactor(Z);

  std::size_t num_states = 0;
  for (const std::size_t &pq : basis_pq.GetPQVals()) {
    const std::size_t p = pq / wrap_factor;
    const std::size_t q = pq % wrap_factor;
    if (p > q) {
      continue;
    }
    const Orbit &op = Z.modelspace->GetOrbit(p);
    const Orbit &oq = Z.modelspace->GetOrbit(q);
    const int ep = op.n * 2 + op.l;
    const int eq = oq.n * 2 + oq.l;
    for (const std::size_t &r : basis_r.GetPVals()) {
      if (r > q) {
        continue;
      }
      const Orbit &oR = Z.modelspace->GetOrbit(r);
      const int er = oR.n * 2 + oR.l;

      if (ep + eq + er <= e3max) {
        num_states += 1;
      }
    }
  }

  std::vector<std::size_t> pqr_states;
  std::vector<double> pqr_factors;
  pqr_states.reserve(num_states);
  pqr_factors.reserve(num_states);
  for (std::size_t i_pq = 0; i_pq < basis_pq.BasisSize(); i_pq += 1) {
    const std::size_t pq = basis_pq.GetPQFactors()[i_pq];
    const std::size_t p = basis_pq.GetPVals()[i_pq];
    const std::size_t q = basis_pq.GetQVals()[i_pq];
    if (p > q) {
      continue;
    }
    const Orbit &op = Z.modelspace->GetOrbit(p);
    const Orbit &oq = Z.modelspace->GetOrbit(q);
    const int ep = op.n * 2 + op.l;
    const int eq = oq.n * 2 + oq.l;
    for (const std::size_t &r : basis_r.GetPVals()) {
      if (r > q) {
        continue;
      }
      const Orbit &oR = Z.modelspace->GetOrbit(r);
      const int er = oR.n * 2 + oR.l;
      double factor = 1.0;
      if ((p == q) && (q == r)) {
        factor = 1.0 / 6.0;
      } else {
        if ((p == q) || (q == r)) {
          factor = 1.0 / 2.0;
        }
      }

      if (ep + eq + er <= e3max) {
        pqr_states.push_back(pq * wrap_factor + r);
        pqr_factors.push_back(factor);
      }
    }
  }

  return ThreeBodyBasis(i_ch_3b, i_ch_2b, Z, pqr_states, wrap_factor,
                        pqr_factors);
}

std::size_t ThreeBodyBasis::NumBytes() const {
  std::size_t size =
      pqr_states.size() * sizeof(std::size_t) +
      p_states.size() * sizeof(std::size_t) +
      q_states.size() * sizeof(std::size_t) +
      r_states.size() * sizeof(std::size_t) +
      pqr_indices.size() * sizeof(std::size_t) +
      pqr_validities.size() * sizeof(int) +
      pqr_me_indices.size() * sizeof(std::vector<std::size_t>) +
      pqr_me_recoupling_factors.size() * sizeof(std::vector<double>);
  for (const auto &indices : pqr_me_indices) {
    size += indices.size() * sizeof(std::size_t);
  }
  for (const auto &recouplings : pqr_me_recoupling_factors) {
    size += recouplings.size() * sizeof(double);
  }
  return size;
}

} // namespace internal
} // namespace comm331