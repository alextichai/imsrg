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

#ifndef COMMUTATOR232_H_
#define COMMUTATOR232_H_

#include <cstdlib>
#include <unordered_map>
#include <vector>

#include "ModelSpace.hh"
#include "Operator.hh"

namespace comm232 {

void comm232ss_expand_impl_new(const Operator &X, const Operator &Y,
                               Operator &Z); // implemented and tested.

void comm232ss_expand_impl(const Operator &X, const Operator &Y,
                           Operator &Z); // implemented and tested.

namespace internal {

// Generally useful methods

std::size_t ExtractWrapFactor(const Operator &Z);
int ExtractJJ1Max(const Operator &Z);
std::vector<std::size_t> Extract2BChannelsValidIn3BChannel(int jj1max,
                                                           std::size_t i_ch_3b,
                                                           const Operator &Z);

// Methods for 1-body bases

std::vector<std::size_t> Get1BPIndices(const std::vector<std::size_t> &p_states,
                                       std::size_t wrap_factor);
std::vector<int> Get1BPValidities(const std::vector<std::size_t> &p_states,
                                  std::size_t wrap_factor);

class OneBodyBasis {
public:
  // TODO: Ctor, factory methods
  static OneBodyBasis FromQuantumNumbers(const Operator &Z, int j2min,
                                         int j2max, int parity, int tz2);

  OneBodyBasis(const std::vector<std::size_t> &p_states_,
               std::size_t wrap_factor_)
      : wrap_factor(wrap_factor_), p_states(p_states_) {}

  std::size_t BasisSize() const { return p_states.size(); }
  const std::vector<std::size_t> &GetPVals() const { return p_states; }
  std::size_t GetLocalIndexForP(std::size_t p) const { return p_indices[p]; }
  int GetLocalValidityForP(std::size_t p) const { return p_validities[p]; }

private:
  std::size_t wrap_factor;
  std::vector<std::size_t> p_states;
  std::vector<std::size_t> p_indices = Get1BPIndices(p_states, wrap_factor);
  std::vector<int> p_validities = Get1BPValidities(p_states, wrap_factor);
};

// Methods for 2-body bases

std::vector<std::size_t> Get2BPStates(const std::vector<std::size_t> &pq_states,
                                      std::size_t wrap_factor);
std::vector<std::size_t> Get2BQStates(const std::vector<std::size_t> &pq_states,
                                      std::size_t wrap_factor);
std::vector<std::size_t>
Get2BPQIndices(const std::vector<std::size_t> &pq_states,
               std::size_t wrap_factor);
std::vector<int> Get2BPQValidities(const std::vector<std::size_t> &pq_states,
                                   std::size_t wrap_factor);

class TwoBodyBasis {
public:
  // TODO: Ctor, factory methods
  static TwoBodyBasis PQInTwoBodyChannel(std::size_t i_ch_2b,
                                         const Operator &Z);
  static TwoBodyBasis PQInTwoBodyChannelWithE3Max(std::size_t i_ch_2b,
                                                  const Operator &Z, int e3max);

  TwoBodyBasis(const std::vector<std::size_t> &pq_states_,
               std::size_t wrap_factor_)
      : wrap_factor(wrap_factor_), pq_states(pq_states_) {}

  std::size_t BasisSize() const { return pq_states.size(); }
  const std::vector<std::size_t> &GetPQVals() const { return pq_states; }
  const std::vector<std::size_t> &GetPVals() const { return p_states; }
  const std::vector<std::size_t> &GetQVals() const { return q_states; }
  std::size_t GetLocalIndexForPQ(std::size_t p, std::size_t q) const {
    return pq_indices[p * wrap_factor + q];
  }
  int GetLocalValidityForPQ(std::size_t p, std::size_t q) const {
    return pq_validities[p * wrap_factor + q];
  }

private:
  std::size_t wrap_factor;
  std::vector<std::size_t> pq_states;
  std::vector<std::size_t> p_states = Get2BPStates(pq_states, wrap_factor);
  std::vector<std::size_t> q_states = Get2BQStates(pq_states, wrap_factor);
  std::vector<std::size_t> pq_indices = Get2BPQIndices(pq_states, wrap_factor);
  std::vector<int> pq_validities = Get2BPQValidities(pq_states, wrap_factor);
};

// Methods for 3-body bases

std::vector<std::size_t>
Get3BPStates(const std::vector<std::size_t> &pqr_states,
             std::size_t wrap_factor);
std::vector<std::size_t>
Get3BQStates(const std::vector<std::size_t> &pqr_states,
             std::size_t wrap_factor);
std::vector<std::size_t>
Get3BRStates(const std::vector<std::size_t> &pqr_states,
             std::size_t wrap_factor);
std::vector<std::size_t>
Get3BPQRIndices(const std::vector<std::size_t> &pqr_states,
                std::size_t wrap_factor);
std::vector<int> Get3BPQRValidities(const std::vector<std::size_t> &pqr_states,
                                    std::size_t wrap_factor);
void GetRecoupling(std::size_t i_ch_3b, std::size_t i_ch_2b, const Operator &Z,
                   const std::vector<std::size_t> &states_3b_p,
                   const std::vector<std::size_t> &states_3b_q,
                   const std::vector<std::size_t> &states_3b_r,
                   std::vector<std::vector<std::size_t>> &indices,
                   std::vector<std::vector<double>> &recoupling);

class ThreeBodyBasis {
public:
  static ThreeBodyBasis From2BAnd1BBasis(std::size_t i_ch_3b,
                                         std::size_t i_ch_2b, const Operator &Z,
                                         const TwoBodyBasis &basis_pq,
                                         const OneBodyBasis &basis_r,
                                         int e3max);

  ThreeBodyBasis(std::size_t i_ch_3b, std::size_t i_ch_2b, const Operator &Z,
                 const std::vector<std::size_t> &pqr_states_,
                 std::size_t wrap_factor_)
      : wrap_factor(wrap_factor_), pqr_states(pqr_states_) {
    GetRecoupling(i_ch_3b, i_ch_2b, Z, p_states, q_states, r_states,
                  pqr_me_indices, pqr_me_recoupling_factors);
  }

  std::size_t BasisSize() const { return pqr_states.size(); }
  const std::vector<std::size_t> &GetPQRVals() const { return pqr_states; }
  const std::vector<std::size_t> &GetPVals() const { return p_states; }
  const std::vector<std::size_t> &GetQVals() const { return q_states; }
  const std::vector<std::size_t> &GetRVals() const { return r_states; }
  std::size_t GetLocalIndexForPQR(std::size_t p, std::size_t q,
                                  std::size_t r) const {
    return pqr_indices[p * wrap_factor * wrap_factor + q * wrap_factor + r];
  }
  int GetLocalValidityForPQR(std::size_t p, std::size_t q,
                             std::size_t r) const {
    return pqr_validities[p * wrap_factor * wrap_factor + q * wrap_factor + r];
  }
  const std::vector<std::size_t> &
  GetRecouplingIndices(std::size_t i_pqr) const {
    return pqr_me_indices[i_pqr];
  }
  const std::vector<double> &GetRecouplingFactors(std::size_t i_pqr) const {
    return pqr_me_recoupling_factors[i_pqr];
  }

private:
  std::size_t wrap_factor;
  std::vector<std::size_t> pqr_states;
  std::vector<std::size_t> p_states = Get3BPStates(pqr_states, wrap_factor);
  std::vector<std::size_t> q_states = Get3BQStates(pqr_states, wrap_factor);
  std::vector<std::size_t> r_states = Get3BRStates(pqr_states, wrap_factor);
  std::vector<std::size_t> pqr_indices =
      Get3BPQRIndices(pqr_states, wrap_factor);
  std::vector<int> pqr_validities = Get3BPQRValidities(pqr_states, wrap_factor);
  std::vector<std::vector<std::size_t>> pqr_me_indices =
      std::vector<std::vector<std::size_t>>(wrap_factor * wrap_factor *
                                            wrap_factor);
  std::vector<std::vector<double>> pqr_me_recoupling_factors =
      std::vector<std::vector<double>>(wrap_factor * wrap_factor * wrap_factor);
};

// Methods for tensors

inline std::size_t Index2B(std::size_t p, std::size_t q, std::size_t dim_q) {
  return p * dim_q + q;
}
inline std::size_t Index3B(std::size_t p, std::size_t q, std::size_t r,
                           std::size_t dim_q, std::size_t dim_r) {
  return p * dim_q * dim_r + q * dim_r + r;
}
inline std::size_t Index4B(std::size_t p, std::size_t q, std::size_t r,
                           std::size_t s, std::size_t dim_q, std::size_t dim_r,
                           std::size_t dim_s) {
  return p * dim_q * dim_r * dim_s + q * dim_r * dim_s + r * dim_s + s;
}

std::vector<double> Generate3BMatrix(const Operator &Z, std::size_t i_ch_3b,
                                     const ThreeBodyBasis &basis_ijc,
                                     const ThreeBodyBasis &basis_abalpha,
                                     const TwoBodyBasis &basis_ij,
                                     const OneBodyBasis &basis_alpha,
                                     const TwoBodyBasis &basis_ab,
                                     const OneBodyBasis &basis_c);

std::vector<double> Generate2BMatrix(const Operator &Z, std::size_t i_ch_2b_ab,
                                     const TwoBodyBasis &basis_ab,
                                     const OneBodyBasis &basis_beta,
                                     const OneBodyBasis &basis_c);

std::vector<double> GenerateOccsMatrix(const Operator &Z,
                                     const TwoBodyBasis &basis_ab,
                                     const OneBodyBasis &basis_c);

std::vector<double> GenerateSixJMatrix(const Operator &Z,
                                     const TwoBodyBasis &basis_ij,
                                     const TwoBodyBasis &basis_ab,
                                     const OneBodyBasis &basis_c,
                                     int JJ_ij,
                                     int JJ_3,
                                     int JJ_ab
                                     );

void EvaluateComm232Diagram1(
  double factor,
  std::size_t i_ch_2b_ij,
  const TwoBodyBasis& basis_ab_e3max,
  const TwoBodyBasis& basis_ij_e3max,
  const TwoBodyBasis& basis_ij,
  const OneBodyBasis& basis_alpha,
  const OneBodyBasis& basis_beta,
  const OneBodyBasis& basis_c,
  const std::vector<double> mat_3b,
  const std::vector<double> mat_2b,
  const std::vector<double> occs,
  const std::vector<double> six_js,
  Operator& Z
);

double Comm232Core(
  const double* slice_sixj,
  const double* slice_occs,
  const double* slice_mat_2b,
  const double* slice_mat_3b,
  std::size_t slice_size
);

class CollectedBases {
  public:

  // Only via move
  CollectedBases(TwoBodyBasis&& basis_2b, TwoBodyBasis&& basis_2b_e3max, OneBodyBasis&& basis_1b, ThreeBodyBasis&& basis_3b) 
  : basis_2b_(std::move(basis_2b)),
  basis_2b_e3max_(std::move(basis_2b_e3max)),
  basis_1b_(std::move(basis_1b)),
  basis_3b_(std::move(basis_3b)) {}

  const TwoBodyBasis& BasisPQ() const { return basis_2b_; }
  const TwoBodyBasis& BasisPQE3Max() const { return basis_2b_e3max_; }
  const OneBodyBasis& BasisR() const { return basis_1b_; }
  const ThreeBodyBasis& BasisPQR() const { return basis_3b_; }

  private:
  TwoBodyBasis basis_2b_;
  TwoBodyBasis basis_2b_e3max_;
  OneBodyBasis basis_1b_;
  ThreeBodyBasis basis_3b_;
};

std::unordered_map<std::size_t, CollectedBases> PrestoreBases(
  std::size_t i_ch_3b,
  int jj1max,
  const Operator& Z,
  int e3max
);

// Previous methods

int GetJJ1Max(const Operator &Y);
std::vector<std::size_t>
Get2BChannelsValidIn3BChannel(int jj1max, const ThreeBodyChannel &ch_3b,
                              const Operator &Y);

class Basis3BLookupSingleChannel {
public:
  Basis3BLookupSingleChannel(std::size_t i_ch_3b, std::size_t i_ch_2b,
                             const Operator &Y);

  std::size_t Get3BChannelIndex() const { return i_ch_3b_; }
  std::size_t Get2BChannelIndex() const { return i_ch_2b_; }

  std::size_t GetNum2BStates() const { return states_2b_pq.size(); }
  std::size_t Get2BStateP(std::size_t state_2b_i) const {
    return states_2b_p[state_2b_i];
  }
  std::size_t Get2BStateQ(std::size_t state_2b_i) const {
    return states_2b_q[state_2b_i];
  }

  std::size_t GetNum3BStates() const { return states_3b_pqr.size(); }
  std::size_t Get3BStateP(std::size_t state_3b_i) const {
    return states_3b_p[state_3b_i];
  }
  std::size_t Get3BStateQ(std::size_t state_3b_i) const {
    return states_3b_q[state_3b_i];
  }
  std::size_t Get3BStateR(std::size_t state_3b_i) const {
    return states_3b_r[state_3b_i];
  }

  const std::vector<std::size_t> &
  GetRecouplingIndices(std::size_t state_3b_i) const {
    return states_3b_indices[state_3b_i];
  }
  const std::vector<double> &
  GetRecouplingFactors(std::size_t state_3b_i) const {
    return states_3b_recoupling[state_3b_i];
  }

  const std::vector<std::size_t> Get3rdStateBasis() const { return basis_r; }

  std::size_t GetStateLocalIndex(std::size_t p, std::size_t q,
                                 std::size_t r) const {
    return pqr_lookup[p * wrap_factor * wrap_factor + q * wrap_factor + r];
  }

  int IsStateValid(std::size_t p, std::size_t q, std::size_t r) const {
    return pqr_lookup_validity[p * wrap_factor * wrap_factor + q * wrap_factor +
                               r];
  }

private:
  std::size_t i_ch_3b_ = 0;
  std::size_t i_ch_2b_ = 0;
  std::size_t wrap_factor = 0;
  std::vector<std::size_t> states_2b_pq;
  std::vector<std::size_t> states_2b_p;
  std::vector<std::size_t> states_2b_q;
  std::vector<std::size_t> basis_r;
  std::vector<std::size_t> states_3b_pqr;
  std::vector<std::size_t> states_3b_p;
  std::vector<std::size_t> states_3b_q;
  std::vector<std::size_t> states_3b_r;
  std::vector<std::vector<std::size_t>> states_3b_indices;
  std::vector<std::vector<double>> states_3b_recoupling;
  std::vector<std::size_t> pqr_lookup =
      std::vector<std::size_t>(wrap_factor * wrap_factor * wrap_factor, 0);
  std::vector<int> pqr_lookup_validity =
      std::vector<int>(wrap_factor * wrap_factor * wrap_factor, 0);
};

void UnpackMatrix(std::size_t i_ch_3b,
                  const Basis3BLookupSingleChannel &lookup_bra,
                  const Basis3BLookupSingleChannel &lookup_ket,
                  const Operator &op, arma::mat &mat);

void Comm232Diagram1(std::size_t i_ch_2b_bra, std::size_t i_ch_2b_ket,
                     const ThreeBodyChannel &ch_3b,
                     const Basis3BLookupSingleChannel &lookup_bra,
                     const Basis3BLookupSingleChannel &lookup_ket,
                     const Operator &X, const arma::mat &Y_mat,
                     double overall_factor, Operator &Z);

void Comm232Diagram2(std::size_t i_ch_2b_bra, std::size_t i_ch_2b_ket,
                     const ThreeBodyChannel &ch_3b,
                     const Basis3BLookupSingleChannel &lookup_bra,
                     const Basis3BLookupSingleChannel &lookup_ket,
                     const Operator &X, const arma::mat &Y_mat,
                     double overall_factor, Operator &Z);

void Comm232Diagram3(std::size_t i_ch_2b_bra, std::size_t i_ch_2b_ket,
                     const ThreeBodyChannel &ch_3b,
                     const Basis3BLookupSingleChannel &lookup_bra,
                     const Basis3BLookupSingleChannel &lookup_ket,
                     const Operator &X, const arma::mat &Y_mat,
                     double overall_factor, Operator &Z);

void Comm232Diagram4(std::size_t i_ch_2b_bra, std::size_t i_ch_2b_ket,
                     const ThreeBodyChannel &ch_3b,
                     const Basis3BLookupSingleChannel &lookup_bra,
                     const Basis3BLookupSingleChannel &lookup_ket,
                     const Operator &X, const arma::mat &Y_mat,
                     double overall_factor, Operator &Z);

inline double GetSPStateJ(std::size_t orb_i, const Operator &Y) {
  return Y.modelspace->GetOrbit(orb_i).j2 * 0.5;
}

inline int GetPhase(std::size_t orb_i, std::size_t orb_j, int J_ij,
                    const Operator &Y) {
  int jj_i = Y.modelspace->GetOrbit(orb_i).j2;
  int jj_j = Y.modelspace->GetOrbit(orb_j).j2;
  return Y.modelspace->phase((jj_i + jj_j) / 2 - J_ij);
}

inline double GetSPStateOcc(std::size_t orb_i, const Operator &Y) {
  return Y.modelspace->GetOrbit(orb_i).occ;
}

inline double GetSPStateOccBar(std::size_t orb_i, const Operator &Y) {
  return 1.0 - Y.modelspace->GetOrbit(orb_i).occ;
}

} // namespace internal
} // namespace comm232

#endif // COMMUTATOR232_H_
