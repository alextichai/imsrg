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
#ifndef COMMUTATOR331_H_
#define COMMUTATOR331_H_

#include <cstdlib>
#include <unordered_map>
#include <vector>

#include "ModelSpace.hh"
#include "Operator.hh"

namespace comm331 {

// The core implementation of the 331 commutator
void comm331ss_expand_impl(const Operator &X, const Operator &Y,
                           Operator &Z); // implemented and tested.

// internal namespace for helper methods and structures
namespace internal {

// Generally useful methods

// Get factor s such that we can make a unique index pq from p and q
// via p * s + q (p + q * s).
//
// This is in practice 1 larger than the largest orbit index.
std::size_t ExtractWrapFactor(const Operator &Z);

// Get the largest 1-body angular momentum in the 3-body space.
int ExtractJJ1Max(const Operator &Z);

// Get indices of 2B channels that are valid subchannels of a 3B channel.
//
// This uses symmetries, i.e. isopsin projection and possible coupling ranges.
std::vector<std::size_t> Extract2BChannelsValidIn3BChannel(int jj1max,
                                                           std::size_t i_ch_3b,
                                                           const Operator &Z);

// Methods for 1-body bases

// Get a lookup table such that lookup[p] = i_p, the local index of p in
// p_states.
std::vector<std::size_t> Get1BPIndices(const std::vector<std::size_t> &p_states,
                                       std::size_t wrap_factor);

// Get a lookup table such that lookup[p] = 1 if p is in p_states, 0 otherwise.
std::vector<int> Get1BPValidities(const std::vector<std::size_t> &p_states,
                                  std::size_t wrap_factor);

// Basis of single-particle states |p>.
class OneBodyBasis {
public:
  // Make a 1B basis (from emax_3body, not emax) using given quantum numbers.
  static OneBodyBasis FromQuantumNumbers(const Operator &Z, int j2min,
                                         int j2max, int parity, int tz2);

  // Make a 1B basis of hole states (from emax_3body, not emax) using given
  // quantum numbers.
  //
  // "hole" refers to the general VS-IMSRG hole where n_p > 0.
  static OneBodyBasis FromQuantumNumbers_H(const Operator &Z, int j2min,
                                           int j2max, int parity, int tz2);

  // Make a 1B basis of particle states (from emax_3body, not emax) using given
  // quantum numbers.
  //
  // "particle" refers to the general VS-IMSRG particle where n_p < 1.
  static OneBodyBasis FromQuantumNumbers_P(const Operator &Z, int j2min,
                                           int j2max, int parity, int tz2);

  // Construct 1B basis directly.
  //
  // Do not do this directly. Prefer using a factory method (see above).
  OneBodyBasis(const std::vector<std::size_t> &p_states_,
               std::size_t wrap_factor_)
      : wrap_factor(wrap_factor_), p_states(p_states_) {}

  // Get the basis size.
  std::size_t BasisSize() const { return p_states.size(); }

  // Get reference to the vector p states in the basis.
  const std::vector<std::size_t> &GetPVals() const { return p_states; }

  // Look up the local index for p in the basis (if it exists).
  //
  // Implementation detail:
  // If p is not in the basis, this will return 0.
  // You should probably not rely on this implementation detail though.
  std::size_t GetLocalIndexForP(std::size_t p) const { return p_indices[p]; }

  // Look up whether p is in the basis.
  int GetLocalValidityForP(std::size_t p) const { return p_validities[p]; }

private:
  std::size_t wrap_factor;
  std::vector<std::size_t> p_states;
  // The following members are automatically constructed in the right way.
  // Please do not overwrite this initialization in the constructor.
  std::vector<std::size_t> p_indices = Get1BPIndices(p_states, wrap_factor);
  std::vector<int> p_validities = Get1BPValidities(p_states, wrap_factor);
};

// Methods for 2-body bases

// Extract vector of p states from pq states such that
// For each i, result[i] = p from pq_states[i] (however pq -> p is defined).
//
// Currently pq = p * wrap_factor + q.
std::vector<std::size_t> Get2BPStates(const std::vector<std::size_t> &pq_states,
                                      std::size_t wrap_factor);

// Extract vector of q states from pq states such that
// For each i, result[i] = q from pq_states[i] (however pq -> q is defined).
//
// Currently pq = p * wrap_factor + q.
std::vector<std::size_t> Get2BQStates(const std::vector<std::size_t> &pq_states,
                                      std::size_t wrap_factor);

// Get a lookup table such that lookup[pq] = i_pq, the local index of pq in
// pq_states.
std::vector<std::size_t>
Get2BPQIndices(const std::vector<std::size_t> &pq_states,
               std::size_t wrap_factor);

// Get a lookup table such that lookup[pq] = 1 if pq is in pq_states, 0
// otherwise.
std::vector<int> Get2BPQValidities(const std::vector<std::size_t> &pq_states,
                                   std::size_t wrap_factor);

// Basis of 2B states |pq>.
class TwoBodyBasis {
public:
  // Create a basis of states |pq> (p <= q) allowed in a specific channel.
  //
  // These are states in the emax_3body model space.
  static TwoBodyBasis PQInTwoBodyChannel(std::size_t i_ch_2b,
                                         const Operator &Z);

  // Create a basis of states |pq> (p <= q) allowed in a specific channel
  // that are compatible with the given e3max truncation.
  //
  // These are states in the emax_3body model space.
  static TwoBodyBasis PQInTwoBodyChannelWithE3Max(std::size_t i_ch_2b,
                                                  const Operator &Z, int e3max);

  // Create a basis of hole-hole states |pq> (p <= q) allowed in a specific
  // channel that are compatible with the given e3max truncation.
  //
  // These are states in the emax_3body model space.
  //
  // "hole" refers to the general VS-IMSRG hole where n_p > 0.
  static TwoBodyBasis PQInTwoBodyChannelWithE3Max_HH(std::size_t i_ch_2b,
                                                     const Operator &Z,
                                                     int e3max);

  // Create a basis of particle-particle states |pq> (p <= q) allowed in a
  // specific channel that are compatible with the given e3max truncation.
  //
  // These are states in the emax_3body model space.
  //
  // "particle" refers to the general VS-IMSRG particle where n_p < 1.
  static TwoBodyBasis PQInTwoBodyChannelWithE3Max_PP(std::size_t i_ch_2b,
                                                     const Operator &Z,
                                                     int e3max);

  // Construct 2B basis directly.
  //
  // Do not do this directly. Prefer using a factory method (see above).
  TwoBodyBasis(const std::vector<std::size_t> &pq_states_,
               std::size_t wrap_factor_, const std::vector<double> &pq_factors_)
      : wrap_factor(wrap_factor_), pq_states(pq_states_),
        pq_factors(pq_factors_) {}

  // Get the basis size.
  std::size_t BasisSize() const { return pq_states.size(); }

  // Get reference to the vector of |pq> states in the basis.
  const std::vector<std::size_t> &GetPQVals() const { return pq_states; }

  // Get reference to the vector of factors for |pq> states in the basis.
  //
  // This may simply be normalization factors or alternatively also
  // occupation number factors.
  const std::vector<double> &GetPQFactors() const { return pq_factors; }

  // Get reference to the vector of p states corresponding to |pq> in the basis.
  const std::vector<std::size_t> &GetPVals() const { return p_states; }

  // Get reference to the vector of q states corresponding to |pq> in the basis.
  const std::vector<std::size_t> &GetQVals() const { return q_states; }

  // Look up the local index for pq in the basis (if it exists).
  //
  // Implementation detail:
  // If pq is not in the basis, this will return 0.
  // You should probably not rely on this implementation detail though.
  std::size_t GetLocalIndexForPQ(std::size_t p, std::size_t q) const {
    return pq_indices[p * wrap_factor + q];
  }

  // Look up whether pq is in the basis.
  int GetLocalValidityForPQ(std::size_t p, std::size_t q) const {
    return pq_validities[p * wrap_factor + q];
  }

private:
  std::size_t wrap_factor;
  std::vector<std::size_t> pq_states;
  std::vector<double> pq_factors;
  // The following members are automatically constructed in the right way.
  // Please do not overwrite this initialization in the constructor.
  std::vector<std::size_t> p_states = Get2BPStates(pq_states, wrap_factor);
  std::vector<std::size_t> q_states = Get2BQStates(pq_states, wrap_factor);
  std::vector<std::size_t> pq_indices = Get2BPQIndices(pq_states, wrap_factor);
  std::vector<int> pq_validities = Get2BPQValidities(pq_states, wrap_factor);
};

// Methods for 3-body bases

// Extract vector of p states from pqr states such that
// For each i, result[i] = p from pqr_states[i] (however pqr -> p is defined).
//
// Currently pqr = p * wrap_factor * wrap_factor + q * wrap_factor + r.
std::vector<std::size_t>
Get3BPStates(const std::vector<std::size_t> &pqr_states,
             std::size_t wrap_factor);

// Extract vector of q states from pqr states such that
// For each i, result[i] = q from pqr_states[i] (however pqr -> q is defined).
//
// Currently pqr = p * wrap_factor * wrap_factor + q * wrap_factor + r.
std::vector<std::size_t>
Get3BQStates(const std::vector<std::size_t> &pqr_states,
             std::size_t wrap_factor);

// Extract vector of r states from pqr states such that
// For each i, result[i] = r from pqr_states[i] (however pqr -> r is defined).
//
// Currently pqr = p * wrap_factor * wrap_factor + q * wrap_factor + r.
std::vector<std::size_t>
Get3BRStates(const std::vector<std::size_t> &pqr_states,
             std::size_t wrap_factor);

// Get a lookup table such that lookup[pqr] = i_pqr, the local index of pqr in
// pqr_states.
std::vector<std::size_t>
Get3BPQRIndices(const std::vector<std::size_t> &pqr_states,
                std::size_t wrap_factor);

// Get a lookup table such that lookup[pqr] = 1 if pqr is in pqr_states, 0
// otherwise.
std::vector<int> Get3BPQRValidities(const std::vector<std::size_t> &pqr_states,
                                    std::size_t wrap_factor);

// Fill vectors in indices, recoupling with recoupling indices and factors
// for pqr states in states_3b_p/q/r.
//
// indices, recoupling must be filled with n empty vectors,
// where n = states_3b_p.size().
void GetRecoupling(std::size_t i_ch_3b, std::size_t i_ch_2b, const Operator &Z,
                   const std::vector<std::size_t> &states_3b_p,
                   const std::vector<std::size_t> &states_3b_q,
                   const std::vector<std::size_t> &states_3b_r,
                   std::vector<std::vector<std::size_t>> &indices,
                   std::vector<std::vector<double>> &recoupling);

// Basis of 3B states |pqr>
class ThreeBodyBasis {
public:
  // Create a basis of states |pqr> from 2B basis |pq> and 1B basis |r>
  // that is compatible with the given e3max truncation.
  //
  // Here no relationship between p, q, r is enforced beyond
  // what is already in the 2B basis.
  static ThreeBodyBasis From2BAnd1BBasis(std::size_t i_ch_3b,
                                         std::size_t i_ch_2b, const Operator &Z,
                                         const TwoBodyBasis &basis_pq,
                                         const OneBodyBasis &basis_r,
                                         int e3max);

  // Create a minimal basis of states |pqr> from 2B basis |pq> and 1B basis |r>
  // that is compatible with the given e3max truncation.
  //
  // Here p <= q <= r.
  static ThreeBodyBasis
  MinFrom2BAnd1BBasis(std::size_t i_ch_3b, std::size_t i_ch_2b,
                      const Operator &Z, const TwoBodyBasis &basis_pq,
                      const OneBodyBasis &basis_r, int e3max);

  // Construct 3B basis directly.
  //
  // Do not do this directly. Prefer using a factory method (see above).
  ThreeBodyBasis(std::size_t i_ch_3b, std::size_t i_ch_2b, const Operator &Z,
                 const std::vector<std::size_t> &pqr_states_,
                 std::size_t wrap_factor_,
                 const std::vector<double> &pqr_factors_)
      : wrap_factor(wrap_factor_), pqr_states(pqr_states_),
        pqr_factors(pqr_factors_) {
    GetRecoupling(i_ch_3b, i_ch_2b, Z, p_states, q_states, r_states,
                  pqr_me_indices, pqr_me_recoupling_factors);
  }

  // Get the basis size.
  std::size_t BasisSize() const { return pqr_states.size(); }

  // Get reference to the vector of |pqr> states in the basis.
  const std::vector<std::size_t> &GetPQRVals() const { return pqr_states; }

  // Get reference to the vector of normalization factors for  |pqr> states in
  // the basis.
  //
  // This may simply be normalization factors or alternatively also
  // occupation number factors.
  const std::vector<double> &GetPQRFactors() const { return pqr_factors; }

  // Get reference to the vector of p states corresponding to |pqr> in the
  // basis.
  const std::vector<std::size_t> &GetPVals() const { return p_states; }

  // Get reference to the vector of q states corresponding to |pqr> in the
  // basis.
  const std::vector<std::size_t> &GetQVals() const { return q_states; }

  // Get reference to the vector of r states corresponding to |pqr> in the
  // basis.
  const std::vector<std::size_t> &GetRVals() const { return r_states; }

  // Look up the local index for pqr in the basis (if it exists).
  //
  // Implementation detail:
  // If pqr is not in the basis, this will return 0.
  // You should probably not rely on this implementation detail though.
  std::size_t GetLocalIndexForPQR(std::size_t p, std::size_t q,
                                  std::size_t r) const {
    return pqr_indices[p * wrap_factor * wrap_factor + q * wrap_factor + r];
  }

  // Look up whether pqr is in the basis.
  int GetLocalValidityForPQR(std::size_t p, std::size_t q,
                             std::size_t r) const {
    return pqr_validities[p * wrap_factor * wrap_factor + q * wrap_factor + r];
  }

  // Get reference to recoupling indices for state |pqr> at index i_pqr.
  const std::vector<std::size_t> &
  GetRecouplingIndices(std::size_t i_pqr) const {
    return pqr_me_indices[i_pqr];
  }

  // Get reference to recoupling factors for state |pqr> at index i_pqr.
  const std::vector<double> &GetRecouplingFactors(std::size_t i_pqr) const {
    return pqr_me_recoupling_factors[i_pqr];
  }

  // Get size of 3B basis in memory.
  std::size_t NumBytes() const;

private:
  std::size_t wrap_factor;
  std::vector<std::size_t> pqr_states;
  std::vector<double> pqr_factors;
  // The following members are automatically constructed in the right way.
  // Please do not overwrite this initialization in the constructor.
  std::vector<std::size_t> p_states = Get3BPStates(pqr_states, wrap_factor);
  std::vector<std::size_t> q_states = Get3BQStates(pqr_states, wrap_factor);
  std::vector<std::size_t> r_states = Get3BRStates(pqr_states, wrap_factor);
  std::vector<std::size_t> pqr_indices =
      Get3BPQRIndices(pqr_states, wrap_factor);
  std::vector<int> pqr_validities = Get3BPQRValidities(pqr_states, wrap_factor);
  // Any constructor should call GetRecoupling() on these members
  // to populate them with the correct factors.
  std::vector<std::vector<std::size_t>> pqr_me_indices =
      std::vector<std::vector<std::size_t>>(pqr_states.size());
  std::vector<std::vector<double>> pqr_me_recoupling_factors =
      std::vector<std::vector<double>>(pqr_states.size());
};

// Methods for tensors

// Construct flattened 2D index from p, q, and dim_q.
inline std::size_t Index2D(std::size_t p, std::size_t q, std::size_t dim_q) {
  return p * dim_q + q;
}

// Construct flattened 3D index from p, q, r, dim_q, and dim_r.
inline std::size_t Index3D(std::size_t p, std::size_t q, std::size_t r,
                           std::size_t dim_q, std::size_t dim_r) {
  return p * dim_q * dim_r + q * dim_r + r;
}

// Construct flattened 4D index from p, q, r, s, dim_q, dim_r, and dim_s.
inline std::size_t Index4D(std::size_t p, std::size_t q, std::size_t r,
                           std::size_t s, std::size_t dim_q, std::size_t dim_r,
                           std::size_t dim_s) {
  return p * dim_q * dim_r * dim_s + q * dim_r * dim_s + r * dim_s + s;
}

} // namespace internal
} // namespace comm331

#endif // COMMUTATOR331_H_
