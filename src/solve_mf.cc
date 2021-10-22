/////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////
///                                                  ____ ///
///        _________________           _____________/   /\ _________________ ///
///       /____/_____/_____/|         /____/_____/ /___/  \ /____/_____/_____/|
///       ///
///      /____/_____/__G_ /||        /____/_____/|/   /\  /\ /____/_____/____
///      /||       ///
///     /____/_____/__+__/|||       /____/_____/|/ G /  \/  \
///     /____/_____/_____/|||       ///
///    |     |     |     ||||      |     |     |/___/   /\  /\       |     | |
///    ||||       /// |  I  |  M  |     ||/|      |  I  |  M  /   /\  /  \/  \
///    |  I  |  M  |     ||/|       ///
///    |_____|_____|_____|/||      |_____|____/ + /  \/   /\  /
///    |_____|_____|_____|/||       /// |     |     |     ||||      |     |
///    /___/   /\  /  \/       |     |     |     ||||       /// |  S  |  R  |
///    ||/|      |  S  |   \   \  /  \/   /        |  S  |  R  |  G  ||/| ///
///    |_____|_____|_____|/||      |_____|____\ __\/   /\  /
///    |_____|_____|_____|/||       /// |     |     |     ||||      |     | \ \
///    /  \/          |     |     |     ||||       /// |     |  +  |     ||/ |
///    |  +  |\ __\/   /           |     |  +  |  +  ||/        ///
///    |_____|_____|_____|/        |_____|_____|/\   \  / |_____|_____|_____|/
///    ///
///                                               \___\/ ///
///                                                                                               ///
///           imsrg++ : Interface for performing standard IMSRG calculations.
///           ///
///                     Usage is imsrg++  option1=value1 option2=value2 ... ///
///                     To get a list of options, type imsrg++ help ///
///                                                                                               ///
///                                                      - Ragnar Stroberg 2016
///                                                      ///
///                                                                                               ///
/////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////
//    imsrg++.cc, part of  imsrg++
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

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include "IMSRG.hh"
#include "Parameters.hh"
#include "PhysicalConstants.hh"
#include "solve_mf_params.h"
#include "version.hh"

struct OpFromFile {
  std::string file2name, file3name, opname;
  int j, p, t, r;  // J rank, parity, dTz, particle rank
};

int main(int argc, char** argv) {
  // Default parameters, and everything passed by command line args.
#ifdef BUILDVERSION
  std::cout << "######  imsrg++ build version: " << BUILDVERSION << std::endl;
#endif

  auto args = ParseMFSolverArgs(argc, argv);

  std::cout << PrettyPrintMFSolverArgs(args);

  // Parameters parameters(argc,argv);
  // if (parameters.help_mode) return 0;

  std::string inputtbme = args.path_to_input_2bme;
  std::string fmt2 = args.input_2bme_fmt;
  int file2e1max = args.input_2bme_emax;
  int file2e2max = args.input_2bme_e2max;
  int file2lmax = args.input_2bme_lmax;

  std::string input3bme = args.path_to_input_3bme;
  std::string input3bme_type = args.input_3bme_type;
  std::string fmt3 = args.input_3bme_fmt;
  int file3e1max = args.input_3bme_emax;
  int file3e2max = args.input_3bme_e2max;
  int file3e3max = args.input_3bme_e3max;

  double hw = args.hbar_omega;

  std::string outfile_1b = args.path_to_output_1bme;
  std::string outfile_2b = args.path_to_output_2bme;
  int outemax = args.output_me_emax;

  bool with_generator = args.with_generator;
  bool with_commutators = args.with_commutators;
  std::string generator = args.generator;
  std::string denominator = args.denominator;

  std::string reference = args.reference_state;
  int targetMass = args.mass;
  std::string basis = args.basis;
  int eMax = args.calc_emax;
  int lmax = args.calc_lmax;  // so far I only use this with atomic systems.
  int E3max = args.calc_e3max;
  int lmax3 = args.calc_lmax3;

  std::string intfile = args.path_to_metadata_file;
  std::string LECs = args.lec_string;

  bool nucleon_mass_correction = args.nucleon_mass_correction;
  bool relativistic_correction = args.relativistic_correction;

  double BetaCM = args.beta_cm;
  double hwBetaCM = args.hbar_omega_beta_cm;

  std::string no2b_precision = args.no2b_precision;
  std::string valence_space = args.valence_space;
  std::string NAT_order = args.nat_order;
  bool freeze_occupations = args.freeze_occupations;
  bool discard_no2b_from_3n = args.discard_no2b_from_3n;
  bool use_NAT_occupations = args.use_nat_occupations;
  int emax_unocc = args.emax_unoccupied;

  std::ofstream meta_file(intfile);
  if (!meta_file.good()) {
    std::cout << "Could not open metadata file: " << intfile << "\n";
    exit(EXIT_FAILURE);
  }
  time_t time_now =
      std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  meta_file << "SOLVE_MF run at " << ctime(&time_now) << "\n";
  meta_file << PrettyPrintMFSolverArgs(args) << "\n";

  // Test 2bme file
  if (inputtbme != "none" and fmt2.find("oakridge") == std::string::npos and
      fmt2 != "schematic") {
    if (not std::ifstream(inputtbme).good()) {
      std::cout << "trouble reading " << inputtbme << "  fmt2 = " << fmt2
                << "   exiting. " << std::endl;
      return 1;
    }
  }
  // Test 3bme file
  if (input3bme != "none") {
    if (not std::ifstream(input3bme).good()) {
      std::cout << "trouble reading " << input3bme << " exiting. " << std::endl;
      return 1;
    }
  }

  ReadWrite rw;
  rw.SetLECs_preset(LECs);
  rw.Set3NFormat(fmt3);

  //  ModelSpace modelspace;

  ModelSpace modelspace =
      (reference == "default" ? ModelSpace(eMax, valence_space)
                              : ModelSpace(eMax, reference, valence_space));

  //  std::cout << __LINE__ << "  constructed modelspace " << std::endl;
  modelspace.SetE3max(E3max);
  modelspace.SetLmax(lmax);
  //  std::cout << __LINE__ << "  done setting E3max and lmax " << std::endl;

  if (emax_unocc > 0) {
    modelspace.SetEmaxUnocc(emax_unocc);
  }

  modelspace.SetHbarOmega(hw);
  if (targetMass > 0) modelspace.SetTargetMass(targetMass);
  if (lmax3 > 0) modelspace.SetLmax3(lmax3);

  //  std::cout << "Making the Hamiltonian..." << std::endl;
  int particle_rank = input3bme == "none" ? 2 : 3;
  Operator Hbare = Operator(modelspace, 0, 0, 0, particle_rank);
  Hbare.SetHermitian();

  std::cout << "Reading interactions..." << std::endl;

  if (inputtbme != "none") {
    if (fmt2 == "me2j")
      rw.ReadBareTBME_Darmstadt(inputtbme, Hbare, file2e1max, file2e2max,
                                file2lmax);
    else if (fmt2 == "navratil" or fmt2 == "Navratil")
      rw.ReadBareTBME_Navratil(inputtbme, Hbare);
    else if (fmt2 == "oslo")
      rw.ReadTBME_Oslo(inputtbme, Hbare);
    else if (fmt2.find("oakridge") !=
             std::string::npos) {  // input format should be:
                                   // singleparticle.dat,vnn.dat
      size_t comma_pos = inputtbme.find_first_of(",");
      if (fmt2.find("bin") != std::string::npos)
        rw.ReadTBME_OakRidge(inputtbme.substr(0, comma_pos),
                             inputtbme.substr(comma_pos + 1), Hbare, "binary");
      else
        rw.ReadTBME_OakRidge(inputtbme.substr(0, comma_pos),
                             inputtbme.substr(comma_pos + 1), Hbare, "ascii");
    } else if (fmt2 == "takayuki")
      rw.ReadTwoBody_Takayuki(inputtbme, Hbare);
    else if (fmt2 == "nushellx")
      rw.ReadNuShellX_int(Hbare, inputtbme);
    else if (fmt2 == "schematic") {
      std::cout << "using schematic potential " << inputtbme << std::endl;
      if (inputtbme == "Minnesota")
        Hbare += imsrg_util::MinnesotaPotential(modelspace);
    }

    std::cout << "done reading 2N" << std::endl;
  }

  // Read in the 3-body file
  if (Hbare.particle_rank >= 3) {
    if (input3bme_type == "full") {
      rw.Read_Darmstadt_3body(input3bme, Hbare, file3e1max, file3e2max,
                              file3e3max);
    }
    if (input3bme_type == "no2b") {
      Hbare.ThreeBody.SetMode("no2b");
      if (no2b_precision == "half") Hbare.ThreeBody.SetMode("no2bhalf");

      Hbare.ThreeBody.ReadFile(
          {input3bme}, {file3e1max, file3e2max, file3e3max, file3e1max});
      rw.File3N = input3bme;

    } else if (input3bme_type == "mono") {
      Hbare.ThreeBody.SetMode("mono");
      Hbare.ThreeBody.ReadFile(
          {input3bme}, {file3e1max, file3e2max, file3e3max, file3e1max});
      rw.File3N = input3bme;
    }
    std::cout << "done reading 3N" << std::endl;
  }

  if (fmt2 != "nushellx")  // Don't need to add kinetic energy if we read a
                           // shell model interaction
  {
    Hbare += imsrg_util::Trel_Op(modelspace);
    if (Hbare.OneBody.has_nan()) {
      std::cout << "  Looks like the Trel op is hosed from the get go. Dying."
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  // correction to kinetic energy because M_proton != M_neutron
  if (nucleon_mass_correction) {
    Hbare += imsrg_util::Trel_Masscorrection_Op(modelspace);
  }

  if (relativistic_correction) {
    Hbare += imsrg_util::KineticEnergy_RelativisticCorr(modelspace);
  }

  // Add a Lawson center of mass term. If hwBetaCM is specified, use that
  // frequency, otherwise use the basis frequency
  if (std::abs(BetaCM) > 1e-3) {
    if (hwBetaCM < 0) hwBetaCM = modelspace.GetHbarOmega();
    std::ostringstream hcm_opname;
    hcm_opname << "HCM_" << hwBetaCM;
    Hbare +=
        BetaCM * imsrg_util::OperatorFromString(modelspace, hcm_opname.str());
  }

  std::cout << "Creating HF" << std::endl;
  HFMBPT hf(Hbare);  // HFMBPT inherits from HartreeFock, so this works for HF
                     // and NAT bases.

  if (not freeze_occupations) hf.UnFreezeOccupations();
  if (discard_no2b_from_3n) hf.DiscardNO2Bfrom3N();
  std::cout << "Solving" << std::endl;

  if (basis != "oscillator") {
    hf.Solve();
  }

  // decide what to keep after normal ordering
  int hno_particle_rank = 2;

  Operator& HNO =
      Hbare;  // The reference & means we overwrite Hbare and save some memory
  if (basis == "HF") {
    HNO = hf.GetNormalOrderedH(hno_particle_rank);
  } else if (basis == "NAT")  // we want to use the natural orbital basis
  {
    hf.UseNATOccupations(use_NAT_occupations);
    hf.OrderNATBy(NAT_order);

    hf.GetNaturalOrbitals();
    HNO = hf.GetNormalOrderedHNAT(hno_particle_rank);

  } else if (basis == "oscillator") {
    HNO = Hbare.DoNormalOrdering();
  }

  rw.Write_me1j(outfile_1b, HNO, outemax, outemax);
  auto no_ext_filename = outfile_2b.substr(0, outfile_2b.find_last_of("."));
  rw.Write_me2jp(no_ext_filename + ".me2jp", HNO, outemax, 2 * outemax,
                 outemax);
  rw.Write_me2j_np(no_ext_filename + ".me2j_np", HNO, outemax, 2 * outemax,
                   outemax);

  if (with_generator || with_commutators) {
    // Create generator
    Operator eta(HNO);
    eta.Erase();
    eta.SetAntiHermitian();

    Generator gen;
    gen.SetType(generator);
    gen.SetDenominatorPartitioning(denominator);
    gen.Update(&HNO, &eta);

    std::string gp_str = "_" + generator + "_" + denominator;

    if (with_generator) {
      // Write generator
      rw.Write_me1j(no_ext_filename + gp_str + "_gen1.me1j", eta, outemax,
                    outemax);
      rw.Write_me2jp(no_ext_filename + gp_str + "_gen2.me2jp", eta, outemax,
                     2 * outemax, outemax);
      rw.Write_me2j_np(no_ext_filename + gp_str + "_gen2.me2j_np", eta, outemax,
                       2 * outemax, outemax);
    }

    if (with_commutators) {
      // Isolate 1- and 2-body parts
      Operator gen1(eta);
      gen1.EraseTwoBody();
      gen1.EraseZeroBody();
      Operator gen2(eta);
      gen2.EraseOneBody();
      gen2.EraseZeroBody();
      Operator ham1(HNO);
      ham1.EraseTwoBody();
      ham1.EraseZeroBody();
      Operator ham2(HNO);
      ham2.EraseOneBody();
      ham2.EraseZeroBody();
      {
        auto comm_gh_11x = Commutator::Commutator(gen1, ham1);
        rw.Write_me1j(no_ext_filename + gp_str + "_comm11X_gen1_ham1.me1j",
                      comm_gh_11x, outemax, outemax);
        rw.Write_me2jp(no_ext_filename + gp_str + "_comm11X_gen1_ham1.me2jp",
                       comm_gh_11x, outemax, 2 * outemax, outemax);
        rw.Write_me2j_np(
            no_ext_filename + gp_str + "_comm11X_gen1_ham1.me2j_np",
            comm_gh_11x, outemax, 2 * outemax, outemax);
      }
      {
        auto comm_hg_11x = Commutator::Commutator(ham1, gen1);
        rw.Write_me1j(no_ext_filename + gp_str + "_comm11X_ham1_gen1.me1j",
                      comm_hg_11x, outemax, outemax);
        rw.Write_me2jp(no_ext_filename + gp_str + "_comm11X_ham1_gen1.me2jp",
                       comm_hg_11x, outemax, 2 * outemax, outemax);
        rw.Write_me2j_np(
            no_ext_filename + gp_str + "_comm11X_ham1_gen1.me2j_np",
            comm_hg_11x, outemax, 2 * outemax, outemax);
      }
      {
        auto comm_gh_12x = Commutator::Commutator(gen1, ham2);
        rw.Write_me1j(no_ext_filename + gp_str + "_comm12X_gen1_ham2.me1j",
                      comm_gh_12x, outemax, outemax);
        rw.Write_me2jp(no_ext_filename + gp_str + "_comm12X_gen1_ham2.me2jp",
                       comm_gh_12x, outemax, 2 * outemax, outemax);
        rw.Write_me2j_np(
            no_ext_filename + gp_str + "_comm12X_gen1_ham2.me2j_np",
            comm_gh_12x, outemax, 2 * outemax, outemax);
      }
      {
        auto comm_hg_12x = Commutator::Commutator(ham1, gen2);
        rw.Write_me1j(no_ext_filename + gp_str + "_comm12X_ham1_gen2.me1j",
                      comm_hg_12x, outemax, outemax);
        rw.Write_me2jp(no_ext_filename + gp_str + "_comm12X_ham1_gen2.me2jp",
                       comm_hg_12x, outemax, 2 * outemax, outemax);
        rw.Write_me2j_np(
            no_ext_filename + gp_str + "_comm12X_ham1_gen2.me2j_np",
            comm_hg_12x, outemax, 2 * outemax, outemax);
      }
      {
        auto comm_gh_21x = Commutator::Commutator(gen2, ham1);
        rw.Write_me1j(no_ext_filename + gp_str + "_comm21X_gen2_ham1.me1j",
                      comm_gh_21x, outemax, outemax);
        rw.Write_me2jp(no_ext_filename + gp_str + "_comm21X_gen2_ham1.me2jp",
                       comm_gh_21x, outemax, 2 * outemax, outemax);
        rw.Write_me2j_np(
            no_ext_filename + gp_str + "_comm21X_gen2_ham1.me2j_np",
            comm_gh_21x, outemax, 2 * outemax, outemax);
      }
      {
        auto comm_hg_21x = Commutator::Commutator(ham2, gen1);
        rw.Write_me1j(no_ext_filename + gp_str + "_comm21X_ham2_gen1.me1j",
                      comm_hg_21x, outemax, outemax);
        rw.Write_me2jp(no_ext_filename + gp_str + "_comm21X_ham2_gen1.me2jp",
                       comm_hg_21x, outemax, 2 * outemax, outemax);
        rw.Write_me2j_np(
            no_ext_filename + gp_str + "_comm21X_ham2_gen1.me2j_np",
            comm_hg_21x, outemax, 2 * outemax, outemax);
      }
      {
        auto comm_gh_22x = Commutator::Commutator(gen2, ham2);
        rw.Write_me1j(no_ext_filename + gp_str + "_comm22X_gen2_ham2.me1j",
                      comm_gh_22x, outemax, outemax);
        rw.Write_me2jp(no_ext_filename + gp_str + "_comm22X_gen2_ham2.me2jp",
                       comm_gh_22x, outemax, 2 * outemax, outemax);
        rw.Write_me2j_np(
            no_ext_filename + gp_str + "_comm22X_gen2_ham2.me2j_np",
            comm_gh_22x, outemax, 2 * outemax, outemax);
      }
      {
        auto comm_hg_22x = Commutator::Commutator(ham2, gen2);
        rw.Write_me1j(no_ext_filename + gp_str + "_comm22X_ham2_gen2.me1j",
                      comm_hg_22x, outemax, outemax);
        rw.Write_me2jp(no_ext_filename + gp_str + "_comm22X_ham2_gen2.me2jp",
                       comm_hg_22x, outemax, 2 * outemax, outemax);
        rw.Write_me2j_np(
            no_ext_filename + gp_str + "_comm22X_ham2_gen2.me2j_np",
            comm_hg_22x, outemax, 2 * outemax, outemax);
      }
    }
  }

  return 0;
}
