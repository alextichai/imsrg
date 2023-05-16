//    Copyright (C) 2018  Ragnar Stroberg
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///
///        _________________
///       /____/_____/_____/|
///      /____/_____/__G_ /||
///     /____/_____/__+__/|||
///    |     |     |     ||||
///    |  I  |  M  |     ||/|
///    |_____|_____|_____|/||
///    |     |     |     ||||
///    |  S  |  R  |     ||/|
///    |_____|_____|_____|/||
///    |     |     |     ||||
///    |     |  +  |     ||/
///    |_____|_____|_____|/
///
///                      ____
///        _____________/   /\ 
///       /____/_____/ /___/  \ 
///      /____/_____/|/   /\  /\ 
///     /____/_____/|/ G /  \/  \ 
///    |     |     |/___/   /\  /\ 
///    |  I  |  M  /   /\  /  \/  \ 
///    |_____|____/ + /  \/   /\  /
///    |     |   /___/   /\  /  \/
///    |  S  |   \   \  /  \/   /
///    |_____|____\ __\/   /\  /
///    |     |     \   \  /  \/
///    |     |  +  |\ __\/   /
///    |_____|_____|/\   \  /
///                   \___\/
///
///        _________________
///       /____/_____/_____/|
///      /____/_____/____ /||
///     /____/_____/_____/|||
///    |     |     |     ||||
///    |  I  |  M  |     ||/|
///    |_____|_____|_____|/||
///    |     |     |     ||||
///    |  S  |  R  |  G  ||/|
///    |_____|_____|_____|/||
///    |     |     |     ||||
///    |     |  +  |  +  ||/
///    |_____|_____|_____|/
///
///
///           imsrg++ : Interface for performing standard IMSRG calculations.
///                     Usage is imsrg++  option1=value1 option2=value2 ...
///                     To get a list of options, type imsrg++ help
///
///                                                      - Ragnar Stroberg 2016
///
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

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

#include "Commutator.hh"
#include "IMSRG.hh"
#include "Parameters.hh"
#include "PerturbationTheory.hh"
#include "PhysicalConstants.hh"
#include "version.hh"

struct OpFromFile {
  std::string file2name, file3name, opname;
  int j, p, t, r;  // J rank, parity, dTz, particle rank
};

int main(int argc, char** argv) {
  // Default parameters, and everything passed by command line args.
  std::cout << "######  imsrg++ build version: " << version::BuildVersion()
            << std::endl;

  Parameters parameters(argc, argv);
  if (parameters.help_mode) return 0;

  std::string inputtbme = parameters.s("2bme");
  std::string inputtbme_NLO = parameters.s("2bme_NLO");
  std::string inputtbme_N2LO = parameters.s("2bme_N2LO");
  std::string input3bme = parameters.s("3bme");
  std::string input3bme_type = parameters.s("3bme_type");
  std::string no2b_precision = parameters.s("no2b_precision");
  std::string reference = parameters.s("reference");
  std::string valence_space = parameters.s("valence_space");
  std::string custom_valence_space = parameters.s("custom_valence_space");
  std::string basis = parameters.s("basis");
  std::string method = parameters.s("method");
  std::string flowfile = parameters.s("flowfile");
  std::string intfile = parameters.s("intfile");
  std::string core_generator = parameters.s("core_generator");
  std::string valence_generator = parameters.s("valence_generator");
  std::string fmt2 = parameters.s("fmt2");
  std::string fmt3 = parameters.s("fmt3");
  std::string scratch = parameters.s("scratch");
  std::string valence_file_format = parameters.s("valence_file_format");
  std::string denominator_partitioning =
      parameters.s("denominator_partitioning");
  std::string NAT_order = parameters.s("NAT_order");

  bool IMSRG3 = parameters.s("IMSRG3") == "true";
  bool imsrg3_n7 = parameters.s("imsrg3_n7") == "true";
  bool reduced_232_impl = parameters.s("reduced_232_impl") == "true";
  bool imsrg3_mp4 = parameters.s("imsrg3_mp4") == "true";
  bool write_omega = parameters.s("write_omega") == "true";
  bool discard_no2b_from_3n = parameters.s("discard_no2b_from_3n") == "true";
  bool discard_residual_input3N =
      parameters.s("discard_residual_input3N") == "true";

  int eMax = parameters.i("emax");
  int E3max = parameters.i("e3max");
  int targetMass = parameters.i("A");
  int nsteps = parameters.i("nsteps");
  int file2e1max = parameters.i("file2e1max");
  int file2e2max = parameters.i("file2e2max");
  int file2lmax = parameters.i("file2lmax");
  int file3e1max = parameters.i("file3e1max");
  int file3e2max = parameters.i("file3e2max");
  int file3e3max = parameters.i("file3e3max");
  int eMax_imsrg = parameters.i("emax_imsrg");
  int e2Max_imsrg = parameters.i("e2max_imsrg");
  int e3Max_imsrg = parameters.i("e3max_imsrg");
  int eMax_3body_imsrg = parameters.i("emax_3body_imsrg");

  double hw = parameters.d("hw");
  double smax = parameters.d("smax");
  double dsmax = parameters.d("dsmax");
  double ds_0 = parameters.d("ds_0");
  double domega = parameters.d("domega");
  double omega_norm_max = parameters.d("omega_norm_max");
  double eta_criterion = parameters.d("eta_criterion");

  std::vector<std::string> opnames = parameters.v("Operators");

  std::vector<Operator> ops;

  using PhysConst::DARWIN_FOLDY;
  using PhysConst::NEUTRON_RCH2;
  using PhysConst::PROTON_RCH2;

  // test 2bme file
  if ((inputtbme != "none") && (fmt2.find("oakridge") == std::string::npos) &&
      (fmt2 != "schematic")) {
    if (!std::ifstream(inputtbme).good()) {
      std::cout << "trouble reading " << inputtbme << "  fmt2 = " << fmt2
                << "   exiting. " << std::endl;
      return 1;
    }
    if ((inputtbme_NLO != "none") && !std::ifstream(inputtbme_NLO).good()) {
      std::cout << "trouble reading " << inputtbme_NLO << "  fmt2 = " << fmt2
                << "   exiting. " << std::endl;
      return 1;
    }
    if ((inputtbme_N2LO != "none") && !std::ifstream(inputtbme_N2LO).good()) {
      std::cout << "trouble reading " << inputtbme_N2LO << "  fmt2 = " << fmt2
                << "   exiting. " << std::endl;
      return 1;
    }
  }
  // test 3bme file
  if (input3bme != "none") {
    if (!std::ifstream(input3bme).good()) {
      std::cout << "trouble reading " << input3bme << " exiting. " << std::endl;
      return 1;
    }
  }

  ReadWrite rw;
  rw.SetScratchDir(scratch);
  rw.Set3NFormat(fmt3);

  // Test whether the scratch directory exists and we can write to it.
  // This is necessary because otherwise you get garbage for transformed
  // operators and it's not obvious what went wrong.
  if ((method == "magnus") && (opnames.size() > 0)) {
    if ((scratch == "/dev/null") || (scratch == "/dev/null/")) {
      std::cout
          << "ERROR!!! using Magnus with scratch = " << scratch
          << " but you're also trying to transform some operators. Dying now. "
          << std::endl;
      exit(EXIT_FAILURE);
    } else if (scratch != "") {
      std::string testfilename = scratch + "/_this_is_a_test_delete_me";
      std::ofstream testout(testfilename);
      testout << "PASSED" << std::endl;
      testout.close();

      // now read it back.
      std::ifstream testin(testfilename);
      std::string checkpassed;
      testin >> checkpassed;
      if ((checkpassed != "PASSED") || (!testout.good()) || (!testin.good())) {
        std::cout << "ERROR in " << __FILE__
                  << " failed test write to scratch directory " << scratch
                  << " that's bad. Dying now." << std::endl;
        exit(EXIT_FAILURE);
      }
    }
  }

  //  ModelSpace modelspace;

  // if a custom space is defined, the input
  // valence_space is just used as a name
  if (custom_valence_space != "") {
    // if no name is given, then just name it "custom"
    if (valence_space == "") {
      parameters.string_par["valence_space"] = "custom";
      flowfile = parameters.DefaultFlowFile();
      intfile = parameters.DefaultIntFile();
    }
    valence_space = custom_valence_space;
  }

  ModelSpace modelspace =
      (reference == "default" ? ModelSpace(eMax, valence_space)
                              : ModelSpace(eMax, reference, valence_space));

  //  std::cout << __LINE__ << "  constructed modelspace " << std::endl;
  modelspace.SetE3max(E3max);
  //  std::cout << __LINE__ << "  done setting E3max and lmax " << std::endl;

  // default to 1 step for single ref, 2 steps for valence decoupling
  if (nsteps < 0) {
    nsteps = modelspace.valence.size() > 0 ? 2 : 1;
  }

  modelspace.SetHbarOmega(hw);
  if (targetMass > 0) modelspace.SetTargetMass(targetMass);

  if (std::find(opnames.begin(), opnames.end(), "rhop_all") != opnames.end()) {
    opnames.erase(
        std::remove(opnames.begin(), opnames.end(), "rhop_all"),
        std::end(opnames));
    for (double r = 0.0; r <= 10.0; r += 0.2) {
      std::ostringstream opn;
      opn << "rhop_" << r;
      opnames.push_back(opn.str());
    }
  }

  if (std::find(opnames.begin(), opnames.end(), "rhon_all") != opnames.end()) {
    opnames.erase(
        std::remove(opnames.begin(), opnames.end(), "rhon_all"),
        std::end(opnames));
    for (double r = 0.0; r <= 10.0; r += 0.2) {
      std::ostringstream opn;
      opn << "rhon_" << r;
      opnames.push_back(opn.str());
    }
  }

  if (std::find(opnames.begin(), opnames.end(), "DaggerHF_valence") !=
      opnames.end()) {
    opnames.erase(
        std::remove(opnames.begin(), opnames.end(), "DaggerHF_valence"),
        std::end(opnames));
    for (auto v : modelspace.valence) {
      opnames.push_back("DaggerHF_" + modelspace.Index2String(v));
    }
    std::cout
        << "I found DaggerHF_valence, so I'm changing the opnames list to :"
        << std::endl;
    for (auto opn : opnames) std::cout << opn << " ,  ";
    std::cout << std::endl;
  }

  if (std::find(opnames.begin(), opnames.end(), "DaggerAlln_valence") !=
      opnames.end()) {
    opnames.erase(
        std::remove(opnames.begin(), opnames.end(), "DaggerAlln_valence"),
        std::end(opnames));
    for (auto v : modelspace.valence) {
      opnames.push_back("DaggerAlln_" + modelspace.Index2String(v));
    }
    std::cout
        << "I found DaggerAlln_valence, so I'm changing the opnames list to :"
        << std::endl;
    for (auto opn : opnames) std::cout << opn << " ,  ";
    std::cout << std::endl;
  }

  //  std::cout << "Making the Hamiltonian..." << std::endl;
  int particle_rank = input3bme == "none" ? 2 : 3;
  Operator Hbare = Operator(modelspace, 0, 0, 0, particle_rank);
  Hbare.SetHermitian();
  Operator Hbare_NLO = Operator(modelspace, 0, 0, 0, particle_rank);
  Hbare_NLO.SetHermitian();
  Operator Hbare_N2LO = Operator(modelspace, 0, 0, 0, particle_rank);
  Hbare_N2LO.SetHermitian();

  std::cout << "Reading interactions..." << std::endl;

  if (inputtbme != "none") {
    if (fmt2 == "me2j") {
      rw.ReadBareTBME_Darmstadt(
          inputtbme,
          Hbare,
          file2e1max,
          file2e2max,
          file2lmax);
    } else if ((fmt2 == "navratil") || (fmt2 == "Navratil")) {
      rw.ReadBareTBME_Navratil(inputtbme, Hbare);
    } else if (fmt2 == "oslo") {
      rw.ReadTBME_Oslo(inputtbme, Hbare);
    } else if (fmt2.find("oakridge") != std::string::npos) {
      // input format should be:
      // singleparticle.dat,vnn.dat
      size_t comma_pos = inputtbme.find_first_of(",");
      if (fmt2.find("bin") != std::string::npos)
        rw.ReadTBME_OakRidge(
            inputtbme.substr(0, comma_pos),
            inputtbme.substr(comma_pos + 1),
            Hbare,
            "binary");
      else
        rw.ReadTBME_OakRidge(
            inputtbme.substr(0, comma_pos),
            inputtbme.substr(comma_pos + 1),
            Hbare,
            "ascii");
    } else if (fmt2 == "takayuki") {
      rw.ReadTwoBody_Takayuki(inputtbme, Hbare);
    } else if (fmt2 == "nushellx") {
      rw.ReadNuShellX_int(Hbare, inputtbme);
    } else if (fmt2 == "schematic") {
      std::cout << "using schematic potential " << inputtbme << std::endl;
      if (inputtbme == "Minnesota")
        Hbare += imsrg_util::MinnesotaPotential(modelspace);
    }

    std::cout << "done reading 2N" << std::endl;
  }

  if (inputtbme_NLO != "none") {
    if (fmt2 == "me2j") {
      rw.ReadBareTBME_Darmstadt(
          inputtbme_NLO,
          Hbare_NLO,
          file2e1max,
          file2e2max,
          file2lmax);
    } else if ((fmt2 == "navratil") || (fmt2 == "Navratil")) {
      rw.ReadBareTBME_Navratil(inputtbme_NLO, Hbare_NLO);
    } else if (fmt2 == "oslo") {
      rw.ReadTBME_Oslo(inputtbme_NLO, Hbare_NLO);
    } else if (fmt2.find("oakridge") != std::string::npos) {
      // input format should be:
      // singleparticle.dat,vnn.dat
      size_t comma_pos = inputtbme_NLO.find_first_of(",");
      if (fmt2.find("bin") != std::string::npos)
        rw.ReadTBME_OakRidge(
            inputtbme_NLO.substr(0, comma_pos),
            inputtbme_NLO.substr(comma_pos + 1),
            Hbare_NLO,
            "binary");
      else
        rw.ReadTBME_OakRidge(
            inputtbme_NLO.substr(0, comma_pos),
            inputtbme_NLO.substr(comma_pos + 1),
            Hbare_NLO,
            "ascii");
    } else if (fmt2 == "takayuki") {
      rw.ReadTwoBody_Takayuki(inputtbme_NLO, Hbare_NLO);
    } else if (fmt2 == "nushellx") {
      rw.ReadNuShellX_int(Hbare_NLO, inputtbme_NLO);
    } else if (fmt2 == "schematic") {
      std::cout << "using schematic potential " << inputtbme_NLO << std::endl;
      if (inputtbme_NLO == "Minnesota")
        Hbare_NLO += imsrg_util::MinnesotaPotential(modelspace);
    }

    std::cout << "done reading 2N (NLO)" << std::endl;
  }

  if (inputtbme_N2LO != "none") {
    if (fmt2 == "me2j") {
      rw.ReadBareTBME_Darmstadt(
          inputtbme_N2LO,
          Hbare_N2LO,
          file2e1max,
          file2e2max,
          file2lmax);
    } else if ((fmt2 == "navratil") || (fmt2 == "Navratil")) {
      rw.ReadBareTBME_Navratil(inputtbme_N2LO, Hbare_N2LO);
    } else if (fmt2 == "oslo") {
      rw.ReadTBME_Oslo(inputtbme_N2LO, Hbare_N2LO);
    } else if (fmt2.find("oakridge") != std::string::npos) {
      // input format should be:
      // singleparticle.dat,vnn.dat
      size_t comma_pos = inputtbme_N2LO.find_first_of(",");
      if (fmt2.find("bin") != std::string::npos)
        rw.ReadTBME_OakRidge(
            inputtbme_N2LO.substr(0, comma_pos),
            inputtbme_N2LO.substr(comma_pos + 1),
            Hbare_N2LO,
            "binary");
      else
        rw.ReadTBME_OakRidge(
            inputtbme_N2LO.substr(0, comma_pos),
            inputtbme_N2LO.substr(comma_pos + 1),
            Hbare_N2LO,
            "ascii");
    } else if (fmt2 == "takayuki") {
      rw.ReadTwoBody_Takayuki(inputtbme_N2LO, Hbare_N2LO);
    } else if (fmt2 == "nushellx") {
      rw.ReadNuShellX_int(Hbare_N2LO, inputtbme_N2LO);
    } else if (fmt2 == "schematic") {
      std::cout << "using schematic potential " << inputtbme_N2LO << std::endl;
      if (inputtbme_N2LO == "Minnesota")
        Hbare_N2LO += imsrg_util::MinnesotaPotential(modelspace);
    }

    std::cout << "done reading 2N (N2LO)" << std::endl;
  }

  // Read in the 3-body file
  if (Hbare.particle_rank >= 3) {
    if (input3bme_type == "full") {
      rw.Read_Darmstadt_3body(
          input3bme,
          Hbare,
          file3e1max,
          file3e2max,
          file3e3max);
    }
    if (input3bme_type == "no2b") {
      Hbare.ThreeBody.SetMode("no2b");
      if (no2b_precision == "half") Hbare.ThreeBody.SetMode("no2bhalf");

      Hbare.ThreeBody.ReadFile(
          {input3bme},
          {file3e1max, file3e2max, file3e3max, file3e1max});
      rw.File3N = input3bme;

    } else if (input3bme_type == "mono") {
      Hbare.ThreeBody.SetMode("mono");
      Hbare.ThreeBody.ReadFile(
          {input3bme},
          {file3e1max, file3e2max, file3e3max, file3e1max});
      rw.File3N = input3bme;
    }
    std::cout << "done reading 3N" << std::endl;
  }

  // Construct Trel and add to Hamiltonian
  Hbare += imsrg_util::Trel_Op(modelspace);
  if (Hbare.OneBody.has_nan()) {
    std::cout << "  Looks like the Trel op is hosed from the get go. Dying."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  std::cout << "Creating HF" << std::endl;
  // HFMBPT inherits from HartreeFock, so this works for HF
  // and NAT bases.
  HFMBPT hf(Hbare);

  hf.UnFreezeOccupations();
  if (discard_no2b_from_3n) hf.DiscardNO2Bfrom3N();
  std::cout << "Solving" << std::endl;
  if (basis != "oscillator") hf.Solve();

  // decide what to keep after normal ordering
  int hno_particle_rank = 2;
  if ((IMSRG3) && (Hbare.ThreeBodyNorm() > 1e-5)) hno_particle_rank = 3;
  if (discard_residual_input3N) hno_particle_rank = 2;
  if (input3bme_type == "no2b") hno_particle_rank = 2;

  // The reference & means we overwrite Hbare and save some memory
  Operator& HNO = Hbare;
  if ((basis == "HF") && (method != "HF")) {
    HNO = hf.GetNormalOrderedH(hno_particle_rank);
  } else if (basis == "NAT") {
    // we want to use the natural orbital basis
    hf.OrderNATBy(NAT_order);

    //  GetNaturalOrbitals() calls GetDensityMatrix(), which computes the 1b
    //  density matrix up to MBPT2 using the NO2B Hamiltonian in the HF basis,
    //  obtained with GetNormalOrderedH(). Then it calls DiagonalizeRho() which
    //  diagonalizes the density matrix, yielding the natural orbital basis.
    hf.GetNaturalOrbitals();
    HNO = hf.GetNormalOrderedHNAT(hno_particle_rank);

    //  SRS: I'm commenting this out because this is not reasonably-expected
    //  default behavior
    //    // For now, even if we use the NAT occupations, we switch back to
    //    naive occupations after the normal ordering
    //    // This should be investigated in more detail.
    //    if (use_NAT_occupations)
    //    {
    //      hf.FillLowestOrbits();
    //      std::cout << "Undoing NO wrt A=" << modelspace.GetAref() << " Z=" <<
    //      modelspace.GetZref() << std::endl; HNO = HNO.UndoNormalOrdering();
    //      hf.UpdateReference();
    //      modelspace.SetReference(modelspace.core); // change the reference
    //      std::cout << "Doing NO wrt A=" << modelspace.GetAref() << " Z=" <<
    //      modelspace.GetZref() << std::endl; HNO = HNO.DoNormalOrdering();
    //    }

  } else if (basis == "oscillator") {
    HNO = Hbare.DoNormalOrdering();
  }

  if (IMSRG3) {
    std::array<size_t, 2> nstates = modelspace.CountThreeBodyStatesInsideCut();
    std::cout << "You have chosen IMSRG3. good luck..." << std::endl;
    std::cout << "Truncations: dE3max = NONE "
              << "   OccNat3Cut = NONE "
              << "  ->  number of 3-body states kept:  " << nstates[0]
              << " out of " << nstates[1] << std::endl;

    if (hno_particle_rank < 3) {
      // if we're doing IMSRG3, we need a 3 body operator
      Operator H3(modelspace, 0, 0, 0, 3);
      std::cout << "Constructed H3" << std::endl;
      H3.ZeroBody = HNO.ZeroBody;
      H3.OneBody = HNO.OneBody;
      H3.TwoBody = HNO.TwoBody;
      HNO = H3;
      std::cout << "Replacing HNO" << std::endl;
      std::cout << "Hbare Three Body Norm is " << Hbare.ThreeBodyNorm()
                << std::endl;
    }
  }

  std::cout << "Hbare 0b = " << std::setprecision(8) << HNO.ZeroBody
            << std::endl;

  if (method != "HF") {
    std::cout << "Perturbative estimates of gs energy:" << std::endl;
    double EMP2 = HNO.GetMP2_Energy();
    double EMP2_3B = HNO.GetMP2_3BEnergy();
    std::cout << "EMP2 = " << EMP2 << std::endl;
    std::cout << "EMP2_3B = " << EMP2_3B << std::endl;
    std::array<double, 3> Emp_3 = HNO.GetMP3_Energy();
    double EMP3 = Emp_3[0] + Emp_3[1] + Emp_3[2];
    std::cout << "E3_pp = " << Emp_3[0] << "  E3_hh = " << Emp_3[1]
              << " E3_ph = " << Emp_3[2] << "   EMP3 = " << EMP3 << std::endl;
    std::cout << "To 3rd order, E = " << HNO.ZeroBody + EMP2 + EMP3 + EMP2_3B
              << std::endl;
  }

  std::cout << "done with pert stuff, method = " << method << std::endl;

  std::cout << basis
            << " Single particle energies and wave functions:" << std::endl;
  hf.PrintSPEandWF();
  std::cout << std::endl;

  if (method == "HF") {
    HNO.PrintTimes();
    return 0;
  }

  // We may want to use a smaller model space for the IMSRG evolution than we
  // used for the HF step. This is most effective when using natural orbitals or
  // when including 3-body operators.
  //  ModelSpace modelspace_imsrg = ( reference=="default" ?
  //  ModelSpace(eMax_imsrg,e2Max_imsrg,e3Max_imsrg,valence_space) :
  //  ModelSpace(eMax_imsrg,e2Max_imsrg,e3Max_imsrg,reference,valence_space) );
  ModelSpace modelspace_imsrg = modelspace;
  if ((eMax_imsrg != -1) || (e2Max_imsrg != -1) || (e3Max_imsrg != -1) ||
      (eMax_3body_imsrg != -1)) {
    if (eMax_imsrg == -1) eMax_imsrg = eMax;
    if (e2Max_imsrg == -1) e2Max_imsrg = 2 * eMax_imsrg;
    if (e3Max_imsrg == -1) e3Max_imsrg = std::min(E3max, 3 * eMax_imsrg);
    if (eMax_3body_imsrg == -1) eMax_3body_imsrg = eMax_imsrg;

    //     ModelSpace modelspace_imsrg = modelspace;
    std::cout
        << "Truncating modelspace for IMSRG calculation: emax e2max e3max  ->  "
        << eMax_imsrg << " " << e2Max_imsrg << " " << e3Max_imsrg << std::endl;
    modelspace_imsrg.SetEmax(eMax_imsrg);
    modelspace_imsrg.SetE2max(e2Max_imsrg);
    modelspace_imsrg.SetE3max(e3Max_imsrg);
    modelspace_imsrg.SetEmax3Body(eMax_3body_imsrg);
    modelspace_imsrg.Init(eMax_imsrg, reference, valence_space);

    // If the occupations in modelspace were different from the naive filling,
    // we want to keep those.
    std::map<index_t, double> hole_map;
    for (auto& i_new : modelspace_imsrg.all_orbits) {
      Orbit& oi_new = modelspace_imsrg.GetOrbit(i_new);
      index_t i_old =
          modelspace.GetOrbitIndex(oi_new.n, oi_new.l, oi_new.j2, oi_new.tz2);
      Orbit& oi_old = modelspace.GetOrbit(i_old);
      hole_map[i_new] = oi_old.occ;
    }
    modelspace_imsrg.SetReference(hole_map);

    HNO = HNO.Truncate(modelspace_imsrg);
    if (IMSRG3) {
      HNO.ThreeBody.SwitchToPN_and_discard();
    }

  } else {
    HNO.SetModelSpace(modelspace_imsrg);
    if (IMSRG3) {
      HNO.ThreeBody.SwitchToPN_and_discard();
    }
  }

  // After truncating, get the perturbative energies again to see how much
  // things changed.
  if (eMax_imsrg != eMax) {
    std::cout << "Perturbative estimates of gs energy:" << std::endl;
    double EMP2 = HNO.GetMP2_Energy();
    double EMP2_3B = HNO.GetMP2_3BEnergy();
    std::cout << "EMP2 = " << EMP2 << std::endl;
    std::cout << "EMP2_3B = " << EMP2_3B << std::endl;
    std::array<double, 3> Emp_3 = HNO.GetMP3_Energy();
    double EMP3 = Emp_3[0] + Emp_3[1] + Emp_3[2];
    std::cout << "E3_pp = " << Emp_3[0] << "  E3_hh = " << Emp_3[1]
              << " E3_ph = " << Emp_3[2] << "   EMP3 = " << EMP3 << std::endl;
    std::cout << "To 3rd order, E = " << HNO.ZeroBody + EMP2 + EMP3 + EMP2_3B
              << std::endl;
  }

  if (method == "MP3") {
    HNO.PrintTimes();
    return 0;
  }

  std::cout << " " << __FILE__ << " line " << __LINE__
            << "noperators = " << HNO.profiler.counter["N_Operators"]
            << std::endl;

  IMSRGSolver imsrgsolver(HNO);
  std::cout << " " << __FILE__ << " line " << __LINE__
            << "noperators = " << HNO.profiler.counter["N_Operators"]
            << std::endl;
  //  imsrgsolver.SetHin(HNO); // necessary?
  imsrgsolver.SetReadWrite(rw);
  imsrgsolver.SetMethod(method);
  imsrgsolver.SetDenominatorPartitioning(denominator_partitioning);
  imsrgsolver.SetEtaCriterion(eta_criterion);
  imsrgsolver.max_omega_written = 500;
  imsrgsolver.SetSmax(smax);
  imsrgsolver.SetFlowFile(flowfile);
  imsrgsolver.SetDs(ds_0);
  imsrgsolver.SetDsmax(dsmax);
  imsrgsolver.SetdOmega(domega);
  imsrgsolver.SetOmegaNormMax(omega_norm_max);

  Commutator::SetUseIMSRG3(IMSRG3);
  Commutator::SetUseIMSRG3N7(imsrg3_n7);
  Commutator::SetUseReduced232Impl(reduced_232_impl);
  Commutator::SetUseIMSRG3_MP4(imsrg3_mp4);
  if (IMSRG3) {
    std::cout << "Using IMSRG(3) commutators. This will probably be slow..."
              << std::endl;
  }
  if (imsrg3_n7) {
    std::cout << "  only including IMSRG3 commutator terms that scale up to n7"
              << std::endl;
  }
  if (reduced_232_impl) {
    std::cout
        << "  using comm232ss implementation fully restricted to emax_3body"
        << std::endl;
  }

  imsrgsolver.SetGenerator(core_generator);
  if ((core_generator.find("imaginary") != std::string::npos) ||
      (core_generator.find("wegner") != std::string::npos)) {
    if (ds_0 > 1e-2) {
      ds_0 = 1e-4;
      dsmax = 1e-2;
      imsrgsolver.SetDs(ds_0);
      imsrgsolver.SetDsmax(dsmax);
    }
  }

  imsrgsolver.Solve();

  if (IMSRG3) {
    std::cout << "Norm of 3-body = " << imsrgsolver.GetH_s().ThreeBodyNorm()
              << std::endl;
  }

  if ((nsteps > 1) && (valence_space != reference)) {
    // two-step decoupling, do core first
    if (method == "magnus") smax *= 2;

    imsrgsolver.SetGenerator(valence_generator);
    std::cout << "Setting generator to " << valence_generator << std::endl;
    //    modelspace.ResetFirstPass();
    modelspace_imsrg.ResetFirstPass();
    if ((valence_generator.find("imaginary") != std::string::npos) ||
        (valence_generator.find("wegner") != std::string::npos)) {
      if (ds_0 > 1e-2) {
        ds_0 = 1e-4;
        dsmax = 1e-2;
        imsrgsolver.SetDs(ds_0);
        imsrgsolver.SetDsmax(dsmax);
      }
    }
    imsrgsolver.SetSmax(smax);
    imsrgsolver.Solve();
  }

  // If we're doing targeted/ensemble normal ordering
  // we now re-normal order wrt to the core
  // and do any remaining flow.
  //  ModelSpace ms2(modelspace);
  ModelSpace ms2(modelspace_imsrg);
  ms2.SetReference(ms2.core);  // change the reference
  bool renormal_order = false;
  //  if (modelspace.valence.size() > 0 )
  if (modelspace_imsrg.valence.size() > 0) {
    renormal_order =
        modelspace_imsrg.holes.size() != modelspace_imsrg.core.size();
    if (!renormal_order) {
      for (auto c : modelspace_imsrg.core) {
        if ((find(
                 modelspace_imsrg.holes.begin(),
                 modelspace_imsrg.holes.end(),
                 c) == modelspace_imsrg.holes.end()) ||
            (std::abs(1 - modelspace_imsrg.GetOrbit(c).occ) > 1e-6)) {
          renormal_order = true;
          break;
        }
      }
    }
  }

  if (renormal_order) {
    HNO = imsrgsolver.GetH_s();

    std::cout << "Undoing NO wrt A=" << modelspace_imsrg.GetAref()
              << " Z=" << modelspace_imsrg.GetZref() << std::endl;
    std::cout << "Before doing so, the spes are " << std::endl;
    for (auto i : modelspace_imsrg.all_orbits)
      std::cout << "  " << i << " : " << HNO.OneBody(i, i) << std::endl;
    if (IMSRG3) {
      rw.Write_NaiveVS3B(intfile + ".vs3b", HNO);
      // Use emax=3 because we are interested in pf shell systems
      rw.Write_me1j(intfile + "_ENO.me1j", HNO, 3, 3);
      rw.Write_me2jp(intfile + "_ENO.me2jp", HNO, 3, 6, 3);
      rw.Write_me3jp(intfile + "_ENO.me3jp", HNO, 3, 6, 9);
      std::cout << "Re-normal-ordering wrt the core. For now, we just throw "
                   "away the 3N at this step."
                << std::endl;
      HNO.SetNumberLegs(4);
      HNO.SetParticleRank(2);
    }

    HNO = HNO.UndoNormalOrdering();
    HNO.SetModelSpace(ms2);
    std::cout << "Doing NO wrt A=" << ms2.GetAref() << " Z=" << ms2.GetZref()
              << "  norbits = " << ms2.GetNumberOrbits() << std::endl;
    HNO = HNO.DoNormalOrdering();

    rw.Write_NaiveVS1B(intfile + ".vs1b", HNO);
    rw.Write_NaiveVS2B(intfile + ".vs2b", HNO);
    // Use emax=3 because we are interested in pf shell systems
    rw.Write_me1j(intfile + "_coreNO.me1j", HNO, 3, 3);
    rw.Write_me2jp(intfile + "_coreNO.me2jp", HNO, 3, 6, 3);

    imsrgsolver.FlowingOps[0] = HNO;
  }

  // Write the output

  // If we're doing a shell model interaction, write the
  // interaction files to disk.
  //  if (modelspace.valence.size() > 0)
  if (modelspace_imsrg.valence.size() > 0) {
    std::cout << "Writing files: " << intfile << std::endl;
    if (valence_file_format == "tokyo") {
      rw.WriteTokyo(imsrgsolver.GetH_s(), intfile + ".snt", "");
    } else {
      rw.WriteNuShellX_int(imsrgsolver.GetH_s(), intfile + ".int");
      rw.WriteNuShellX_sps(imsrgsolver.GetH_s(), intfile + ".sp");
    }

  } else {
    // single ref. just print the zero body pieces out. (maybe check if
    // its magnus?)
    std::cout << "Core Energy = " << std::setprecision(6)
              << imsrgsolver.GetH_s().ZeroBody << std::endl;
    if (method != "magnus") {
      for (index_t i = 0; i < ops.size(); ++i) {
        // the first operator is the Hamiltonian
        Operator& op = imsrgsolver.FlowingOps[i + 1];
        std::cout << opnames[i] << " = " << op.ZeroBody << std::endl;
        if (opnames[i] == "Rp2") {
          int Z = modelspace_imsrg.GetTargetZ();
          int A = modelspace_imsrg.GetTargetMass();
          std::cout << " IMSRG point proton radius = " << sqrt(op.ZeroBody)
                    << std::endl;
          std::cout << " IMSRG charge radius = "
                    << sqrt(
                           op.ZeroBody + PROTON_RCH2 +
                           NEUTRON_RCH2 * (A - Z) / Z + DARWIN_FOLDY)
                    << std::endl;
        }
      }
    }
  }

  /////////////////////
  /// Transform operators and write them

  if (method == "magnus") {
    /// if method is magnus, we didn't do this already. So we need to unpack any
    /// operators from file.

    int count_from_file = 0;

    std::cout << "transforming operators" << std::endl;

    std::cout << "transforming NLO hamiltonian" << std::endl;

    if (basis == "oscillator") {
      Hbare_NLO = Hbare_NLO.DoNormalOrdering();
    } else if (basis == "HF") {
      Hbare_NLO = hf.TransformToHFBasis(Hbare_NLO).DoNormalOrdering();
    } else if (basis == "NAT") {
      Operator op_2b = hf.TransformHOToNATBasis(Hbare_NLO);
      op_2b.SetParticleRank(2);

      Hbare_NLO =
          hf.GetNormalOrdered3BOperator(Hbare_NLO) + op_2b.DoNormalOrdering();
    }
    std::cout << "   HF: " << Hbare_NLO.ZeroBody << std::endl;

    if ((eMax_imsrg != -1) || (e2Max_imsrg != -1) || (e3Max_imsrg) != -1) {
      //     ModelSpace modelspace_imsrg = modelspace;
      std::cout << "Truncating modelspace for IMSRG calculation: emax e2max "
                   "e3max  ->  "
                << eMax_imsrg << " " << e2Max_imsrg << " " << e3Max_imsrg
                << std::endl;
      Hbare_NLO = Hbare_NLO.Truncate(modelspace_imsrg);
    }

    Hbare_NLO = imsrgsolver.Transform(Hbare_NLO);
    Hbare_NLO += imsrgsolver.GetH_s();

    std::cout << "transforming N2LO hamiltonian" << std::endl;

    if (basis == "oscillator") {
      Hbare_N2LO = Hbare_N2LO.DoNormalOrdering();
    } else if (basis == "HF") {
      Hbare_N2LO = hf.TransformToHFBasis(Hbare_N2LO).DoNormalOrdering();
    } else if (basis == "NAT") {
      Operator op_2b = hf.TransformHOToNATBasis(Hbare_N2LO);
      op_2b.SetParticleRank(2);

      Hbare_N2LO =
          hf.GetNormalOrdered3BOperator(Hbare_N2LO) + op_2b.DoNormalOrdering();
    }
    std::cout << "   HF: " << Hbare_N2LO.ZeroBody << std::endl;

    if ((eMax_imsrg != -1) || (e2Max_imsrg != -1) || (e3Max_imsrg) != -1) {
      //     ModelSpace modelspace_imsrg = modelspace;
      std::cout << "Truncating modelspace for IMSRG calculation: emax e2max "
                   "e3max  ->  "
                << eMax_imsrg << " " << e2Max_imsrg << " " << e3Max_imsrg
                << std::endl;
      Hbare_N2LO = Hbare_N2LO.Truncate(modelspace_imsrg);
    }

    Hbare_N2LO = imsrgsolver.Transform(Hbare_N2LO);

    for (size_t i = 0; i < opnames.size(); ++i) {
      auto opname = opnames[i];
      std::cout << i << ": " << opname << " " << std::endl;

      Operator op;

      op = imsrg_util::OperatorFromString(modelspace, opname);

      if ((basis == "oscillator") || (opname == "OccRef")) {
        op = op.DoNormalOrdering();
      } else if (basis == "HF") {
        op = hf.TransformToHFBasis(op).DoNormalOrdering();
      } else if (basis == "NAT") {
        Operator op_2b = hf.TransformHOToNATBasis(op);
        op_2b.SetParticleRank(2);

        op = hf.GetNormalOrdered3BOperator(op) + op_2b.DoNormalOrdering();
      }
      std::cout << "   HF: " << op.ZeroBody << std::endl;

      if ((eMax_imsrg != -1) || (e2Max_imsrg != -1) || (e3Max_imsrg) != -1) {
        //     ModelSpace modelspace_imsrg = modelspace;
        std::cout << "Truncating modelspace for IMSRG calculation: emax e2max "
                     "e3max  ->  "
                  << eMax_imsrg << " " << e2Max_imsrg << " " << e3Max_imsrg
                  << std::endl;
        op = op.Truncate(modelspace_imsrg);
      }

      op = imsrgsolver.Transform(op);

      std::cout << "Before renormal ordering Op(5,4) is "
                << std::setprecision(10) << op.OneBody(5, 4) << std::endl;
      if (renormal_order) {
        op = op.UndoNormalOrdering();
        op.SetModelSpace(ms2);
        op = op.DoNormalOrdering();
      }
      std::cout << "   IMSRG: " << op.ZeroBody << std::endl;
      std::cout << opname << "_IMSRG: " << op.ZeroBody << std::endl;

      std::cout << "      " << op.GetJRank() << " " << op.GetTRank() << " "
                << op.GetParity() << "   " << op.GetNumberLegs() << std::endl;
      if (((op.GetJRank() + op.GetTRank() + op.GetParity()) < 1) &&
          (op.GetNumberLegs() % 2 == 0)) {
        std::cout << "writing scalar files " << std::endl;
        if (valence_file_format == "tokyo") {
          rw.WriteTokyo(op, intfile + opname + ".snt", "op");
        } else {
          rw.WriteNuShellX_op(op, intfile + opname + ".int");
        }
      } else if (op.GetNumberLegs() % 2 == 1) {
        // odd number of legs -> this is
        // a dagger operator
        rw.WriteDaggerOperator(op, intfile + opname + ".dag", opname);
      } else {
        std::cout << "writing tensor files " << std::endl;
        if (valence_file_format == "tokyo") {
          rw.WriteTensorTokyo(intfile + opname + "_2b.snt", op);
        } else {
          rw.WriteTensorOneBody(intfile + opname + "_1b.op", op, opname);
          rw.WriteTensorTwoBody(intfile + opname + "_2b.op", op, opname);
        }
      }

    }  // for opnames
  }    // if method == "magnus"

  std::cout << "E_NLO = " << Hbare_NLO.ZeroBody << "\n";
  std::cout
      << "E_N2LO = "
      << GetSecondOrderCorrection(imsrgsolver.GetH_s(), Hbare_NLO, Hbare_NLO) +
             Hbare_N2LO.ZeroBody
      << "\n";

  if (write_omega) {
    std::string scratch = rw.GetScratchDir();
    imsrgsolver.FlushOmegaToScratch();
    for (int i = 0; i < imsrgsolver.GetNOmegaWritten(); i++) {
      std::ostringstream inputfile, outputfile;
      inputfile << scratch << "/OMEGA_" << std::setw(6) << std::setfill('0')
                << getpid() << std::setw(3) << std::setfill('0') << i;
      outputfile << intfile << "_Omega_" << i;
      rw.CopyFile(inputfile.str(), outputfile.str());
    }

    std::ofstream file_occ;
    std::ostringstream name_occ;
    int wint = 4;
    int wdouble = 26;
    int pdouble = 16;
    name_occ << intfile << "_occ.dat";
    file_occ.open(name_occ.str(), std::ofstream::out);
    for (auto i : modelspace.all_orbits) {
      Orbit& oi = modelspace.GetOrbit(i);
      if (std::abs(oi.occ) > 1e-6) {
        file_occ << std::setw(wint) << oi.n << std::setw(wint) << oi.l
                 << std::setw(wint) << oi.j2 << std::setw(wint) << oi.tz2
                 << std::setw(wdouble) << std::setiosflags(std::ios::fixed)
                 << std::setprecision(pdouble) << std::scientific << oi.occ
                 << std::endl;
      }
    }
    file_occ.close();
    if (basis == "NAT") {
      name_occ.str("");
      name_occ << intfile << "_occ_nat.dat";
      file_occ.open(name_occ.str(), std::ofstream::out);
      for (auto i : modelspace.all_orbits) {
        Orbit& oi = modelspace.GetOrbit(i);
        if (std::abs(oi.occ_nat) > 1e-6) {
          file_occ << std::setw(wint) << oi.n << std::setw(wint) << oi.l
                   << std::setw(wint) << oi.j2 << std::setw(wint) << oi.tz2
                   << std::setw(wdouble) << std::setiosflags(std::ios::fixed)
                   << std::setprecision(pdouble) << std::scientific
                   << oi.occ_nat << std::endl;
        }
      }
      file_occ.close();
    }

    bool filesucess = false;
    if (basis == "HF") {
      filesucess = hf.C.save(intfile + "C.mat");
    } else if (basis == "NAT") {
      filesucess = hf.C_HO2NAT.save(intfile + "C.mat");
    }

    if (filesucess == false) {
      std::cout << "Couldn't save HF coefficient matrix." << std::endl;
    }
  }

  if (IMSRG3) {
    std::cout << "Norm of 3-body = " << imsrgsolver.GetH_s().ThreeBodyNorm()
              << std::endl;
  }
  Hbare.PrintTimes();

  return 0;
}
