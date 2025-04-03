/////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////
///                                                  ____                                         ///
///        _________________           _____________/   /\               _________________        ///
///       /____/_____/_____/|         /____/_____/ /___/  \             /____/_____/_____/|       ///
///      /____/_____/__G_ /||        /____/_____/|/   /\  /\           /____/_____/____ /||       ///
///     /____/_____/__+__/|||       /____/_____/|/ G /  \/  \         /____/_____/_____/|||       ///
///    |     |     |     ||||      |     |     |/___/   /\  /\       |     |     |     ||||       ///
///    |  I  |  M  |     ||/|      |  I  |  M  /   /\  /  \/  \      |  I  |  M  |     ||/|       ///
///    |_____|_____|_____|/||      |_____|____/ + /  \/   /\  /      |_____|_____|_____|/||       ///
///    |     |     |     ||||      |     |   /___/   /\  /  \/       |     |     |     ||||       ///
///    |  S  |  R  |     ||/|      |  S  |   \   \  /  \/   /        |  S  |  R  |  G  ||/|       ///
///    |_____|_____|_____|/||      |_____|____\ __\/   /\  /         |_____|_____|_____|/||       ///
///    |     |     |     ||||      |     |     \   \  /  \/          |     |     |     ||||       ///
///    |     |  +  |     ||/       |     |  +  |\ __\/   /           |     |  +  |  +  ||/        ///
///    |_____|_____|_____|/        |_____|_____|/\   \  /            |_____|_____|_____|/         ///
///                                               \___\/                                          ///
///                                                                                               ///
///           imsrg++ : Interface for performing standard IMSRG calculations.                     ///
///                     Usage is imsrg++  option1=value1 option2=value2 ...                       ///
///                     To get a list of options, type imsrg++ help                               ///
///                                                                                               ///
///                                                      - Ragnar Stroberg 2016                   ///
///                                                                                               ///
/////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////////
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
/////////////////////////////////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <stdio.h>
#include <string>
#include <omp.h>
#include <boost/format.hpp>
#include "Commutator.hh"
#include "IMSRG.hh"
#include "Parameters.hh"
#include "PhysicalConstants.hh"
#include "version.hh"

struct OpFromFile {
   std::string file2name,file3name,opname;
   int j,p,t,r; // J rank, parity, dTz, particle rank
};

int main(int argc, char** argv)
{
  // Default parameters, and everything passed by command line args.
  std::cout << "######  imsrg++ build version: " << version::BuildVersion() << std::endl;

  Parameters parameters(argc,argv);
  if (parameters.help_mode) return 0;

  std::string inputobme = parameters.s("1bme");
  std::string inputtbme = parameters.s("2bme");
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
  std::string omefile = parameters.s("omefile");
  std::string vs_out = parameters.s("vs_out");
  std::string core_generator = parameters.s("core_generator");
  std::string valence_generator = parameters.s("valence_generator");
  std::string fmt2 = parameters.s("fmt2");
  std::string fmt3 = parameters.s("fmt3");
  std::string input_op_fmt = parameters.s("input_op_fmt");
  std::string denominator_delta_orbit = parameters.s("denominator_delta_orbit");
  std::string LECs = parameters.s("LECs");
  std::string scratch = parameters.s("scratch");
  std::string valence_file_format = parameters.s("valence_file_format");
  std::string occ_file = parameters.s("occ_file");
  std::string physical_system = parameters.s("physical_system");
  std::string denominator_partitioning = parameters.s("denominator_partitioning");
  std::string NAT_order = parameters.s("NAT_order");
  std::string iso_ch = parameters.s("isospin_ch");
  std::string kerdir = parameters.s("kerdir");

  bool use_brueckner_bch = parameters.s("use_brueckner_bch") == "true";
  bool nucleon_mass_correction = parameters.s("nucleon_mass_correction") == "true";
  bool relativistic_correction = parameters.s("relativistic_correction") == "true";
  bool IMSRG3 = parameters.s("IMSRG3") == "true";
  bool imsrg3_n7 = parameters.s("imsrg3_n7") == "true";
  bool reduced_232_impl = parameters.s("reduced_232_impl") == "true";
  bool imsrg3_mp4 = parameters.s("imsrg3_mp4") == "true";
  bool imsrg3_at_end = parameters.s("imsrg3_at_end") == "true";
  bool imsrg3_no_qqq = parameters.s("imsrg3_no_qqq") == "true";
  bool write_omega = parameters.s("write_omega") == "true";
  bool freeze_occupations = parameters.s("freeze_occupations")=="true";
  bool discard_no2b_from_3n = parameters.s("discard_no2b_from_3n")=="true";
  bool hunter_gatherer = parameters.s("hunter_gatherer") == "true";
  bool goose_tank = parameters.s("goose_tank") == "true";
  bool discard_residual_input3N = parameters.s("discard_residual_input3N")=="true";
  bool use_NAT_occupations = (parameters.s("use_NAT_occupations")=="true") ? true : false;
  bool order_NAT_by_energy = (parameters.s("order_NAT_by_energy")=="true") ? true : false;
  bool store_3bme_pn = (parameters.s("store_3bme_pn")=="true");
  bool only_2b_eta = (parameters.s("only_2b_eta")=="true");
  bool only_2b_omega = (parameters.s("only_2b_omega")=="true");
  bool only_2b_omega_at_end = (parameters.s("only_2b_omega_at_end")=="true");
  bool perturbative_triples = (parameters.s("perturbative_triples")=="true");
  bool brueckner_restart = false;
  bool write_HO_ops = parameters.s("write_HO_ops") == "true";  // added by Antoine Belley
  bool write_HF_ops = parameters.s("write_HF_ops") == "true";  // added by Antoine Belley
  bool use_HF_reference_in_NAT = parameters.s("use_HF_reference_in_NAT") == "true";
  bool sum_rule = parameters.s("moments") == "true";
  bool kernel = parameters.s("kernel") == "true";
  bool op_val = parameters.s("op_val") == "true";
  bool write_H = parameters.s("write_Hamiltonian") == "true";
  bool write_omega_me = parameters.s("write_omega_me") == "true";
  bool def_vs = parameters.s("def_params_vs") == "true";

  int eMax = parameters.i("emax");
  int lmax = parameters.i("lmax"); // so far I only use this with atomic systems.
  int E3max = parameters.i("e3max");
  int lmax3 = parameters.i("lmax3");
  int targetMass = parameters.i("A");
  int nsteps = parameters.i("nsteps");
  int file2e1max = parameters.i("file2e1max");
  int file2e2max = parameters.i("file2e2max");
  int file2lmax = parameters.i("file2lmax");
  int file3e1max = parameters.i("file3e1max");
  int file3e2max = parameters.i("file3e2max");
  int file3e3max = parameters.i("file3e3max");
  int atomicZ = parameters.i("atomicZ");
  int emax_unocc = parameters.i("emax_unocc");
  int eMax_imsrg = parameters.i("emax_imsrg");
  int e2Max_imsrg = parameters.i("e2max_imsrg");
  int e3Max_imsrg = parameters.i("e3max_imsrg");
  int eMax_3body_imsrg = parameters.i("emax_3body_imsrg");
  int imsrg3_commutator_depth = parameters.i("imsrg3_commutator_depth");
  int L_MixMom = parameters.i("L_MixMom");
  int N_Magnus = parameters.i("N_Magnus");

  double hw = parameters.d("hw");
  double smax = parameters.d("smax");
  double ode_tolerance = parameters.d("ode_tolerance");
  double dsmax = parameters.d("dsmax");
  double ds_0 = parameters.d("ds_0");
  double domega = parameters.d("domega");
  double omega_norm_max = parameters.d("omega_norm_max");
  double denominator_delta = parameters.d("denominator_delta");
  double BetaCM = parameters.d("BetaCM");
  double hwBetaCM = parameters.d("hwBetaCM");
  double eta_criterion = parameters.d("eta_criterion");
  double hw_trap = parameters.d("hw_trap");
  double dE3max = parameters.d("dE3max");
  double OccNat3Cut = parameters.d("OccNat3Cut");
  double threebody_threshold = parameters.d("threebody_threshold");

  double pol = parameters.d("polarisability"); // To evaluate m_-1 (TEST, needs multiple evaluations)

  double qL = parameters.d("qL");
  double qR = parameters.d("qR");

  std::string fL = parameters.s("fL");
  std::string fR = parameters.s("fR");

  std::vector<std::string> opnames = parameters.v("Operators");
  std::vector<std::string> opsfromfile = parameters.v("OperatorsFromFile");
  std::vector<std::string> opnamesPT1 = parameters.v("OperatorsPT1");
  std::vector<std::string> opnamesRPA = parameters.v("OperatorsRPA");
  std::vector<std::string> opnamesTDA = parameters.v("OperatorsTDA");

  std::vector<Operator> ops;
  std::vector<std::string> spwf = parameters.v("SPWF");

  using PhysConst::PROTON_RCH2;
  using PhysConst::NEUTRON_RCH2;
  using PhysConst::DARWIN_FOLDY;

  // Test 2bme file
  if (inputtbme != "none" and fmt2.find("oakridge") == std::string::npos and fmt2 != "schematic")
  {
    if (not std::ifstream(inputtbme).good())
    {
      std::cout << "trouble reading " << inputtbme << "  fmt2 = " << fmt2 << "   exiting. " << std::endl;
      return 1;
    }
  }
  // Test 3bme file
  if (input3bme != "none")
  {
    if (not std::ifstream(input3bme).good())
    {
      std::cout << "trouble reading " << input3bme << " exiting. " << std::endl;
      return 1;
    }
  }

  // unpack the awkward input format for reading an operator from file, and put it into a struct.
  // the format should look like OpName^j_t_p_r^/path/to/2bfile^/path/to/3bfile  if particle rank of Op is 2-body, then 3bfile is not needed.
  std::vector<OpFromFile> opsfromfile_unpacked;
  // If we're reading in other operators, make sure those are ok too
  for (auto &tag : opsfromfile)
  {
    std::istringstream ss(tag);
    std::string opname, qnumbers, f2name, f3name = "";

    OpFromFile opff;

    getline(ss, opname, '^');
    getline(ss, qnumbers, '^');
    getline(ss, f2name, '^');
    if (not ss.eof())
      getline(ss, f3name, '^');
    opff.opname = opname;
    opff.file2name = f2name;
    opff.file3name = f3name;

    ss.str(qnumbers);
    ss.clear();
    std::string tmp;
    getline(ss, tmp, '_');
    std::istringstream(tmp) >> opff.j;
    getline(ss, tmp, '_');
    std::istringstream(tmp) >> opff.t;
    getline(ss, tmp, '_');
    std::istringstream(tmp) >> opff.p;
    getline(ss, tmp, '_');
    std::istringstream(tmp) >> opff.r;

    std::cout << "Parsed tag. opname = " << opff.opname << "  " << opff.j << " " << opff.t << " " << opff.p << " " << opff.r << "   file2 = " << opff.file2name << "    file3 = " << opff.file3name << std::endl;

    if (opff.file2name != "")
    {
      if (not std::ifstream(opff.file2name).good())
      {
        std::cout << "trouble reading " << opff.file2name << " exiting. " << std::endl;
        return 1;
      }
    }

    if (opff.file3name != "") // is there a 3-body file too?
    {
      if (not std::ifstream(opff.file3name).good())
      {
        std::cout << "trouble reading " << opff.file3name << " exiting. " << std::endl;
        return 1;
      }
    }
    // if the files look good, then add it to the list
    opsfromfile_unpacked.push_back(opff);
  }

  ReadWrite rw;
  rw.SetLECs_preset(LECs);
  rw.SetScratchDir(scratch);
  rw.Set3NFormat( fmt3 );

  // deal with some short-hand method names
  if (method == "NSmagnus") // "No split" magnus
  {
    omega_norm_max = 50000;
    method = "magnus";
  }
  if (method.find("brueckner") != std::string::npos)
  {
    if (method == "brueckner2")
      brueckner_restart = true;
    if (method == "brueckner1step")
    {
      nsteps = 1;
      core_generator = valence_generator;
    }
    use_brueckner_bch = true;
    omega_norm_max = 500;
    method = "magnus";
  }

  // Test whether the scratch directory exists and we can write to it.
  // This is necessary because otherwise you get garbage for transformed operators and it's
  // not obvious what went wrong.
  if ( ((method == "magnus") || (method == "magnus_backoff")) and  ( (opnames.size() + opsfromfile.size()) > 0 )  )
  {
    if ( scratch=="/dev/null" or scratch=="/dev/null/")
    {
      std::cout << "ERROR!!! using Magnus with scratch = " << scratch << " but you're also trying to transform some operators. Dying now. " << std::endl;
      exit(EXIT_FAILURE);
    }
    else if ( scratch != "" )
    {
      std::string testfilename = scratch + "/_this_is_a_test_delete_me";
      std::ofstream testout(testfilename);
      testout << "PASSED" << std::endl;
      testout.close();

      // now read it back.
      std::ifstream testin(testfilename);
      std::string checkpassed;
      testin >> checkpassed;
      if ( (checkpassed != "PASSED") or ( not testout.good() ) or ( not testin.good() ) )
      {
        std::cout << "ERROR in " << __FILE__ <<  " failed test write to scratch directory " << scratch << " that's bad. Dying now." << std::endl;
        exit(EXIT_FAILURE);
      }

    }
  }

  //////////////////////////////////////////
  /// Setting the model space up

  if (custom_valence_space != "") // if a custom space is defined, the input valence_space is just used as a name
  {
    if (valence_space == "") // if no name is given, then just name it "custom"
    {
      parameters.string_par["valence_space"] = "custom";
      flowfile = parameters.DefaultFlowFile();
      intfile = parameters.DefaultIntFile();
    }
    valence_space = custom_valence_space;
  }

  ModelSpace modelspace = (reference == "default" ? ModelSpace(eMax, valence_space) : ModelSpace(eMax, reference, valence_space));

  modelspace.SetE3max(E3max);
  modelspace.SetLmax(lmax);

  if (emax_unocc > 0)
    modelspace.SetEmaxUnocc(emax_unocc);

  if (physical_system == "atomic")
    modelspace.InitSingleSpecies(eMax, reference, valence_space);

  if (occ_file != "none" and occ_file != "")
    modelspace.Init_occ_from_file(eMax, valence_space, occ_file);

  if (nsteps < 0) // default to 1 step for single ref, 2 steps for valence decoupling
    nsteps = modelspace.valence.size() > 0 ? 2 : 1;

  modelspace.SetHbarOmega(hw);

  if (targetMass > 0)
    modelspace.SetTargetMass(targetMass);

  if (lmax3 > 0)
    modelspace.SetLmax3(lmax3);

  // For both dagger operators and single particle wave functions, it's convenient to
  // just get every orbit in the valence space. So if SPWF="valence" ,  we append all valence orbits
  if ( std::find( spwf.begin(), spwf.end(), "valence" ) != spwf.end() )
  {
    // this erase/remove idiom is needed because remove just shuffles things around rather than actually removing it.
    spwf.erase( std::remove( spwf.begin(), spwf.end(), "valence" ), std::end(spwf) );
    for ( auto v : modelspace.valence )
    {
      spwf.push_back( modelspace.Index2String(v) );
    }
  }

  if ( std::find( opnames.begin(), opnames.end(), "rhop_all") != opnames.end() )
  {
    opnames.erase( std::remove( opnames.begin(), opnames.end(), "rhop_all"), std::end(opnames) );
    for ( double r=0.0; r<=10.0; r+=0.2 )
    {
       std::ostringstream opn;
       opn << "rhop_" << r;
       opnames.push_back( opn.str() );
    }
  }

  if ( std::find( opnames.begin(), opnames.end(), "rhon_all") != opnames.end() )
  {
    opnames.erase( std::remove( opnames.begin(), opnames.end(), "rhon_all"), std::end(opnames) );
    for ( double r=0.0; r<=10.0; r+=0.2 )
    {
       std::ostringstream opn;
       opn << "rhon_" << r;
       opnames.push_back( opn.str() );
    }
  }

  if ( std::find( opnames.begin(), opnames.end(), "DaggerHF_valence") != opnames.end() )
  {
    opnames.erase( std::remove( opnames.begin(), opnames.end(), "DaggerHF_valence"), std::end(opnames) );
    for ( auto v : modelspace.valence )
    {
      opnames.push_back( "DaggerHF_"+modelspace.Index2String(v) );
    }
    std::cout << "I found DaggerHF_valence, so I'm changing the opnames list to :" << std::endl;
    for ( auto opn : opnames ) std::cout << opn << " ,  ";
    std::cout << std::endl;
  }

  if ( std::find( opnames.begin(), opnames.end(), "DaggerAlln_valence") != opnames.end() )
  {
    opnames.erase( std::remove( opnames.begin(), opnames.end(), "DaggerAlln_valence"), std::end(opnames) );
    for ( auto v : modelspace.valence )
    {
      opnames.push_back( "DaggerAlln_"+modelspace.Index2String(v) );
    }
    std::cout << "I found DaggerAlln_valence, so I'm changing the opnames list to :" << std::endl;
    for ( auto opn : opnames ) std::cout << opn << " ,  ";
    std::cout << std::endl;
  }

  /////////////////////////////
  ///  Hamiltonian settings

  int particle_rank = input3bme == "none" ? 2 : 3;

  Operator Hbare = Operator(modelspace, 0, 0, 0, particle_rank);

  Hbare.SetHermitian();

  Commutator::SetUseGooseTank(goose_tank);
  Commutator::SetThreebodyThreshold(threebody_threshold);

  std::cout << "Reading interactions..." << std::endl;

  if (inputtbme != "none")
  {
    if (fmt2 == "me2j")
      rw.ReadBareTBME_Darmstadt(inputtbme, Hbare, file2e1max, file2e2max, file2lmax);
    if (fmt2 == "me2jp")
    {
      rw.Read_me1j(inputobme, Hbare, file2e1max, file2lmax);
      rw.Read_me2jp(inputtbme, Hbare, file2e1max, file2e2max, file2lmax);
    }
    else if (fmt2 == "navratil" or fmt2 == "Navratil")
      rw.ReadBareTBME_Navratil(inputtbme, Hbare);
    else if (fmt2 == "oslo")
      rw.ReadTBME_Oslo(inputtbme, Hbare);
    else if (fmt2.find("oakridge") != std::string::npos)
    { // input format should be: singleparticle.dat,vnn.dat
      size_t comma_pos = inputtbme.find_first_of(",");
      if (fmt2.find("bin") != std::string::npos)
        rw.ReadTBME_OakRidge(inputtbme.substr(0, comma_pos), inputtbme.substr(comma_pos + 1), Hbare, "binary");
      else
        rw.ReadTBME_OakRidge(inputtbme.substr(0, comma_pos), inputtbme.substr(comma_pos + 1), Hbare, "ascii");
    }
    else if (fmt2 == "takayuki")
      rw.ReadTwoBody_Takayuki(inputtbme, Hbare);
    else if (fmt2 == "nushellx")
      rw.ReadNuShellX_int(Hbare, inputtbme);
    else if (fmt2 == "schematic")
    {
      std::cout << "using schematic potential " << inputtbme << std::endl;
      if (inputtbme == "Minnesota")
        Hbare += imsrg_util::MinnesotaPotential(modelspace);
    }
    std::cout << "Done reading 2N" << std::endl;
  }

  // Read in the 3-body file
  if (Hbare.particle_rank >= 3)
  {
    if (input3bme_type == "full")
    {
      rw.Read_Darmstadt_3body(input3bme, Hbare, file3e1max, file3e2max, file3e3max);
    }
    if (input3bme_type == "no2b")
    {
      Hbare.ThreeBody.SetMode("no2b");
      if (no2b_precision == "half")
        Hbare.ThreeBody.SetMode("no2bhalf");

      Hbare.ThreeBody.ReadFile({input3bme}, {file3e1max, file3e2max, file3e3max, file3e1max});
      rw.File3N = input3bme;
    }
    else if (input3bme_type == "mono")
    {
      Hbare.ThreeBody.SetMode("mono");
      Hbare.ThreeBody.ReadFile({input3bme}, {file3e1max, file3e2max, file3e3max, file3e1max});
      rw.File3N = input3bme;
    }
    std::cout << "done reading 3N" << std::endl;
  }

  if (store_3bme_pn)
  {
    Hbare.ThreeBody.TransformToPN();
  }

  if (inputtbme == "none" and physical_system == "atomic")
  {
    using PhysConst::M_ELECTRON;
    using PhysConst::M_NUCLEON;
    int Z = (atomicZ>=0) ?  atomicZ : modelspace.GetTargetZ() ;
    Hbare -= Z*imsrg_util::VCentralCoulomb_Op(modelspace, lmax) * sqrt((M_ELECTRON*1e6)/M_NUCLEON ) ;
    Hbare += imsrg_util::VCoulomb_Op(modelspace, lmax) * sqrt((M_ELECTRON*1e6)/M_NUCLEON ) ;  // convert oscillator length from fm with nucleon mass to nm with electon mass (in eV).
    Hbare += imsrg_util::KineticEnergy_Op(modelspace); // Don't need to rescale this, because it's related to the oscillator frequency, which we input.
    Hbare /= PhysConst::HARTREE; // Convert to Hartree
  }

  if (fmt2 != "nushellx" and fmt2 != "me2jp" and physical_system != "atomic" and hw_trap < 0)  // Don't need to add kinetic energy if we read a shell model interaction
  {
    // Adding kinetic energy
    Hbare += imsrg_util::Trel_Op(modelspace); 

    // Added for evaluating the monopole IEWSR
    if (pol != 0.)
    {
      std::cout << "POLARISABILITY" << std::endl;
      Operator mono = imsrg_util::RSquaredOp(modelspace);
      
      Hbare += pol * mono;
    }

    if (Hbare.OneBody.has_nan())
    {
       std::cout << "  Looks like the Trel op is hosed from the get go. Dying." << std::endl;
       std::exit(EXIT_FAILURE);
    }
  }

  // Add an external harmonic trap
  if (hw_trap > 0)
  {
    Hbare += 0.5 * (PhysConst::M_NUCLEON * hw_trap * hw_trap) / (PhysConst::HBARC * PhysConst::HBARC) * imsrg_util::RSquaredOp(modelspace);
    Hbare += imsrg_util::KineticEnergy_Op(modelspace); // use lab-frame kinetic energy
  }

  // correction to kinetic energy because M_proton != M_neutron
  if (nucleon_mass_correction)
  {
    Hbare += imsrg_util::Trel_Masscorrection_Op(modelspace);
  }

  if ( relativistic_correction)
  {
    Hbare += imsrg_util::KineticEnergy_RelativisticCorr(modelspace);
  }

  // Add a Lawson center of mass term. If hwBetaCM is specified, use that frequency, otherwise use the basis frequency
  if (std::abs(BetaCM) > 1e-3)
  {
    if (hwBetaCM < 0)
      hwBetaCM = modelspace.GetHbarOmega();
    std::ostringstream hcm_opname;
    hcm_opname << "HCM_" << hwBetaCM;
    Hbare += BetaCM * imsrg_util::OperatorFromString(modelspace, hcm_opname.str());
  }

  // Solving Hartree Fock to start
  std::cout << "Creating HF" << std::endl;

  HFMBPT hf(Hbare); // HFMBPT inherits from HartreeFock, so this works for HF and NAT bases.

  if (not freeze_occupations)
    hf.UnFreezeOccupations();
  if (discard_no2b_from_3n)
    hf.DiscardNO2Bfrom3N();

  std::cout << "Solving" << std::endl;

  if (basis != "oscillator")
  {
    hf.Solve();
  }

  // decide what to keep after normal ordering
  int hno_particle_rank = 2;

  if (discard_residual_input3N)
    hno_particle_rank = 2;
  if (input3bme_type == "no2b")
    hno_particle_rank = 2;

  Operator& HNO = Hbare; // The reference & means we overwrite Hbare and save some memory
  if (basis == "HF" and method != "HF")
  {
    HNO = hf.GetNormalOrderedH(hno_particle_rank);
  }
  else if (basis == "NAT") // we want to use the natural orbital basis
  {
    // for backwards compatibility: order_NAT_by_energy overrides NAT_order
    if (order_NAT_by_energy)
      NAT_order = "energy";

    hf.UseNATOccupations(use_NAT_occupations);
    hf.OrderNATBy(NAT_order);

    //  GetNaturalOrbitals() calls GetDensityMatrix(), which computes the 1b density matrix up to MBPT2
    //  using the NO2B Hamiltonian in the HF basis, obtained with GetNormalOrderedH().
    //  Then it calls DiagonalizeRho() which diagonalizes the density matrix, yielding the natural orbital basis.
    hf.GetNaturalOrbitals();
    if (use_HF_reference_in_NAT)
    {
      hf.UseHFForHoleStates();
    }
    HNO = hf.GetNormalOrderedHNAT(hno_particle_rank);
  }
  else if (basis == "oscillator")
  {
    HNO = Hbare.DoNormalOrdering();
  }

  // If the length of spwf is zero, nothing happens
  imsrg_util::WriteSPWaveFunctions( spwf, hf, intfile);

  HNO -= BetaCM * 1.5 * hwBetaCM; // This is just the zero-body piece. The other stuff was added earlier.
  std::cout << "Hbare 0b = " << std::setprecision(8) << HNO.ZeroBody << std::endl;

  if (method != "HF")
  {
    std::cout << "Perturbative estimates of gs energy:" << std::endl;
    double EMP2 = HNO.GetMP2_Energy();
    double EMP2_3B = HNO.GetMP2_3BEnergy();
    std::cout << "EMP2 = " << EMP2 << std::endl;
    std::cout << "EMP2_3B = " << EMP2_3B << std::endl;
  }

  std::cout << "done with pert stuff, method = " << method << std::endl;

  // Calculate all the desired operators. If we're using magnus, we'll do this after the flow is over
  // TODO(mheinz): revert this logic to standard magnus logic and shift backoff details into solver.
  if (!(( method == "magnus" ) || (method == "magnus_backoff")))
  {
    for (auto& opname : opnames)
    {
        ops.emplace_back( imsrg_util::OperatorFromString(modelspace, opname) );
    }

    // Calculate first order perturbative correction to some operators, if that's what we asked for.
    // Strictly speaking, it doesn't make much sense to do this and then proceed with the IMSRG calculation,
    // but I'm not here to tell people what to do...
    for (auto& opnamept1 : opnamesPT1 )
    {
      ops.emplace_back( imsrg_util::FirstOrderCorr_1b( imsrg_util::OperatorFromString(modelspace,opnamept1)   , HNO ) );
      opnames.push_back( opnamept1+"PT1" );
    }
    for (auto& opnametda : opnamesTDA )
    {  // passing the argument "TDA" just sets the phhp and hpph blocks to zero in the RPA calculation
      ops.emplace_back( imsrg_util::RPA_resummed_1b( imsrg_util::OperatorFromString(modelspace,opnametda)   , HNO, "TDA" ) );
      opnames.push_back( opnametda+"TDA" );
    }
    for (auto& opnamerpa : opnamesRPA )
    {
      ops.emplace_back( imsrg_util::RPA_resummed_1b( imsrg_util::OperatorFromString(modelspace,opnamerpa)   , HNO, "RPA" ) );
      opnames.push_back( opnamerpa+"RPA" );
    }
  


    for ( auto& opff : opsfromfile_unpacked)
    {
      Operator op(modelspace, opff.j, opff.t, opff.p, opff.r );
      if (opff.r>2) 
      { 
        op.ThreeBody.SetMode("no2b");
        op.ThreeBody.Allocate();
      }
      if ( input_op_fmt == "navratil" )
      {
        rw.Read2bCurrent_Navratil( opff.file2name, op );
      }
      else if ( input_op_fmt == "miyagi" )
      {
        if (opff.file2name != "")
        {   
            Operator optmp = rw.ReadOperator2b_Miyagi( opff.file2name, modelspace );
            op.TwoBody = optmp.TwoBody;
        }
        if ( opff.r>2 and opff.file3name != "")  rw.Read_Darmstadt_3body( opff.file3name, op,  file3e1max,file3e2max,file3e3max);
      }
      else if ( input_op_fmt == "heinz")
      {
        if ( opff.r>2 and opff.file3name != "")  
        {

          op.ThreeBody.SetMode("no2b");
          if (no2b_precision == "half")  op.ThreeBody.SetMode("no2bhalf");

          op.ThreeBody.ReadFile( {opff.file3name}, {file3e1max, file3e2max, file3e3max, file3e1max} );
          // rw.Read_Darmstadt_3body( opff.file3name, op,  file3e1max,file3e2max,file3e3max);
        }
      }
      ops.push_back( op );
      opnames.push_back( opff.opname );
    }


   if (ops.size()>0)
   {
     std::cout << "operators to transform: " << std::endl;
     for ( auto& opn : opnames ) std::cout << opn << " ";
     std::cout << std::endl;
   }

   for (size_t i=0;i<ops.size();++i)
   {
      // We don't transform a DaggerHF, because we want the a^dagger to already refer to the HF basis.
     if ((basis == "HF") and (opnames[i].find("DaggerHF") == std::string::npos)  )
     {
       ops[i] = hf.TransformToHFBasis(ops[i]);
     }
     else if ((basis == "NAT") and (opnames[i].find("DaggerHF") == std::string::npos)  )
     {
       ops[i] = hf.TransformHOToNATBasis(ops[i]);
     }
     ops[i] = ops[i].DoNormalOrdering();
       std::cout << basis << " expectation value  " << opnames[i] << "  " << ops[i].ZeroBody << std::endl;
     if (method == "MP3")
     {
       double dop = ops[i].MP1_Eval( HNO );
       std::cout << "Operator 1st order correction  " << dop << "  ->  " << ops[i].ZeroBody + dop << std::endl;
     }
    if ( opnames[i] == "Rp2" )
    {
      double Rp2 = ops[i].ZeroBody;
      int Z = modelspace.GetTargetZ();
      int A = modelspace.GetTargetMass();
      std::cout << " HF point proton radius = " << sqrt( Rp2 ) << std::endl;
      std::cout << " HF charge radius = " << ( abs(Rp2)<1e-6 ? 0.0 : sqrt( Rp2 + PROTON_RCH2 + NEUTRON_RCH2*(A-Z)/Z + DARWIN_FOLDY) ) << std::endl;
    }    
   }// for ops.size
  }// if method != "magnus"

  if (basis == "HF" or basis == "NAT")
  {
    std::cout << basis << " Single particle energies and wave functions:" << std::endl;
    hf.PrintSPEandWF();
    std::cout << std::endl;
  }

  if (method == "HF")
  {
    HNO.PrintTimes();
    return 0;
  }

  if (method == "FCI")
  {
   if ( valence_file_format == "tokyo" )
   {
      HNO = HNO.UndoNormalOrdering();
      for (size_t i=0; i<ops.size();i++)
      {
         ops[i] = ops[i].UndoNormalOrdering();
      }


      modelspace.SetReference("vacuum");
      rw.WriteTokyo(HNO,intfile+".snt", "");
      // Haven't yet implemented FCI operators for Tokyo format. I should do this...
      for (size_t i=0; i<ops.size();i++)
      {
         if (ops[i].GetJRank()==0 and ops[i].GetTRank()==0 )
         {
           rw.WriteTokyo(ops[i], intfile + "_" + opnames[i] + ".snt","");
         }
         else
         {
          rw.WriteTensorTokyo(intfile+opnames[i]+".snt",ops[i]);
         }
      }
   }
   else // Write in NuShellX Format
   {
     // we want the 1b piece to be diagonal in the vacuum NO representation
      HNO = HNO.UndoNormalOrdering();
      double previous_zero_body = HNO.ZeroBody;
      modelspace.SetReference("vacuum");
      HartreeFock hfvac(HNO);
      hfvac.Solve();
  
      // Operator Hvac = hfvac.GetNormalOrderedH();
      HNO = hfvac.GetNormalOrderedH();
      std::cout << "HNO had zero body = " << HNO.ZeroBody << "  and I add " << previous_zero_body << " to it. " << std::endl;
      HNO.ZeroBody += previous_zero_body;
  
      rw.WriteNuShellX_int(HNO,intfile+".int");
      rw.WriteNuShellX_sps(HNO,intfile+".sp");
  
      std::cout << "NO wrt vacuum. One Body term is hopfully still diagonal?" << std::endl << HNO.OneBody << std::endl;
  
      for (index_t i=0;i<ops.size();++i)
      {
        ops[i] = ops[i].UndoNormalOrdering();
        if ((ops[i].GetJRank()+ops[i].GetTRank()+ops[i].GetParity())<1)
        {
          rw.WriteNuShellX_op(ops[i],intfile+opnames[i]+".int");
        }
        else
        {
          rw.WriteTensorOneBody(intfile+opnames[i]+"_1b.op",ops[i],opnames[i]);
          rw.WriteTensorTwoBody(intfile+opnames[i]+"_2b.op",ops[i],opnames[i]);
        }
      }
    }
    HNO.PrintTimes();
    return 0;
  }

  // We may want to use a smaller model space for the IMSRG evolution than we used for the HF step.
  // This is most effective when using natural orbitals or when including 3-body operators.

  ModelSpace modelspace_imsrg = modelspace;

  if ((eMax_imsrg != -1) or (e2Max_imsrg != -1) or (e3Max_imsrg != -1) or (eMax_3body_imsrg != -1))
  {
    if (eMax_imsrg == -1)
      eMax_imsrg = eMax;
    if (e2Max_imsrg == -1)
      e2Max_imsrg = 2 * eMax_imsrg;
    if (e3Max_imsrg == -1)
      e3Max_imsrg = std::min(E3max, 3 * eMax_imsrg);
    if (eMax_3body_imsrg == -1)
      eMax_3body_imsrg = eMax_imsrg;

    std::cout << "Truncating modelspace for IMSRG calculation: emax e2max e3max  ->  " << eMax_imsrg << " " << e2Max_imsrg << " " << e3Max_imsrg << std::endl;
    modelspace_imsrg.SetEmax(eMax_imsrg);
    modelspace_imsrg.SetE2max(e2Max_imsrg);
    modelspace_imsrg.SetE3max(e3Max_imsrg);
    modelspace_imsrg.SetEmax3Body(eMax_3body_imsrg);
    modelspace_imsrg.Init(eMax_imsrg, reference, valence_space);

    if (physical_system == "atomic")
      modelspace_imsrg.InitSingleSpecies(eMax_imsrg, reference, valence_space);
    if (occ_file != "none" and occ_file != "")
      modelspace_imsrg.Init_occ_from_file(eMax_imsrg, valence_space, occ_file);

    // If the occupations in modelspace were different from the naive filling, we want to keep those.
    std::map<index_t, double> hole_map;
    for (auto &i_new : modelspace_imsrg.all_orbits)
    {
      Orbit &oi_new = modelspace_imsrg.GetOrbit(i_new);
      index_t i_old = modelspace.GetOrbitIndex(oi_new.n, oi_new.l, oi_new.j2, oi_new.tz2);
      Orbit &oi_old = modelspace.GetOrbit(i_old);
      hole_map[i_new] = oi_old.occ;
    }
    modelspace_imsrg.SetReference(hole_map);

    HNO = HNO.Truncate(modelspace_imsrg);
  }
  else
  {
    HNO.SetModelSpace(modelspace_imsrg);
  }

  const int emax_reference = std::max(modelspace_imsrg.e_fermi.at(-1), modelspace_imsrg.e_fermi.at(1));
  const int emax_3b_reference = std::min(modelspace_imsrg.GetEMax3Body(), emax_reference);
  const int e3max_reference = std::min(3 * emax_3b_reference, modelspace_imsrg.GetE3max());
  const std::string emax_ref_string = "_e_" + std::to_string(emax_reference);
  const std::string emax3b_ref_string = "_e_" + std::to_string(emax_3b_reference);
  const std::string e3max_ref_string = "_E3_" + std::to_string(e3max_reference);

 // After truncating, get the perturbative energies again to see how much things changed.
  if (eMax_imsrg != eMax)
  {
    std::cout << "Perturbative estimates of gs energy:" << std::endl;
    double EMP2 = HNO.GetMP2_Energy();
    double EMP2_3B = HNO.GetMP2_3BEnergy();
    std::cout << "EMP2 = " << EMP2 << std::endl;
    std::cout << "EMP2_3B = " << EMP2_3B << std::endl;
    std::array<double, 3> Emp_3 = HNO.GetMP3_Energy();
    double EMP3 = Emp_3[0] + Emp_3[1] + Emp_3[2];
    std::cout << "E3_pp = " << Emp_3[0] << "  E3_hh = " << Emp_3[1] << " E3_ph = " << Emp_3[2] << "   EMP3 = " << EMP3 << std::endl;
    std::cout << "To 3rd order, E = " << HNO.ZeroBody + EMP2 + EMP3 + EMP2_3B << std::endl;
  }

  if (method == "MP3")
  {
    HNO.PrintTimes();
    return 0;
  }

  std::cout << " " << __FILE__ << " line " << __LINE__ << "noperators = " << HNO.profiler.counter["N_Operators"] << std::endl;

  IMSRGSolver imsrgsolver(HNO);
  std::cout << " " << __FILE__ << " line " << __LINE__ << "noperators = " << HNO.profiler.counter["N_Operators"] << std::endl;

  imsrgsolver.SetReadWrite(rw);
  imsrgsolver.SetMethod(method);
  imsrgsolver.SetDenominatorPartitioning(denominator_partitioning);
  imsrgsolver.SetEtaCriterion(eta_criterion);
  imsrgsolver.GetGenerator().SetOnly2bEta(only_2b_eta);
  imsrgsolver.max_omega_written = 500;
  imsrgsolver.SetHunterGatherer( hunter_gatherer );
  imsrgsolver.SetPerturbativeTriples(perturbative_triples);
  imsrgsolver.SetSmax(smax);
  imsrgsolver.SetFlowFile(flowfile);
  imsrgsolver.SetDs(ds_0);
  imsrgsolver.SetDsmax(dsmax);
  imsrgsolver.SetDenominatorDelta(denominator_delta);
  imsrgsolver.SetdOmega(domega);
  imsrgsolver.SetOmegaNormMax(omega_norm_max);
  imsrgsolver.SetODETolerance(ode_tolerance);
  if (denominator_delta_orbit != "none")
    imsrgsolver.SetDenominatorDeltaOrbit(denominator_delta_orbit);

//  if (method == "NSmagnus") // "No split" magnus
//  {
//    omega_norm_max=50000;
//    method = "magnus";
//  }
//  if (method.find("brueckner") != std::string::npos)
//  {
//    if (method=="brueckner2") brueckner_restart=true;
//    if (method=="brueckner1step")
//    {
//       nsteps = 1;
//       core_generator = valence_generator;
//    }
//    use_brueckner_bch = true;
//    omega_norm_max=500;
//    method = "magnus";
//  }

  Commutator::SetUseBruecknerBCH(use_brueckner_bch);

  if (use_brueckner_bch)
  {
    std::cout << "Using Brueckner flavor of BCH" << std::endl;
  }

  if (method == "flow" or method == "flow_RK4")
  {
    for (auto &op : ops)
      imsrgsolver.AddOperator(op);
    std::cout << " Added ops. FlowingOps.size = " << imsrgsolver.FlowingOps.size() << std::endl;
  }

  // IMSRG core decoupling
  imsrgsolver.SetGenerator(core_generator);

  if (core_generator.find("imaginary") != std::string::npos or core_generator.find("wegner") != std::string::npos)
  {
    if (ds_0 > 1e-2)
    {
      ds_0 = 1e-4;
      dsmax = 1e-2;
      imsrgsolver.SetDs(ds_0);
      imsrgsolver.SetDsmax(dsmax);
    }
  }

  imsrgsolver.Solve();

  if (brueckner_restart)
  {
     arma::mat newC = hf.C * arma::expmat( -imsrgsolver.GetOmega(0).OneBody  );
    // if (input3bme != "none") Hbare.SetParticleRank(3);
     HNO = hf.GetNormalOrderedH(newC);
     imsrgsolver.SetHin(HNO);
     imsrgsolver.s = 0;
     imsrgsolver.Solve();
  }

  if (nsteps > 1 and valence_space != reference) // two-step decoupling, do core first
  {
    if ((method == "magnus") || (method == "magnus_backoff"))
      smax *= 2;

    imsrgsolver.SetGenerator(valence_generator);
    std::cout << "Setting generator to " << valence_generator << std::endl;
    modelspace_imsrg.ResetFirstPass();
    if (valence_generator.find("imaginary") != std::string::npos or valence_generator.find("wegner") != std::string::npos)
    {
      if (ds_0 > 1e-2)
      {
        ds_0 = 1e-4;
        dsmax = 1e-2;
        imsrgsolver.SetDs(ds_0);
        imsrgsolver.SetDsmax(dsmax);
      }
    }
    imsrgsolver.SetSmax(smax);
    // IMSRG valence space decoupling
    imsrgsolver.Solve();
  }

  //   // Transform all the operators
  //   if (method == "magnus")
  //   {
  //     if (ops.size()>0) std::cout << "transforming operators" << std::endl;
  //     for (size_t i=0;i<ops.size();++i)
  //     {
  //       std::cout << opnames[i] << " " << std::endl;
  //       ops[i] = imsrgsolver.Transform(ops[i]);
  //       std::cout << " (" << ops[i].ZeroBody << " ) " << std::endl;
  //       // rw.WriteOperatorHuman(ops[i],intfile+opnames[i]+"_step2.op");
  //     }
  //     std::cout << std::endl;
  //     // increase smax in case we need to do additional steps
  //     smax *= 1.5;
  //     imsrgsolver.SetSmax(smax);
  //   }
  //   if (method == "flow" or method == "flow_RK4" )
  //   {
  //     for (size_t i=0;i<ops.size();++i)
  //     {
  //       ops[i] = imsrgsolver.GetOperator(i+1); // the zero-th operator is the Hamiltonian
  //     }
  //   }

  // If we're doing targeted/ensemble normal ordering we now re-normal order wrt to the core and do any remaining flow.

  ModelSpace ms2(modelspace_imsrg);
  ms2.SetReference(ms2.core); // change the reference
  bool renormal_order = false;

  if (modelspace_imsrg.valence.size() > 0)
  {
    renormal_order = modelspace.holes.size() != modelspace_imsrg.core.size();
    if (not renormal_order)
    {
      for (auto c : modelspace_imsrg.core)
      {
        if ((find(modelspace_imsrg.holes.begin(), modelspace_imsrg.holes.end(), c) == modelspace_imsrg.holes.end()) or (std::abs(1 - modelspace_imsrg.GetOrbit(c).occ) > 1e-6))
        {
          renormal_order = true;
          break;
        }
      }
    }
  }
  if (renormal_order)
  {
    HNO = imsrgsolver.GetH_s();

    // TODO:
    // if() {
    // int emax_imsrg = emax_reference;
    // std::string emax_imsrg_string = std::to_string(emax_imsrg);
    // rw.Write_me1j(intfile + "_" + emax_imsrg_string + ".me1j", HNO, emax_imsrg, emax_imsrg);
    // rw.Write_me2jp(intfile + "_" + emax_imsrg_string + ".me2jp", HNO, emax_imsrg, 2 * emax_imsrg, emax_imsrg);
    // exit(-1);
    // }

    //    int nOmega = imsrgsolver.GetOmegaSize() + imsrgsolver.GetNOmegaWritten();
    //    std::cout << "Undoing NO wrt A=" << modelspace.GetAref() << " Z=" << modelspace.GetZref() << std::endl;
    std::cout << "Undoing NO wrt A=" << modelspace_imsrg.GetAref() << " Z=" << modelspace_imsrg.GetZref() << std::endl;
    std::cout << "Before doing so, the spes are " << std::endl;

    for (auto i : modelspace_imsrg.all_orbits)
      std::cout << "  " << i << " : " << HNO.OneBody(i, i) << std::endl;

    HNO = HNO.UndoNormalOrdering();
    HNO.SetModelSpace(ms2);
    std::cout << "Doing NO wrt A=" << ms2.GetAref() << " Z=" << ms2.GetZref() << "  norbits = " << ms2.GetNumberOrbits() << std::endl;
    HNO = HNO.DoNormalOrdering();

    rw.Write_NaiveVS1B(vs_out + ".vs1b", HNO);
    rw.Write_NaiveVS2B(vs_out + ".vs2b", HNO);
    // Use emax=3 because we are interested in pf shell systems
    rw.Write_me1j(vs_out + "_coreNO" + emax_ref_string + ".me1j", HNO, emax_reference, emax_reference);
    rw.Write_me2jp(vs_out + "_coreNO" + emax_ref_string + ".me2jp", HNO, emax_reference, 2 * emax_reference, emax_reference);

    imsrgsolver.FlowingOps[0] = HNO;

    // More flowing is unnecessary, since things should stay decoupled.
    //    imsrgsolver.SetHin(HNO);
    //    imsrgsolver.SetEtaCriterion(1e-4);
    //    imsrgsolver.Solve();
    //     Change operators to the new basis, then apply the rest of the transformation
    //    std::cout << "Final transformation on the operators..." << std::endl;
    //    int iop = 0;
    //    for (auto& op : ops)
    //    {
    //      std::cout << opnames[iop++] << std::endl;
    //      op = op.UndoNormalOrdering();
    //      op.SetModelSpace(ms2);
    //      op = op.DoNormalOrdering();
    //      // transform using the remaining omegas
    //      op = imsrgsolver.Transform_Partial(op,nOmega);
    //    }
  }

  // Write the output
  //
  // If we're doing a shell model interaction, write the interaction files to disk.
  if (modelspace_imsrg.valence.size() > 0)
  {
    if (valence_file_format == "antoine") // this is still being tested...
    {
      rw.WriteAntoine_int(imsrgsolver.GetH_s(), intfile + ".ant");
      rw.WriteAntoine_input(imsrgsolver.GetH_s(), intfile + ".inp", modelspace_imsrg.GetAref(), modelspace_imsrg.GetZref());
    }
    std::cout << "Writing files: " << intfile << std::endl;
    if (valence_file_format == "tokyo")
    {
      rw.WriteTokyo(imsrgsolver.GetH_s(), vs_out + ".snt", "");
    }
    else
    {
      rw.WriteNuShellX_int(imsrgsolver.GetH_s(), intfile + ".int");
      rw.WriteNuShellX_sps(imsrgsolver.GetH_s(), intfile + ".sp");
    }

    //  if (method == "magnus" or method=="flow_RK4")
    //  {
    //     for (index_t i=0;i<ops.size();++i)
    //     {
    //        if ( ((ops[i].GetJRank()+ops[i].GetTRank()+ops[i].GetParity())<1) and (ops[i].GetNumberLegs()%2==0) )
    //        {
    //          if (valence_file_format == "tokyo")
    //          {
    //            rw.WriteTokyo(ops[i],intfile+opnames[i]+".snt", "op");
    //          }
    //          else
    //          {
    //            rw.WriteNuShellX_op(ops[i],intfile+opnames[i]+".int");
    //          }
    //        }
    //        else if ( ops[i].GetNumberLegs()%2==1) // odd number of legs -> this is a dagger operator
    //        {
    //            rw.WriteNuShellX_op(ops[i],intfile+opnames[i]+".int"); // do this for now. later make a *.dag format.
    //          rw.WriteDaggerOperator( ops[i], intfile+opnames[i]+".dag",opnames[i]);
    //        }
    //        else
    //        {
    //          if (valence_file_format == "tokyo")
    //          {
    //            rw.WriteTensorTokyo(intfile+opnames[i]+"_2b.snt",ops[i]);
    //          }
    //          else
    //          {
    //            rw.WriteTensorOneBody(intfile+opnames[i]+"_1b.op",ops[i],opnames[i]);
    //            rw.WriteTensorTwoBody(intfile+opnames[i]+"_2b.op",ops[i],opnames[i]);
    //          }
    //        }
    //     }
    //  }
  }
  else // single ref. just print the zero body pieces out
  {
    std::cout << "Core Energy = " << std::setprecision(6) << imsrgsolver.GetH_s().ZeroBody << std::endl;
    if (!((method == "magnus") || (method == "magnus_backoff")))
    {
      for (index_t i = 0; i < ops.size(); ++i)
      {
        Operator &op = imsrgsolver.FlowingOps[i + 1]; // the first operator is the Hamiltonian
        std::cout << opnames[i] << " = " << op.ZeroBody << std::endl;
        if (opnames[i] == "Rp2")
        {
          int Z = modelspace_imsrg.GetTargetZ();
          int A = modelspace_imsrg.GetTargetMass();
          std::cout << " IMSRG point proton radius = " << sqrt(op.ZeroBody) << std::endl;
          std::cout << " IMSRG charge radius = " << sqrt(op.ZeroBody + PROTON_RCH2 + NEUTRON_RCH2 * (A - Z) / Z + DARWIN_FOLDY) << std::endl;
        }
        if ((op.GetJRank() > 0) or (op.GetTRank() > 0)) // if it's a tensor, you probably want the full operator
        {
          std::cout << "Writing operator to " << intfile + opnames[i] + ".op" << std::endl;
          rw.WriteOperatorHuman(op, intfile + opnames[i] + ".op");
        }
      }
    }
  }

/////////////////////
/// Transform operators and write them

  if ((method == "magnus") || (method == "magnus_backoff"))
  {

    /// if method is magnus, we didn't do this already. So we need to unpack any operators from file.

    for ( auto& opff : opsfromfile_unpacked)
    {
      opnames.push_back( opff.opname + "_FROMFILE");
    }

    int count_from_file =0;

    if (opnames.size()>0) std::cout << "transforming operators" << std::endl;

    // fetch Hamiltonian
    Operator H_HF = *(imsrgsolver.H_0);
    Operator Hs = imsrgsolver.GetH_s();

    // Evaluating Sum Rules for the monopole operator before and after the IMSRG flow
    if(sum_rule)
    {
      Operator Monopole = imsrg_util::ElectricMultipoleOp(modelspace, 0, 2, "isoscalar");
      
      /////////

      Operator H0_HO = H_HF;

      arma::mat CHO  = hf.C;
      arma::mat CHOt = CHO.t();

      H0_HO = H0_HO.UndoNormalOrdering();
      H0_HO = H0_HO.Transform(CHOt); // Hamiltonioan in the HO basis

      if (pol == 0.)
      {
        Operator M0 = imsrg_util::Mom0(modelspace, Monopole); // Remember to multiply by 4 * PI the result
        Operator M1 = imsrg_util::Mom1(modelspace, H0_HO, Monopole);

        if (basis == "oscillator")
        {
          M0 = M0.DoNormalOrdering();
          M1 = M1.DoNormalOrdering();
        }
        else if (basis == "HF")
        {
          M0 = hf.TransformToHFBasis(M0).DoNormalOrdering();
          M1 = hf.TransformToHFBasis(M1).DoNormalOrdering();
        }   
        else if (basis == "NAT")
        {
          M0 = hf.TransformHOToNATBasis(M0).DoNormalOrdering();
          M1 = hf.TransformHOToNATBasis(M1).DoNormalOrdering();
        }

        Operator M0_s = imsrgsolver.Transform(M0);
        Operator M1_s = imsrgsolver.Transform(M1);

        std::cout << "Monopole moments values" << std::endl;
        std::cout << std::endl;
        std::cout << "M0(0) = " << M0.ZeroBody   << "   M1(0) = " << M1.ZeroBody   << std::endl;
        std::cout << "M0(s) = " << M0_s.ZeroBody << "   M1(s) = " << M1_s.ZeroBody << std::endl;
        std::cout << std::endl;

        //Operator op_s = imsrgsolver.Transform(Monopole);

        // sum rules at HF and IMSRG level
        /*if(Monopole.rank_J == 0)
        {
          std::cout << "Mean field values for sum rules" << std::endl;
          Commutator::EvaluateCommutatorSumRule(Monopole,H_HF,9);
          Commutator::EvaluateCommutatorSumRuleSymmetric(Monopole,H_HF,9);
    
          
          Operator S1 = Commutator::EvaluateCommutatorSumRule_op(Monopole, Hs, 1);   // Using evolved H and unevolved operator
          Operator Ss = Commutator::EvaluateCommutatorSumRule_op(op_s, Hs, 1); // Using both evolved H and operator
          

          std::cout << "m1 sum rule with unevolved H and op:  " << - 1./2. * S0.ZeroBody << std::endl;
          std::cout << "m1 sum rule with evolved H and un op: " << - 1./2. * S1.ZeroBody << std::endl;
          std::cout << "m1 sum rule with evolved H and op:    " << - 1./2. * Ss.ZeroBody << std::endl;
          std::cout << "m1 sum rule with evolved S0:          " << - 1./2. * Sn.ZeroBody << std::endl;

          std::cout << "IMSRG values for sum rules" << std::endl;
          Commutator::EvaluateCommutatorSumRule(Monopole,Hs,9);
          Commutator::EvaluateCommutatorSumRuleSymmetric(Monopole,Hs,9);
        }*/
      }

      Operator Rm2  = imsrg_util::Rm2_corrected_Op(modelspace, modelspace.GetTargetMass(), modelspace.GetTargetZ());
      Operator mono = imsrg_util::RSquaredOp(modelspace);

      /////////

      if (basis == "oscillator")
      {
        Rm2  = Rm2.DoNormalOrdering();
        mono = mono.DoNormalOrdering();
      }
      else if (basis == "HF")
      {
        Rm2  = hf.TransformToHFBasis(Rm2).DoNormalOrdering();
        mono = hf.TransformToHFBasis(mono).DoNormalOrdering();
      }
      else if (basis == "NAT")
      {
        Rm2 = hf.TransformHOToNATBasis(Rm2).DoNormalOrdering();
        mono = hf.TransformHOToNATBasis(mono).DoNormalOrdering();
      }

      Operator Rm2_s  = imsrgsolver.Transform(Rm2);
      Operator mono_s = imsrgsolver.Transform(mono);

      std::cout << "Energy and Excitation Operator expectation values, lda = " << pol << std::endl;
      std::cout << std::endl;
      std::cout << "E(0) = " << H_HF.ZeroBody << "   Q(0) = " << mono.ZeroBody   << "   Rm2(0) = " << Rm2.ZeroBody   << std::endl;
      std::cout << "E(s) = " << Hs.ZeroBody   << "   Q(s) = " << mono_s.ZeroBody << "   Rm2(s) = " << Rm2_s.ZeroBody << std::endl;
      std::cout << std::endl;
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Kernel function for coorelated response (Andrea, January 2025) /////////////
    ///////////////////////////////////////////////////////////////////////////////

    if (kernel)
    {
      // Set the Hamiltonian in the HO basis
      H_HF = H_HF.UndoNormalOrdering();
      arma::mat CHO  = hf.C;
      arma::mat CHOt = CHO.t();
      Operator H0_HO = H_HF.Transform(CHOt); // Hamiltonioan in the HO basis

      int L = L_MixMom;

      imsrg_response::KernelParams par;

      par.hf         = &hf;
      par.kerdir     = kerdir;
      par.modelspace = &modelspace;
      par.N_Magnus   = N_Magnus;
      par.omefile    = omefile;

      imsrg_response::computeKernel(qL, qR, H0_HO, L, fL, fR, par);
    }

    bool kernel1 = false;

    if (kernel1)
    {
      H_HF = H_HF.UndoNormalOrdering();
      arma::mat CHO  = hf.C;
      arma::mat CHOt = CHO.t();
      Operator H0_HO = H_HF.Transform(CHOt); // Hamiltonioan in the HO basis

      int L = L_MixMom; // select multipolarity, default is 0
      std::string pn = iso_ch; // select isospin channel, default is proton

      // Initialising left and right operators

      Operator OpL = Operator(modelspace, 0, 0, 0, 2);
      Operator OpR = Operator(modelspace, 0, 0, 0, 2);

      Operator OpSubL = Operator(modelspace, 0, 0, 0, 2);
      Operator OpSubR = Operator(modelspace, 0, 0, 0, 2);

      if (L == 0) // In the monopole case one needs to set the operator to subtract
      {
        if (qL == 0)
        {
          OpL     = imsrg_util::ElectricMultipoleOp(modelspace, L, 2, pn);
          OpSubL  = imsrg_util::RSquaredOp(modelspace, pn);
          OpSubL /= sqrt(4 * PhysConst::PI);
        }
        else
        {
          OpL     = imsrg_util::BesselMultipoleOp(modelspace, L, qL, pn);
          OpSubL  = imsrg_util::BesselMonopoleOp(modelspace, qL, pn);
          OpSubL /= sqrt(4 * PhysConst::PI);
        }
        if (qR == 0)
        {
          OpR     = imsrg_util::ElectricMultipoleOp(modelspace, L, 2, pn);
          OpSubR  = imsrg_util::RSquaredOp(modelspace, pn);
          OpSubR /= sqrt(4 * PhysConst::PI);
        }
        else
        {
          OpR     = imsrg_util::BesselMultipoleOp(modelspace, L, qR, pn);
          OpSubR  = imsrg_util::BesselMonopoleOp(modelspace, qR, pn);
          OpSubR /= sqrt(4 * PhysConst::PI);
        }
      }
      else // Otherwise not
      {
        if (qL == 0)
          OpL = imsrg_util::ElectricMultipoleOp(modelspace, L,  0, pn);
        else if (qL > 80)
        {
          int offset = int(100 - qL);

          Operator cond = imsrg_util::RadialPower(modelspace, L + offset, pn);
          cond  = hf.TransformToHFBasis(cond).DoNormalOrdering();
          double fac = cond.ZeroBody;

          std::cout << "conditioning factor: " << fac << std::endl;

          OpL  = imsrg_util::ElectricMultipoleOp(modelspace, L,  offset, pn); // e.g. qL = 98, offset = 2
          OpL /= fac * fac;
        }
        else
          OpL = imsrg_util::BesselMultipoleOp(modelspace, L, qL, pn);

        if (qR == 0)
          OpR = imsrg_util::ElectricMultipoleOp(modelspace, L,  0, pn);
        else if (qR > 80)
        {
          int offset = int(100 - qR);

          Operator cond = imsrg_util::RadialPower(modelspace, L + offset, pn);
          cond = hf.TransformToHFBasis(cond).DoNormalOrdering();
          double fac = cond.ZeroBody;

          OpR  = imsrg_util::ElectricMultipoleOp(modelspace, L,  offset, pn);
          OpR /= fac * fac;
        }
        else
          OpR = imsrg_util::BesselMultipoleOp(modelspace, L, qR, pn);
      }

      Operator MixMom0 = imsrg_util::Mix0(modelspace, OpL, OpR);
      Operator MixMom1 = imsrg_util::Mix1(modelspace, H0_HO, OpL, OpR);

      // testing

      Operator T = imsrg_util::KineticEnergy_Op(modelspace);

      Operator MixT_1 = imsrg_util::Mix1(modelspace, T, OpL, OpR);

      Operator OpTL = Commutator::Commutator(T, OpL);
      Operator OpTR = Commutator::Commutator(T, OpR);

      Operator MixT_3 = imsrg_util::Mix1(modelspace, H0_HO, OpTL, OpTR);

      ///////

      MixMom0 = hf.TransformToHFBasis(MixMom0).DoNormalOrdering();
      MixMom1 = hf.TransformToHFBasis(MixMom1).DoNormalOrdering();

      MixT_1 = hf.TransformToHFBasis(MixT_1).DoNormalOrdering(); // testing
      MixT_3 = hf.TransformToHFBasis(MixT_3).DoNormalOrdering();

      if (L == 0)
      {
        OpSubL = hf.TransformToHFBasis(OpSubL).DoNormalOrdering();
        OpSubR = hf.TransformToHFBasis(OpSubR).DoNormalOrdering();
      }

      double mom0_0 = MixMom0.ZeroBody;
      double mom1_0 = MixMom1.ZeroBody;
      double momT1_0 = MixT_1.ZeroBody;
      double momT3_0 = MixT_3.ZeroBody;

      if (L == 0)
        mom0_0 -= OpSubL.ZeroBody * OpSubR.ZeroBody;

      // Previous implementation with only one BCH-contracted operator

      // Operator Magnus = Operator(modelspace, 0, 0, 0, 2);

      // Magnus.SetAntiHermitian();

      // rw.Read_me1j (omefile + ".me1j.gz",  Magnus, eMax, eMax);
      // rw.Read_me2jp(omefile + ".me2jp.gz", Magnus, eMax, 2 * eMax, eMax);

      // MixMom0 = Commutator::BCH_Transform(MixMom0, Magnus);
      // MixMom1 = Commutator::BCH_Transform(MixMom1, Magnus);

      // if (L == 0)
      // {
      //   OpSubL = Commutator::BCH_Transform(OpSubL, Magnus);
      //   OpSubR = Commutator::BCH_Transform(OpSubR, Magnus);
      // }

      for (int i = 0; i < N_Magnus; i++)
      {
        Operator Magnus_i = Operator(modelspace, 0, 0, 0, 2);

        Magnus_i.SetAntiHermitian();

        rw.Read_me1j (omefile + "_" + std::to_string(i) + ".me1j.gz",  Magnus_i, eMax, eMax);
        rw.Read_me2jp(omefile + "_" + std::to_string(i) + ".me2jp.gz", Magnus_i, eMax, 2 * eMax, eMax);

        MixMom0 = Commutator::BCH_Transform(MixMom0, Magnus_i);
        MixMom1 = Commutator::BCH_Transform(MixMom1, Magnus_i);
        MixT_1 = Commutator::BCH_Transform(MixT_1, Magnus_i);
        MixT_3 = Commutator::BCH_Transform(MixT_3, Magnus_i);

        //cm = Commutator::BCH_Transform(cm, Magnus_i);

        if (L == 0)
        {
          OpSubL = Commutator::BCH_Transform(OpSubL, Magnus_i);
          OpSubR = Commutator::BCH_Transform(OpSubR, Magnus_i);
        }
      }

      //std::cout << "Spurious COM contribution (IMSRG):  " << cm.ZeroBody << std::endl;

      double mom0_s = MixMom0.ZeroBody;
      double mom1_s = MixMom1.ZeroBody;
      double momT1_s = MixT_1.ZeroBody;
      double momT3_s = MixT_3.ZeroBody;

      if (L == 0)
        mom0_s -= OpSubL.ZeroBody * OpSubR.ZeroBody;

      auto ss = boost::format{"%s/L=%i_%s_%.3f_%.3f.dat"} % kerdir % L % pn % qL % qR;
      std::string filename = ss.str();

      imsrg_util::printKernel(std::cout, qL, qR, mom0_0, mom1_0, momT1_0, momT3_0, mom0_s, mom1_s, momT1_s, momT3_s);

      // Print to file
      std::ofstream file(filename);
      if (file)
      {
        imsrg_util::printKernel(file, qL, qR, mom0_0, mom1_0, momT1_0, momT3_0, mom0_s, mom1_s, momT1_s, momT3_s);
        file.close();
        std::cout << "Data written to " << filename << std::endl;
      }
      else
      {
        std::cout << "Error opening file " << filename << std::endl;
      }
    }

    bool TE = false;

    if(TE)
    {
      H_HF = H_HF.UndoNormalOrdering();
      arma::mat CHO  = hf.C;
      arma::mat CHOt = CHO.t();
      Operator H0_HO = H_HF.Transform(CHOt);

      int L = 0;

      double q = 0.01;

      for(int i = 0; i < 1000; i++)
      {
        double sub_0 = 0.;
        double sub_s = 0.;

        Operator Op    = imsrg_response::SetOperator(modelspace, q, L, "IS");
        Operator OpSub = imsrg_response::SetSub(modelspace, q, "IS");

        OpSub = hf.TransformToHFBasis(OpSub).DoNormalOrdering();
        sub_0 = OpSub.ZeroBody;

        OpSub = imsrgsolver.Transform(OpSub);
        sub_s = OpSub.ZeroBody;

        Operator MixMom0 = imsrg_response::Mix0(modelspace, Op, Op);
        Operator MixMom1 = imsrg_response::Mix1(modelspace, H0_HO, Op, Op);

        MixMom0 = hf.TransformToHFBasis(MixMom0).DoNormalOrdering();
        MixMom1 = hf.TransformToHFBasis(MixMom1).DoNormalOrdering();

        double mom0_0 = MixMom0.ZeroBody - pow(sub_0, 2);
        double mom1_0 = MixMom1.ZeroBody;

        MixMom0 = imsrgsolver.Transform(MixMom0);
        MixMom1 = imsrgsolver.Transform(MixMom1);

        double mom0_s = MixMom0.ZeroBody - pow(sub_s, 2);
        double mom1_s = MixMom1.ZeroBody;

        std::cout << std::scientific<< q << "\t" << mom0_0 << "\t" << mom1_0 << "\t" << mom0_s << "\t" << mom1_s << "\t" << mom1_0 / mom0_0 << "\t" << mom1_s / mom0_s <<std::endl;

        q += 0.001;
      }
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Operator evaluation from pre-stored Magnus operators (Andrea, January 2025)
    ///////////////////////////////////////////////////////////////////////////////

    if(op_val)
    {
      std::vector<Operator> oplist;

      for (size_t i = 0; i < opnames.size(); i++)
      {
        auto opname = opnames[i];
        Operator op = imsrg_util::OperatorFromString(modelspace, opname);

        oplist.push_back(op);

        oplist.at(i) = hf.TransformToHFBasis(op).DoNormalOrdering();
        std::cout << opname << "_HF: " << oplist.at(i).ZeroBody << std::endl;
      }
      // Previous implementation with only one BCH-contracted operator

      // Operator Magnus = Operator(modelspace, 0, 0, 0, 2);

      // Magnus.SetAntiHermitian();

      // rw.Read_me1j (omefile + ".me1j.gz",  Magnus, eMax, eMax);
      // rw.Read_me2jp(omefile + ".me2jp.gz", Magnus, eMax, 2 * eMax, eMax);

      // op = Commutator::BCH_Transform(op, Magnus);

      for (int i = 0; i < N_Magnus; i++)
      {
        Operator Magnus_i = Operator(modelspace, 0, 0, 0, 2);

        Magnus_i.SetAntiHermitian();

        rw.Read_me1j (omefile + "_" + std::to_string(i) + ".me1j.gz",  Magnus_i, eMax, eMax);
        rw.Read_me2jp(omefile + "_" + std::to_string(i) + ".me2jp.gz", Magnus_i, eMax, 2 * eMax, eMax);

        for (size_t i = 0; i < opnames.size(); i++)
        {
          oplist.at(i) = Commutator::BCH_Transform(oplist.at(i), Magnus_i);
        }
      }

      for (int i = 0; i < N_Magnus; i++)
      {
        std::cout << opnames[i] << "_IMSRG: " << oplist.at(i).ZeroBody << std::endl;
      }
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Valence-space writing of radii and higher multipole contributions 
    ///////////////////////////////////////////////////////////////////////////////

    if(def_vs)
    {
      Operator Q2 = imsrg_util::ElectricMultipoleOp(modelspace, 2,  0, "proton"); // Only protons are considered for the charge distribution
      Operator Q3 = imsrg_util::ElectricMultipoleOp(modelspace, 3,  0, "proton");
      Operator Q4 = imsrg_util::ElectricMultipoleOp(modelspace, 4,  0, "proton");

      Operator var2 = imsrg_util::Mix0(modelspace, Q2, Q2);
      Operator var3 = imsrg_util::Mix0(modelspace, Q3, Q3);
      Operator var4 = imsrg_util::Mix0(modelspace, Q4, Q4);

      var2 = hf.TransformToHFBasis(var2).DoNormalOrdering();
      var3 = hf.TransformToHFBasis(var3).DoNormalOrdering();
      var4 = hf.TransformToHFBasis(var4).DoNormalOrdering();

      var2 = imsrgsolver.Transform(var2);
      var3 = imsrgsolver.Transform(var3);
      var4 = imsrgsolver.Transform(var4);

      var2 = var2.UndoNormalOrdering();
      var3 = var3.UndoNormalOrdering();
      var4 = var4.UndoNormalOrdering();

      var2.SetModelSpace(ms2);
      var3.SetModelSpace(ms2);
      var4.SetModelSpace(ms2);

      var2 = var2.DoNormalOrdering();
      var3 = var3.DoNormalOrdering();
      var4 = var4.DoNormalOrdering();

      rw.WriteTokyo(var2, vs_out + "_QQ2_IM.snt", "op");
      rw.WriteTokyo(var3, vs_out + "_QQ3_IM.snt", "op");
      rw.WriteTokyo(var4, vs_out + "_QQ4_IM.snt", "op");
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Write the original and evolved Hamiltonian in the HO basis /////////////////
    ///////////////////////////////////////////////////////////////////////////////

    if (write_H)
    {
      int emax_imsrg = eMax;

      // H(0): generate vacuum represetations (HF basis)
      H_HF = H_HF.UndoNormalOrdering();

      // Transformation from HF to HO basis
      arma::mat CHO  = hf.C;
      arma::mat CHOt = CHO.t();

      // Transform H(0) from the HF to the HO basis and write to file
      Operator H0_HO = H_HF.Transform(CHOt);

      rw.Write_me1j(intfile + ".me1j.gz", H0_HO, emax_imsrg, emax_imsrg);                    // One-body part
      rw.Write_me2jp(intfile + ".me2jp.gz", H0_HO, emax_imsrg, 2 * emax_imsrg, emax_imsrg);  // Two-body part

      bool write_evolved_H = false;

      if (smax != 0. && write_evolved_H)
      {
        // H(s): generate vacuum represetations (HF basis)
        Hs = Hs.UndoNormalOrdering();

        // Transform H(0) from the HF to the HO basis and write to file
        Operator Hs_HO = Hs.Transform(CHOt);

        rw.Write_me1j (intfile + "_s" + std::to_string(smax) + ".me1j.gz",  Hs_HO, emax_imsrg, emax_imsrg);                  // One-body part
        rw.Write_me2jp(intfile + "_s" + std::to_string(smax) + ".me2jp.gz", Hs_HO, emax_imsrg, 2 * emax_imsrg, emax_imsrg);  // Two-body part
      }        
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Print the Magnus operator, Andrea 30/01/2025 ///////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////

    if (write_omega_me)
    {
      std::cout << "--------- Writing Omega ---------" << std::endl; 
      int emax_imsrg = eMax;

      std::cout << "Number of Omega[i]:\t" << imsrgsolver.Omega.size() << std::endl;
      std::cout << std::endl;

      std::cout << "Printing the BCH-contracted operator" << std::endl;
      std::cout << std::endl;

      Operator Omega = imsrgsolver.GetOmega(0);

      for (size_t i = 1; i < imsrgsolver.Omega.size(); ++i)
      {
        Operator step = imsrgsolver.GetOmega(i);
        Omega = Commutator::BCH_Product(Omega, step);
      }

      rw.Write_me1j (omefile + ".me1j.gz",  Omega, emax_imsrg, emax_imsrg);                  // One-body part
      rw.Write_me2jp(omefile + ".me2jp.gz", Omega, emax_imsrg, 2 * emax_imsrg, emax_imsrg);  // Two-body part

      std::cout << std::endl;
      std::cout << "Printing the Omega[i] operators" << std::endl;
      std::cout << std::endl;

      for (size_t i = 0; i < imsrgsolver.Omega.size(); ++i)
      {
        Operator Omega_i = imsrgsolver.Omega[i];
        
        rw.Write_me1j (omefile + "_" + std::to_string(i) + ".me1j.gz",  Omega_i, emax_imsrg, emax_imsrg);                  // One-body part
        rw.Write_me2jp(omefile + "_" + std::to_string(i) + ".me2jp.gz", Omega_i, emax_imsrg, 2 * emax_imsrg, emax_imsrg);  // Two-body part
      }
    }

    ///////////////////////////////////////////////////////////////////////////////
    // New project on correlated diabatic surfaces/ ///////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////

    bool diabatic_surf = false;

    if (diabatic_surf)
    {
      int L_dia = 2;
      std::string pn = "isoscalar";

      Operator Q  = imsrg_util::ElectricMultipoleOp(modelspace, L_dia, 0, pn);

      Operator T = imsrg_util::KineticEnergy_Op(modelspace);

      


    }

    ///////////////////////////////////////////////////////////////////////////////

    for (size_t i = 0; i < opnames.size(); ++i)
    {
      auto opname = opnames[i];
      std::cout << i << ": " << opname << " " << std::endl;

      Operator op;

      if (opname.find("_FROMFILE") != std::string::npos)
      {
        OpFromFile &opff = opsfromfile_unpacked[count_from_file];
        std::cout << "reading " << opff.opname << " with " << opff.j << " " << opff.t << " " << opff.p << " " << opff.r << "  from file " << opff.file2name << std::endl;
        op = Operator(modelspace, opff.j, opff.t, opff.p, opff.r);
        if (opff.r > 2)
        {
          op.ThreeBody.SetMode("no2b");
          op.ThreeBody.Allocate();
        }
        if (input_op_fmt == "navratil")
        {
          rw.Read2bCurrent_Navratil(opff.file2name, op);
        }
        else if (input_op_fmt == "miyagi")
        {
          if (opff.file2name != "")
          {
            Operator optmp = rw.ReadOperator2b_Miyagi(opff.file2name, modelspace);
            op.TwoBody = optmp.TwoBody;
          }
          if (opff.r > 2 and opff.file3name != "")
            rw.Read_Darmstadt_3body(opff.file3name, op, file3e1max, file3e2max, file3e3max);
        }
        else if (input_op_fmt == "heinz")
        {
          if (opff.r > 2 and opff.file3name != "")
          {

            op.ThreeBody.SetMode("no2b");
            if (no2b_precision == "half")
              op.ThreeBody.SetMode("no2bhalf");

            op.ThreeBody.ReadFile({opff.file3name}, {file3e1max, file3e2max, file3e3max, file3e1max});
            // rw.Read_Darmstadt_3body( opff.file3name, op,  file3e1max,file3e2max,file3e3max);
          }
        }
        count_from_file++;
        opname = opff.opname; // Get rid of the _FROMFILE bit.
      }
      else
      {
        op = imsrg_util::OperatorFromString(modelspace, opname);
      }

      // Added by Antoine Belley
      if (write_HO_ops)
      {
        std::cout << "writing HO tensor files " << std::endl;
        if (valence_file_format == "tokyo")
        {
          rw.WriteTensorTokyo(intfile + opnames[i] + "_HO_2b.snt", op);
        }
        else
        {
          rw.WriteTensorOneBody(intfile + opnames[i] + "_HO_1b.op", op, opnames[i]);
          rw.WriteTensorTwoBody(intfile + opnames[i] + "_HO_2b.op", op, opnames[i]);
        }
      }

      if (basis == "oscillator" or opname == "OccRef")
      {
        op = op.DoNormalOrdering();
      }
      else if (basis == "HF")
      {
        op = hf.TransformToHFBasis(op).DoNormalOrdering();
      }
      else if (basis == "NAT")
      {
        op = hf.TransformHOToNATBasis(op).DoNormalOrdering();
      }
      std::cout << std::setprecision(24) << "   HF: " << op.ZeroBody << std::endl;
      std::cout << opname << "_HF: " << op.ZeroBody << std::endl;

      if ((eMax_imsrg != -1) or (e2Max_imsrg != -1) or (e3Max_imsrg) != -1)
      {
        std::cout << "Truncating modelspace for IMSRG calculation: emax e2max e3max  ->  " << eMax_imsrg << " " << e2Max_imsrg << " " << e3Max_imsrg << std::endl;
        op = op.Truncate(modelspace_imsrg);
      }

      // Added by Antoine Belley
      if (write_HF_ops)
      {
        std::cout << "writing HF tensor files " << std::endl;
        if (valence_file_format == "tokyo")
        {
          rw.WriteTensorTokyo(intfile + opnames[i] + "_HF_2b.snt", op);
        }
        else
        {
          rw.WriteTensorOneBody(intfile + opnames[i] + "_HF_1b.op", op, opnames[i]);
          rw.WriteTensorTwoBody(intfile + opnames[i] + "_HF_2b.op", op, opnames[i]);
        }
      }

      // Unclear whether we should do NO2B here as well...
      // std::cout << "Before renormal ordering Op(5,4) is " << std::setprecision(10) << op.OneBody(5,4) << std::endl;
      if (renormal_order)
      {
        op = op.UndoNormalOrdering();
        op.SetModelSpace(ms2);
        op = op.DoNormalOrdering();
        if ((op.GetJRank() == 0) && (op.GetTRank() == 0) && (op.GetParity() == 0))
        {
          rw.Write_NaiveVS1B(intfile + opname + ".vs1b", op);
          rw.Write_NaiveVS2B(intfile + opname + ".vs2b", op);
          rw.Write_me1j(intfile + opname + "_coreNO" + emax_ref_string + ".me1j", op, emax_reference, emax_reference);
          rw.Write_me2jp(intfile + opname + "_coreNO" + emax_ref_string + ".me2jp", op, emax_reference, 2 * emax_reference, emax_reference);
        }
      }
      //      std::cout << " (" << ops[i].ZeroBody << " ) " << std::endl;
      std::cout << "   IMSRG: " << op.ZeroBody << std::endl;
      std::cout << opname << "_IMSRG: " << op.ZeroBody << std::endl;
      //      rw.WriteOperatorHuman(ops[i],intfile+opnames[i]+"_step2.op");
      //      std::cout << "After renormal ordering Op(5,4) is " << std::setprecision(10) << op.OneBody(5,4) << std::endl;

      std::cout << "      " << op.GetJRank() << " " << op.GetTRank() << " " << op.GetParity() << "   " << op.GetNumberLegs() << std::endl;
      if (((op.GetJRank() + op.GetTRank() + op.GetParity()) < 1) and (op.GetNumberLegs() % 2 == 0))
      {
        std::cout << "writing scalar files " << std::endl;
        if (valence_file_format == "tokyo")
        {
          rw.WriteTokyo(op, intfile + opname + ".snt", "op");
        }
        else
        {
          rw.WriteNuShellX_op(op, intfile + opname + ".int");
        }
      }
      else if (op.GetNumberLegs() % 2 == 1) // odd number of legs -> this is a dagger operator
      {
        rw.WriteDaggerOperator(op, intfile + opname + ".dag", opname);
      }
      else
      {
        std::cout << "writing tensor files " << std::endl;
        if (valence_file_format == "tokyo")
        {
          rw.WriteTensorTokyo(intfile + opname + "_2b.snt", op);
        }
        else
        {
          rw.WriteTensorOneBody(intfile + opname + "_1b.op", op, opname);
          rw.WriteTensorTwoBody(intfile + opname + "_2b.op", op, opname);
        }
      }

    } // for opnames

  }// if method == "magnus"

  if (method == "flow" or method == "flow_RK4")
  {
    for (size_t i = 0; i < ops.size(); ++i)
    {
      auto op = imsrgsolver.GetOperator(i + 1); // the zero-th operator is the Hamiltonian
      auto opname = opnames[i];

      if (renormal_order)
      {
        op = op.UndoNormalOrdering();
        op.SetModelSpace(ms2);
        op = op.DoNormalOrdering();
      }
      std::cout << "   IMSRG: " << op.ZeroBody << std::endl;

      std::cout << "      " << op.GetJRank() << " " << op.GetTRank() << " " << op.GetParity() << "   " << op.GetNumberLegs() << std::endl;
      if (((op.GetJRank() + op.GetTRank() + op.GetParity()) < 1) and (op.GetNumberLegs() % 2 == 0))
      {
        std::cout << "writing scalar files " << std::endl;
        if (valence_file_format == "tokyo")
        {
          rw.WriteTokyo(op, intfile + opname + ".snt", "op");
        }
        else
        {
          rw.WriteNuShellX_op(op, intfile + opname + ".int");
        }
      }
      else if (op.GetNumberLegs() % 2 == 1) // odd number of legs -> this is a dagger operator
      {
        rw.WriteDaggerOperator(op, intfile + opname + ".dag", opname);
      }
      else
      {
        std::cout << "writing tensor files " << std::endl;
        if (valence_file_format == "tokyo")
        {
          rw.WriteTensorTokyo(intfile + opname + "_2b.snt", op);
        }
        else
        {
          rw.WriteTensorOneBody(intfile + opname + "_1b.op", op, opname);
          rw.WriteTensorTwoBody(intfile + opname + "_2b.op", op, opname);
        }
      }
    }
  }

  if (write_omega)
  {
    std::string scratch = rw.GetScratchDir();
    imsrgsolver.FlushOmegaToScratch();
    for (int i=0; i < imsrgsolver.GetNOmegaWritten() ; i++)
    {
       std::ostringstream inputfile,outputfile;
       inputfile << scratch << "/OMEGA_" << std::setw(6) << std::setfill('0') << getpid() << std::setw(3) << std::setfill('0') << i;
       outputfile << intfile << "_Omega_" << i;
       // rw.CopyFile( inputfile.str(), outputfile.str() );
    }
    //    rw.WriteOmega(intfile,scratch, imsrgsolver.n_omega_written);



    std::ofstream file_occ;
    std::ostringstream name_occ;
    int wint = 4; int wdouble = 26; int pdouble = 16;
    name_occ << intfile << "_occ.dat";
    file_occ.open( name_occ.str(), std::ofstream::out);
    for (auto i : modelspace.all_orbits)
    {
      Orbit& oi = modelspace.GetOrbit(i);
      if ( std::abs(oi.occ)>1e-6 )
      {
        file_occ << std::setw(wint) << oi.n << std::setw(wint) << oi.l << std::setw(wint) << oi.j2 << std::setw(wint) << oi.tz2
                 << std::setw(wdouble) << std::setiosflags(std::ios::fixed) << std::setprecision(pdouble) << std::scientific << oi.occ << std::endl;
      }
    }
    file_occ.close();
    if (basis == "NAT")
    {
      name_occ.str("");
      name_occ << intfile << "_occ_nat.dat";
      file_occ.open( name_occ.str(), std::ofstream::out);
      for (auto i : modelspace.all_orbits)
      {
        Orbit& oi = modelspace.GetOrbit(i);
        if ( std::abs(oi.occ_nat)>1e-6 )
        {
          file_occ << std::setw(wint) << oi.n << std::setw(wint) << oi.l << std::setw(wint) << oi.j2 << std::setw(wint) << oi.tz2
                   << std::setw(wdouble) << std::setiosflags(std::ios::fixed) << std::setprecision(pdouble) << std::scientific << oi.occ_nat << std::endl;
        }
      }
      file_occ.close();
    }

    bool filesucess = false;
    if (basis == "HF")
    {
       filesucess = hf.C.save(intfile+"C.mat");
    }
    else if (basis == "NAT")
    {
       filesucess = hf.C_HO2NAT.save(intfile+"C.mat");
    }



    //    bool filesucess = hf.C.save(intfile+"C.mat");
    if (filesucess == false)
    {
      std::cout<<"Couldn't save HF coefficient matrix."<<std::endl;
    }
    // std::cout << "writing Omega to " << intfile << "_omega.op" << std::endl;
    // rw.WriteOperatorHuman(imsrgsolver.Omega.back(),intfile+"_omega.op");
  }
  Hbare.PrintTimes();

  return 0;
}