#ifndef SOLVE_MF_PARAMS_H_
#define SOLVE_MF_PARAMS_H_

#include <string>

#include "lib/cli11/CLI11.hpp"

struct MFSolverArgs {
  // Inputs
  std::string path_to_input_2bme = "";
  std::string input_2bme_fmt = "me2j";
  int input_2bme_emax = 0;
  int input_2bme_e2max = -1;
  int input_2bme_lmax = -1;
  std::string path_to_input_3bme = "none";
  std::string input_3bme_type = "full";
  std::string input_3bme_fmt = "me3j";
  int input_3bme_emax = 0;
  int input_3bme_e2max = -1;
  int input_3bme_e3max = 0;
  double hbar_omega = 0.0;
  // Outputs
  std::string path_to_output_1bme = "";
  std::string path_to_output_2bme = "";
  int output_me_emax = 0;
  // Mean-field calculation
  std::string reference_state = "";
  int mass = -1;
  std::string basis = "HF";
  int calc_emax = 0;
  int calc_lmax = -1;
  int calc_e3max = 0;
  int calc_lmax3 = -1;
  // Details for output metadata
  std::string path_to_metadata_file = "";
  std::string lec_string = "standard_LECs";
  // Corrections to T_rel
  bool nucleon_mass_correction = false;
  bool relativistic_correction = false;
  // Lawson term params
  double beta_cm = 0.0;
  double hbar_omega_beta_cm = 0.0;
  // Advanced parameters
  std::string no2b_precision = "single";
  std::string valence_space = "";
  std::string nat_order = "occupation";
  bool freeze_occupations = false;
  bool discard_no2b_from_3n = false;
  bool use_nat_occupations = false;
  int emax_unoccupied = -1;
};

inline MFSolverArgs DefaultMFSolverArgs() { return MFSolverArgs(); }

inline MFSolverArgs ParseMFSolverArgs(int argc, char** argv) {
  MFSolverArgs args;

  CLI::App app(
      "Mean-field solver to produce (un)normal-ordered matrix elements.");

  // 2-body input file
  app.add_option("--path-to-input-2bme", args.path_to_input_2bme,
                 "Path to input 2-body potential matrix elements.")
      ->required()
      ->check(CLI::ExistingFile);
  app.add_option(
      "--input-2bme-format", args.input_2bme_fmt,
      "Format of input 2-body potential matrix elements (default: me2j).");

  // 2-body file truncations
  app.add_option("--input-2bme-emax", args.input_2bme_emax,
                 "emax of input 2-body potential matrix element file.")
      ->required()
      ->check(CLI::NonNegativeNumber);
  app.add_option("--input-2bme-e2max", args.input_2bme_e2max,
                 "e2max of input 2-body potential matrix element file "
                 "(default: 2*emax).")
      ->check(CLI::NonNegativeNumber | CLI::IsMember({-1}));
  app.add_option(
         "--input-2bme-lmax", args.input_2bme_lmax,
         "lmax of input 2-body potential matrix element file (default: emax).")
      ->check(CLI::NonNegativeNumber | CLI::IsMember({-1}));

  // 3-body input file
  app.add_option(
         "--path-to-input-3bme", args.path_to_input_3bme,
         "Path to input 3-body potential matrix elements (default: none).")
      ->check(CLI::ExistingFile | CLI::IsMember({"none"}));
  app.add_option(
      "--input-3bme-type", args.input_3bme_type,
      "Type of input 3-body potential matrix elements (default: full).");
  app.add_option(
      "--input-3bme-format", args.input_3bme_fmt,
      "Format of input 3-body potential matrix elements (default: me3j).");

  // 3-body file truncations
  app.add_option(
         "--input-3bme-emax", args.input_3bme_emax,
         "emax of input 3-body potential matrix element file (default: 0).")
      ->check(CLI::NonNegativeNumber);
  app.add_option("--input-3bme-e2max", args.input_3bme_e2max,
                 "e2max of input 2-body potential matrix element file "
                 "(default: 2*emax).")
      ->check(CLI::NonNegativeNumber | CLI::IsMember({-1}));
  app.add_option(
         "--input-3bme-e3max", args.input_3bme_e3max,
         "e3max of input 3-body potential matrix element file (default: 0).")
      ->check(CLI::NonNegativeNumber);

  // hw of input files
  app.add_option("--hbar-omega", args.hbar_omega,
                 "Oscillator frequency hbar omega of input files.")
      ->required();

  auto existing_path_validator = CLI::Validator(
      [](const std::string& s) {
        auto parent = s.substr(0, s.find_last_of("/\\"));
        auto path_type = CLI::detail::check_path(parent.c_str());
        if (path_type == CLI::detail::path_type::directory) {
          return std::string("");
        }
        return "Path to file does not exist: " + s;
      },
      "EXISTING_PATH", "ExistingPathToFileValidator");

  // Output NO2B files
  app.add_option("--path-to-output-1bme", args.path_to_output_1bme,
                 "Path to output normal-ordered 1-body matrix elements.")
      ->required()
      ->check(existing_path_validator);
  app.add_option("--path-to-output-2bme", args.path_to_output_2bme,
                 "Path to output normal-ordered 2-body matrix elements.")
      ->required()
      ->check(existing_path_validator);
  app.add_option("--output-emax", args.output_me_emax,
                 "emax of output matrix element files.")
      ->required()
      ->check(CLI::NonNegativeNumber);

  // Mean-field calculation details
  app.add_option("--reference-state", args.reference_state,
                 "Reference state in format elem(Z)A (e.g. O16, Ca40).")
      ->required();
  app.add_option("--target-mass", args.mass,
                 "Mass of target system (default: same as reference).")
      ->check(CLI::NonNegativeNumber | CLI::IsMember({-1}));
  app.add_option("--basis", args.basis,
                 "Type of basis to construct (default: HF).")
      ->check(CLI::IsMember({"HO", "HF", "NAT"}));

  // Model-space size for calculation
  app.add_option("--calc-emax", args.calc_emax,
                 "Model-space size emax to be used in calculation.")
      ->required();
  app.add_option("--calc-lmax", args.calc_lmax,
                 "Model-space truncation lmax to be used in calculation "
                 "(default: emax).")
      ->check(CLI::NonNegativeNumber | CLI::IsMember({-1}));
  app.add_option("--calc-e3max", args.calc_e3max,
                 "Model-space 3-body truncation e3max to be used in "
                 "calculation (default: 0).")
      ->check(CLI::NonNegativeNumber);
  app.add_option("--calc-lmax3", args.calc_lmax3,
                 "Model-space truncation lmax to be used for 3-body operator "
                 "in calculation "
                 "(default: emax).")
      ->check(CLI::NonNegativeNumber | CLI::IsMember({-1}));

  // Meta-data
  app.add_option("--path-to-metadata-file", args.path_to_metadata_file,
                 "Path to file for meta-data.")
      ->required()
      ->check(existing_path_validator);
  app.add_option("--LEC-string", args.lec_string,
                 "Optional \"LEC\" string with additional information "
                 "(default: standard_LECs).");

  // T_rel corrections
  app.add_flag("--include-nucleon-mass-correction",
               args.nucleon_mass_correction,
               "Account for mass difference between protons and neutrons "
               "(default: false).");
  app.add_flag("--include-relativistic-correction",
               args.relativistic_correction,
               "Add a relativistic correction (default: false).");

  // Lawson term
  app.add_option(
      "--beta-cm", args.beta_cm,
      "Coefficient for Lawson term (for center-of-mass investigations).");
  app.add_option("--hbar-omegabeta-cm", args.hbar_omega_beta_cm,
                 "Oscillator frequency for Lawson term (for center-of-mass "
                 "investigations).");

  // Advanced parameters
  // Currently unsupported

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    app.exit(e);
    exit(EXIT_FAILURE);
  }

  return args;
}

inline std::string PrettyPrintMFSolverArgs(const MFSolverArgs& args) {}

#endif  // SOLVE_MF_PARAMS_H_