#ifndef SOLVE_MF_PARAMS_H_
#define SOLVE_MF_PARAMS_H_

#include <string>

#include "fmt/core.h"
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
  bool with_generator = false;
  std::string generator = "imaginary-time";
  std::string denominator = "Moller_Plesset";
  bool with_commutators = false;
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

inline void HandleDefaultMFSolverArgs(MFSolverArgs &args) {
  MFSolverArgs defaults;

  if (args.input_2bme_e2max == defaults.input_2bme_e2max) {
    args.input_2bme_e2max = 2 * args.input_2bme_emax;
  }
  if (args.input_2bme_lmax == defaults.input_2bme_lmax) {
    args.input_2bme_lmax = args.input_2bme_emax;
  }

  if (args.input_3bme_e2max == defaults.input_3bme_e2max) {
    args.input_3bme_e2max = 2 * args.input_3bme_emax;
  }

  if (args.mass == defaults.mass) {
    std::size_t last_index =
        args.reference_state.find_last_not_of("0123456789");
    args.mass = std::stoi(args.reference_state.substr(last_index + 1));
  }

  if (args.calc_lmax == defaults.calc_lmax) {
    args.calc_lmax = args.calc_emax;
  }
  if (args.calc_lmax3 == defaults.calc_lmax3) {
    args.calc_lmax3 = args.calc_emax;
  }

  if (args.valence_space == defaults.valence_space) {
    args.valence_space = args.reference_state;
  }
}

inline MFSolverArgs ParseMFSolverArgs(int argc, char **argv) {
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
      [](const std::string &s) {
        if (s.find_last_of("/\\") == s.npos) {
          return std::string("");
        }
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
  app.add_flag("--with-generator", args.with_generator,
               "Compute and save matrix elements for generator "
               "(default: false).");
  app.add_option("--generator", args.generator,
                 "Type of generator to be used for commutators (default: "
                 "imaginary-time).")
      ->check(CLI::IsMember({"white", "atan", "imaginary-time"}));
  app.add_option(
         "--denominator", args.denominator,
         "Type of energy denominator for generator construction(default: "
         "Moller_Plesset).")
      ->check(CLI::IsMember({"Moller_Plesset", "Epstein_Nesbet"}));
  app.add_flag("--with-commutators", args.with_commutators,
               "Compute IMSRG(2) commutators and saved output "
               "(default: false).");

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
  } catch (const CLI::ParseError &e) {
    app.exit(e);
    exit(EXIT_FAILURE);
  }

  HandleDefaultMFSolverArgs(args);

  return args;
}

inline std::string PrettyPrintMFSolverArgs(const MFSolverArgs &args) {
  std::string ret_val = "";

  ret_val += "Input 2-body matrix elements:\n";
  ret_val += fmt::format("path = {}\n", args.path_to_input_2bme);
  ret_val += fmt::format("fmt = {}\n", args.input_2bme_fmt);
  ret_val +=
      fmt::format("emax, e2max, lmax = {}, {}, {}\n\n", args.input_2bme_emax,
                  args.input_2bme_e2max, args.input_2bme_lmax);

  ret_val += "Input 3-body matrix elements:\n";
  ret_val += fmt::format("path = {}\n", args.path_to_input_3bme);
  ret_val += fmt::format("type = {}\n", args.input_3bme_type);
  ret_val += fmt::format("fmt = {}\n", args.input_3bme_fmt);
  ret_val +=
      fmt::format("emax, e2max, e3max = {}, {}, {}\n\n", args.input_3bme_emax,
                  args.input_3bme_e2max, args.input_3bme_e3max);

  ret_val += fmt::format("hbar omega = {}\n\n", args.hbar_omega);

  ret_val += "Output NO2B matrix elements:\n";
  ret_val += fmt::format("emax = {}\n", args.output_me_emax);
  ret_val += fmt::format("path to 1B MEs = {}\n", args.path_to_output_1bme);
  ret_val += fmt::format("path to 2B MEs = {}\n\n", args.path_to_output_2bme);

  ret_val += "Extra output matrix elements:\n";
  ret_val += fmt::format("save generator = {}\n", args.with_generator);
  ret_val += fmt::format("save commutators = {}\n", args.with_commutators);
  ret_val += fmt::format("generator type = {}, denominator type = {}\n\n",
                         args.generator, args.denominator);

  ret_val += "Mean-field calculation:\n";
  ret_val += fmt::format("Reference state: {}, A = {}\n", args.reference_state,
                         args.mass);
  ret_val += fmt::format("basis = {}\n", args.basis);
  ret_val += fmt::format("emax, lmax, e3max, lmax3 = {}, {}, {}, {}\n\n",
                         args.calc_emax, args.calc_lmax, args.calc_e3max,
                         args.calc_lmax3);

  ret_val += "Meta-data specifications:\n";
  ret_val +=
      fmt::format("path to metadata file = {}\n", args.path_to_metadata_file);
  ret_val += fmt::format("LEC specifier = {}\n\n", args.lec_string);

  ret_val += "Additional details:\n";
  ret_val += fmt::format("Nucleon mass correction: {}\n",
                         args.nucleon_mass_correction);
  ret_val += fmt::format("Relativistic correction: {}\n",
                         args.relativistic_correction);
  ret_val += fmt::format("Lawson term coefficient = {}\n", args.beta_cm);
  ret_val +=
      fmt::format("Lawson term frequency = {}\n\n", args.hbar_omega_beta_cm);

  ret_val += "Advanced parameters:\n";
  ret_val += fmt::format("no2b_precision: {}\n", args.no2b_precision);
  ret_val += fmt::format("valence_space: {}\n", args.valence_space);
  ret_val += fmt::format("nat_order: {}\n", args.nat_order);
  ret_val += fmt::format("freeze_occupations: {}\n", args.freeze_occupations);
  ret_val +=
      fmt::format("discard_no2b_from_3n: {}\n", args.discard_no2b_from_3n);
  ret_val += fmt::format("use_nat_occupations: {}\n", args.use_nat_occupations);
  ret_val += fmt::format("emax_unoccupied: {}\n", args.emax_unoccupied);

  return ret_val;
}

#endif // SOLVE_MF_PARAMS_H_
