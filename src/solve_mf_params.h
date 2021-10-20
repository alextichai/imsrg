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

inline MFSolverArgs DefaultMFSolverArgs() {
    return MFSolverArgs();
}

inline MFSolverArgs ParseMFSolverArgs(int argc, char** argv) {
    MFSolverArgs args;

    CLI::App app("Mean-field solver to produce (un)normal-ordered matrix elements.");

    app.add_option(
        "--path-to-input-2bme",
        args.path_to_input_2bme,
        "Path to input 2-body potential matrix elements.")->required();

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        app.exit(e);
        exit(EXIT_FAILURE);
    }

    return args;
}

inline std::string PrettyPrintMFSolverArgs(const MFSolverArgs& args) {

}

#endif  // SOLVE_MF_PARAMS_H_
