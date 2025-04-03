#ifndef imsrg_response_hh
#define imsrg_response_hh 1

#include "ModelSpace.hh"
#include "Operator.hh"
#include "HartreeFock.hh"
#include "HFMBPT.hh"
#include "IMSRGSolver.hh"
#include "imsrg_util.hh"
#include "PhysicalConstants.hh"
#include <gsl/gsl_math.h>
#include <vector>
#include <array>
#include <boost/format.hpp>

namespace imsrg_response
{
    // Excitation operators (electromagnetic probe)

    Operator C_Op (ModelSpace &modelspace, int L, double q); // Longitudinal (Coulomb) multipole transition operator
    Operator C_Sub(ModelSpace &modelspace, double q);        // Longitudinal (Coulomb) monopole transition operator (for L = 0 m0)
    Operator TE_Op(ModelSpace &modelspace, int L, double q); // Transverse Electric multipole transition operator
    // TODO Operator TM_Op(ModelSpace &modelspace, int L, double q); // Transverse Magnetic multipole transition operator

    // Excitation operators (Isoscalar probe)

    Operator IS_Op (ModelSpace &modelspace, int L, double q);
    Operator IS_Sub(ModelSpace &modelspace, double q);
    Operator IV_Op (ModelSpace &modelspace, int L, double q);

    // "Measuring" operators (Isoscalar and Isovector multipole operators at q = 0), also usable for q = 0 limit of excitation operators

    Operator Trans(ModelSpace  &modelspace, int L, std::string pn)  // Electric multipole operator for transition // TODO add magnetic operators
    {
        if (L != 0)
            return imsrg_util::ElectricMultipoleOp(modelspace, L, 0, pn);
        else
            return imsrg_util::ElectricMultipoleOp(modelspace, 0, 2, pn);
    };
    Operator T_Sub(ModelSpace  &modelspace, std::string pn) // Subracting L = 0 for m0
    {
        return 1. / sqrt(4 * PhysConst::PI) * imsrg_util::RSquaredOp(modelspace, pn);
    };

    // Radial Integrals

    double RadialIntegral_Bessel(int na, int la, int nb, int lb, int L, double q, ModelSpace &modelspace);
    double RadialIntegral_TE(int na, int la, double ja, int nb, int lb, int tz2, int L, double q, ModelSpace &modelspace);

    double HO_gr(int n, int l, double x);
    double jL_ho(int na, int la, int nb, int lb, int L, double y);

    // Kernels evaluation

    Operator Mix0(ModelSpace& modelspace, const Operator& L, const Operator& R);
    Operator Mix1(ModelSpace& modelspace, const Operator& H, const Operator& L, const Operator& R);

    // Add "routines" for syntetic calls in the main

    struct KernelParams
    {
        ModelSpace* modelspace;
        HFMBPT*     hf;

        int N_Magnus;

        std::string omefile;
        std::string kerdir;
    };

    Operator SetOperator(ModelSpace& modelspace, double q, int L, std::string field);
    Operator SetSub(ModelSpace& modelspace, double q, std::string field);

    void computeKernel(double qL, double qR, Operator H, int L, std::string fL, std::string fR, KernelParams par);

    // Print functions

    void printKernel(std::ostream& out, std::string fL, double qL, std::string fR, double qR, double mom0_0, double mom1_0, double mom0_s, double mom1_s);

}

#endif