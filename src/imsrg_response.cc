#include "imsrg_response.hh"
#include "AngMom.hh"
#include "Commutator.hh"
#include "GaussLaguerre.hh"
#include "DarkMatterNREFT.hh"
#include "M0nu.hh"
#include "omp.h"
#include <cstdlib>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf_bessel.h> // to use bessel functions
#include <gsl/gsl_sf_laguerre.h>
#include <gsl/gsl_sf_gamma.h>  // for gsl_sf_gamma,  gsl_sf_fact, gsl_sf_doublefact
#include <gsl/gsl_sf_hyperg.h> // for gsl_sf_hyperg_1F1
#include <iostream>
#include <fstream>
#include <iomanip>
#include <math.h>
#include <vector>
#include <string>
#include <map>
#include <array>

/// imsrg_util namespace. Used to define some helpful functions.
namespace imsrg_response
{

using PhysConst::HBARC;
using PhysConst::M_PROTON;
using PhysConst::M_NEUTRON;
using PhysConst::M_NUCLEON;
using PhysConst::M_ELECTRON;
using PhysConst::PROTON_SPIN_G;
using PhysConst::NEUTRON_SPIN_G;
using PhysConst::ELECTRON_SPIN_G;
using PhysConst::F_PI;
using PhysConst::ALPHA_FS;
using PhysConst::PI;
using PhysConst::SQRT2;
using PhysConst::SQRTPI;
using PhysConst::INVSQRT2;
using PhysConst::LOG2;

/// Returns the q-dependent longitudinal electric (Coulomb) multipole transition operator. In the q->0 limit this is strictly equivalent to the LWA definition
Operator C_Op(ModelSpace &modelspace, int L, double q)
{
  Operator EL(modelspace, L, 0, L % 2, 2);

  if (q == 0.)
  {
    if (L != 0)
      EL = imsrg_util::ElectricMultipoleOp(modelspace, L, 0); // Exact limit for q -> 0
    else
      EL = imsrg_util::ElectricMultipoleOp(modelspace, 0, 2); // NOT the limit for q -> 0, the constant term (1) has been neglected
  }
  else
  {
    for (int i : modelspace.proton_orbits) // Coulomb only interacts with protons
    {
      Orbit &oi = modelspace.GetOrbit(i);
      double ji = 0.5 * oi.j2;

      for (int j : EL.OneBodyChannels.at({oi.l, oi.j2, oi.tz2}))
      {
        if (j < i)
          continue;

        Orbit &oj = modelspace.GetOrbit(j);
        double jj = 0.5 * oj.j2;

        double r2int = RadialIntegral_Bessel(oi.n, oi.l, oj.n, oj.l, L, q, modelspace);

        EL.OneBody(i, j) = (1 + modelspace.phase(oi.l + oj.l + L)) / 2. * modelspace.phase(jj + L - 0.5) * sqrt((2 * ji + 1) * (2 * jj + 1) * (2 * L + 1) / 4. / PI) * AngMom::ThreeJ(ji, jj, L, 0.5, -0.5, 0) * r2int;
        EL.OneBody(j, i) = modelspace.phase((oi.j2 + oj.j2) / 2 + 1) * EL.OneBody(i, j);
      }
    }
    EL *= gsl_sf_doublefact(2 * L + 1) / pow(q, L);
  }
  return EL;
}

/// Returns the q-dependent isoscalar multipole transition operator. In the q->0 limit this is strictly equivalent to the LWA definition
Operator IS_Op(ModelSpace &modelspace, int L, double q)
{
  Operator IS(modelspace, L, 0, L % 2, 2);

  if (q == 0.)
  {
    if (L != 0)
      IS = imsrg_util::ElectricMultipoleOp(modelspace, L, 0, "isoscalar"); // Exact limit for q -> 0
    else
      IS = imsrg_util::ElectricMultipoleOp(modelspace, 0, 2, "isoscalar"); // NOT the limit for q -> 0, the constant term (1) has been neglected
  }
  else
  {
    for (int i : modelspace.all_orbits)
    {
      Orbit &oi = modelspace.GetOrbit(i);
      double ji = 0.5 * oi.j2;

      for (int j : IS.OneBodyChannels.at({oi.l, oi.j2, oi.tz2}))
      {
        if (j < i)
          continue;

        Orbit &oj = modelspace.GetOrbit(j);
        double jj = 0.5 * oj.j2;

        double r2int = RadialIntegral_Bessel(oi.n, oi.l, oj.n, oj.l, L, q, modelspace);

        // if (L == 0 && i == j)
        //   r2int -= 1.;

        IS.OneBody(i, j) = (1 + modelspace.phase(oi.l + oj.l + L)) / 2. * modelspace.phase(jj + L - 0.5) * sqrt((2 * ji + 1) * (2 * jj + 1) * (2 * L + 1) / 4. / PI) * AngMom::ThreeJ(ji, jj, L, 0.5, -0.5, 0) * r2int;
        IS.OneBody(j, i) = modelspace.phase((oi.j2 + oj.j2) / 2 + 1) * IS.OneBody(i, j);
      }
    }
    // if (L == 0)
      // IS *= -6. / pow(q, 2);
      // IS *= 15. / pow(q, 2);
    
    IS *= gsl_sf_doublefact(2 * L + 1) / pow(q, L);
  }
  return IS;
}

/// Returns the q-dependent isovector multipole transition operator. In the q->0 limit this is strictly equivalent to the LWA definition
Operator IV_Op(ModelSpace &modelspace, int L, double q)
{
  Operator IV(modelspace, L, 0, L % 2, 2);

  if (q == 0.)
  {
    IV = imsrg_util::ElectricMultipoleOp(modelspace, L, 0, "isovector"); // Exact limit for q -> 0
  }
  else
  {
    for (int i : modelspace.all_orbits)
    {
      Orbit &oi = modelspace.GetOrbit(i);
      double ji = 0.5 * oi.j2;

      for (int j : IV.OneBodyChannels.at({oi.l, oi.j2, oi.tz2}))
      {
        if (j < i)
          continue;

        Orbit &oj = modelspace.GetOrbit(j);
        double jj = 0.5 * oj.j2;

        double r2int = RadialIntegral_Bessel(oi.n, oi.l, oj.n, oj.l, L, q, modelspace);

        double iv_ch = oi.tz2;

        IV.OneBody(i, j) = iv_ch * (1 + modelspace.phase(oi.l + oj.l + L)) / 2. * modelspace.phase(jj + L - 0.5) * sqrt((2 * ji + 1) * (2 * jj + 1) * (2 * L + 1) / 4. / PI) * AngMom::ThreeJ(ji, jj, L, 0.5, -0.5, 0) * r2int;
        IV.OneBody(j, i) = modelspace.phase((oi.j2 + oj.j2) / 2 + 1) * IV.OneBody(i, j);
      }
    }
    
    IV *= gsl_sf_doublefact(2 * L + 1) / pow(q, L);
  }
  return IV;
}

/// Returns the q-dependent longitudinal electric (Coulomb) monopole transition operator in scalar format (for m0 subtraction)
Operator C_Sub(ModelSpace& modelspace, double q)
{
  Operator SpOp = Operator(modelspace, 0, 0, 0, 2);

  SpOp.OneBody.zeros();

  if (q == 0.)
    SpOp = imsrg_util::RSquaredOp(modelspace, "proton"); // NOT the limit for q -> 0, the constant term (1) has been neglected
  else
  {
    for (int i : modelspace.proton_orbits)
    {
      Orbit &oi = modelspace.GetOrbit(i);

      for (int j : SpOp.OneBodyChannels.at({oi.l, oi.j2, oi.tz2}))
      {
        if (j < i)
          continue;

        Orbit &oj = modelspace.GetOrbit(j);

        double r2int = RadialIntegral_Bessel(oi.n, oi.l, oj.n, oj.l, 0, q, modelspace);

        SpOp.OneBody(i, j) = r2int;
        SpOp.OneBody(j, i) = r2int;
      }
    }
  }
  SpOp /= sqrt(4 * PI);

  return SpOp;
}

/// Returns the q-dependent isoscalar monopole transition operator in scalar format (for m0 subtraction)
Operator IS_Sub(ModelSpace& modelspace, double q)
{
  Operator SpOp = Operator(modelspace, 0, 0, 0, 2);

  SpOp.OneBody.zeros();

  if (q == 0.)
    SpOp = imsrg_util::RSquaredOp(modelspace, "isoscalar"); // NOT the limit for q -> 0, the constant term (1) has been neglected
  else
  {
    for (int i : modelspace.all_orbits)
    {
      Orbit &oi = modelspace.GetOrbit(i);

      for (int j : SpOp.OneBodyChannels.at({oi.l, oi.j2, oi.tz2}))
      {
        if (j < i)
          continue;

        Orbit &oj = modelspace.GetOrbit(j);

        double r2int = RadialIntegral_Bessel(oi.n, oi.l, oj.n, oj.l, 0, q, modelspace);
        // double r2int = RadialIntegral_Bessel(oi.n, oi.l, oj.n, oj.l, 2, q, modelspace);

        // if (i == j)
        //   r2int -= 1.;

        SpOp.OneBody(i, j) = r2int;
        SpOp.OneBody(j, i) = r2int;
      }
    }
    // SpOp *= -6. / pow(q, 2);
    // SpOp *= 15. / pow(q, 2);
  }
  SpOp /= sqrt(4 * PI);

  return SpOp;
}

/// Returns the q-dependent transverse electric multipole transition operator. In the q->0 limit this is strictly equivalent to the LWA definition
Operator TE_Op(ModelSpace &modelspace, int L, double q)
{
  if (L == 0)
  {
    std::cout << "Trying to build a transverse electric multipole operator with L = 0 : exit" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  Operator EL(modelspace, L, 0, L % 2, 2);

  if (q == 0.)
    EL = imsrg_util::ElectricMultipoleOp(modelspace, L, 0); // Exact limit for q -> 0
  else
  {
    for (int i : modelspace.all_orbits) // Magnetic contibutions from protons must be also taken into account
    {
        Orbit &oi = modelspace.GetOrbit(i);
        double ji = 0.5 * oi.j2;

        for (int j : EL.OneBodyChannels.at({oi.l, oi.j2, oi.tz2}))
        {
            Orbit &oj = modelspace.GetOrbit(j);
            double jj = 0.5 * oj.j2;

            double r2int = RadialIntegral_TE(oi.n, oi.l, ji, oj.n, oj.l, oi.tz2, L, q, modelspace);

            EL.OneBody(i, j) = (1 + modelspace.phase(oi.l + oj.l + L)) / 2. * modelspace.phase(jj + L - 0.5) * sqrt((2 * ji + 1) * (2 * jj + 1) * (2 * L + 1) / 4. / PI) * AngMom::ThreeJ(ji, jj, L, 0.5, -0.5, 0) * r2int;
        }
    }
    EL *= gsl_sf_doublefact(2 * L + 1) / (L + 1) / pow(q, L);
  }
  return EL;
}

// Radial component of the Longitudinal-Electric (Coulomb) multipole operator (in units of e)

double RadialIntegral_Bessel(int na, int la, int nb, int lb, int L, double q, ModelSpace &modelspace)
{
  // if (L == 0) L = 2;

  double hw   = modelspace.GetHbarOmega();
  double bosc = HBARC / sqrt(M_NUCLEON * hw);

  long double I = 0.;

  bool Simpson  = false;
  bool Laguerre = false;

  if (Simpson)  // Using Simpson quadrature
  {
      size_t npoints = 301;  // Should be a multiple of 3 plus 1 for correct interval division

      std::vector<double> RGRID(npoints);
      std::vector<double> BESSEL(npoints);
      std::vector<double> INTEGRAND(npoints);

      double dr = 8.0 / double(npoints - 1);

      for (size_t i = 0; i < npoints; i++)
      {
          RGRID[i] = i * dr;
          BESSEL[i] = gsl_sf_bessel_jl(L, q * bosc * RGRID[i]);
          INTEGRAND[i] = HO_gr(nb, lb, RGRID[i]) * RGRID[i] * RGRID[i] * BESSEL[i] * HO_gr(na, la, RGRID[i]);
      }

      // Apply Simpson's 3/8 rule
      I = INTEGRAND[0] + INTEGRAND[npoints - 1];

      for (size_t i = 1; i < npoints - 1; i++)
      {
          if (i % 3 == 0)
              I += 2 * INTEGRAND[i];  // Coefficient 2 for multiples of 3
          else
              I += 3 * INTEGRAND[i];  // Coefficient 3 for others
      }

      I *= 3.0 / 8.0 * dr;  // Final multiplication factor
  }
  else if (Laguerre) // Using Gauss-Laguerre quadrature
  {
    double Norm = sqrt(tgamma(na + 1) * tgamma(nb + 1) / tgamma(na + la + 1.5) / tgamma(nb + lb + 1.5));

    int npoints = 200; // current options for npoints are 0-50, 100, and 200.

    for (int i = 0; i < npoints; i++)
    {
      //double x_i = GaussLaguerre::gauss_laguerre_points[npoints][i][0]; // From 0 to 50 points
      //double w_i = GaussLaguerre::gauss_laguerre_points[npoints][i][1]; // From 0 to 50 points
      double x_i = GaussLaguerre::gauss_laguerre_points_200[i][0];
      double w_i = GaussLaguerre::gauss_laguerre_points_200[i][1];
      double f_i = Norm * gsl_sf_laguerre_n(na, la + 0.5, x_i) * gsl_sf_laguerre_n(nb, lb + 0.5, x_i) * pow(x_i, 0.5 * (la + lb + 1)) * gsl_sf_bessel_jl(L, pow(x_i, 0.5) * q * bosc);

      I += w_i * f_i;
    }
  }
  else // Using exact definition with confluent hypergeometric functions
  {
    double y = pow(0.5 * bosc * q, 2);

    I = jL_ho(na, la, nb, lb, L, y);
  }

  return I;
}

double RadialIntegral_y_Bessel(int na, int la, int nb, int lb, int L, double q, ModelSpace &modelspace)
{
  double hw   = modelspace.GetHbarOmega();
  double bosc = HBARC / sqrt(M_NUCLEON * hw);

  long double I = 0.;

  bool Simpson  = false;

  if (Simpson)  // Using Simpson quadrature
  {
      size_t npoints = 301;  // Should be a multiple of 3 plus 1 for correct interval division

      std::vector<double> RGRID(npoints);
      std::vector<double> BESSEL(npoints);
      std::vector<double> INTEGRAND(npoints);

      double dr = 8.0 / double(npoints - 1);

      for (size_t i = 0; i < npoints; i++)
      {
          RGRID[i] = i * dr;
          BESSEL[i] = gsl_sf_bessel_yl(L, q * bosc * RGRID[i]);
          INTEGRAND[i] = HO_gr(nb, lb, RGRID[i]) * RGRID[i] * RGRID[i] * BESSEL[i] * HO_gr(na, la, RGRID[i]);
      }

      // Apply Simpson's 3/8 rule
      I = INTEGRAND[0] + INTEGRAND[npoints - 1];

      for (size_t i = 1; i < npoints - 1; i++)
      {
          if (i % 3 == 0)
              I += 2 * INTEGRAND[i];  // Coefficient 2 for multiples of 3
          else
              I += 3 * INTEGRAND[i];  // Coefficient 3 for others
      }

      I *= 3.0 / 8.0 * dr;  // Final multiplication factor
  }
  else // Using Gauss-Laguerre quadrature
  {
    double Norm = sqrt(tgamma(na + 1) * tgamma(nb + 1) / tgamma(na + la + 1.5) / tgamma(nb + lb + 1.5));

    int npoints = 200; // current options for npoints are 0-50, 100, and 200.

    for (int i = 0; i < npoints; i++)
    {
      //double x_i = GaussLaguerre::gauss_laguerre_points[npoints][i][0]; // From 0 to 50 points
      //double w_i = GaussLaguerre::gauss_laguerre_points[npoints][i][1]; // From 0 to 50 points
      double x_i = GaussLaguerre::gauss_laguerre_points_200[i][0];
      double w_i = GaussLaguerre::gauss_laguerre_points_200[i][1];
      double f_i = Norm * gsl_sf_laguerre_n(na, la + 0.5, x_i) * gsl_sf_laguerre_n(nb, lb + 0.5, x_i) * pow(x_i, 0.5 * (la + lb + 1)) * gsl_sf_bessel_yl(L, pow(x_i, 0.5) * q * bosc);

      I += w_i * f_i;
    }
  }

  return I;
}

// Radial component of the Transverse-Electric multipole operator (in units of e)

double RadialIntegral_TE(int na, int la, double ja, int nb, int lb, int tz2, int L, double q, ModelSpace &modelspace)
{
    double gs = tz2 < 0 ? PROTON_SPIN_G : NEUTRON_SPIN_G;

    double LS = 0.5 * (ja * (ja + 1.) - double(la) * (double(la) + 1.) - 0.75);

    double mu = 0.5 * HBARC / M_NUCLEON;

    double hw   = modelspace.GetHbarOmega();
    double bosc = HBARC / sqrt(M_NUCLEON * hw);

    long double I = 0.;

    bool Simpson = false; // use for very small q

    if (Simpson)  // Using Simpson quadrature
    {
        size_t npoints = 301;  // Should be a multiple of 3 plus 1 for correct interval division

        std::vector<double> RGRID(npoints);
        std::vector<double> FUNCTION(npoints);
        std::vector<double> INTEGRAND(npoints);

        double dr = 8.0 / double(npoints - 1);

        for (size_t i = 0; i < npoints; i++)
        {
            RGRID[i] = i * dr;

            // Here is the setting of the function to integrate
            FUNCTION[i] = mu * q * HBARC * gs * LS * gsl_sf_bessel_jl(L, q * bosc * RGRID[i]); // both for proton and neutrons anyway

            if(tz2 < 0) // only protons
                FUNCTION[i] += (1 + 2. * mu * q) * (double(L + 1) * gsl_sf_bessel_jl(L, q * bosc * RGRID[i]) - q * bosc *RGRID[i] * gsl_sf_bessel_jl(L + 1, q * bosc * RGRID[i]));

            INTEGRAND[i] = HO_gr(nb, lb, RGRID[i]) * RGRID[i] * RGRID[i] * FUNCTION[i] * HO_gr(na, la, RGRID[i]);
        }

        // Apply Simpson's 3/8 rule
        I = INTEGRAND[0] + INTEGRAND[npoints - 1];

        for (size_t i = 1; i < npoints - 1; i++)
        {
            if (i % 3 == 0)
                I += 2 * INTEGRAND[i];  // Coefficient 2 for multiples of 3
            else
                I += 3 * INTEGRAND[i];  // Coefficient 3 for others
        }

        I *= 3.0 / 8.0 * dr;  // Final multiplication factor
    }
    else // Using Gauss-Laguerre quadrature
    {
        double Norm = sqrt(tgamma(na + 1) * tgamma(nb + 1) / tgamma(na + la + 1.5) / tgamma(nb + lb + 1.5));

        int npoints = 200; // current options for npoints are 0-50, 100, and 200.

        for (int i = 0; i < npoints; i++)
        {
            //double x_i = GaussLaguerre::gauss_laguerre_points[npoints][i][0]; // From 0 to 50 points
            //double w_i = GaussLaguerre::gauss_laguerre_points[npoints][i][1]; // From 0 to 50 points
            double x_i = GaussLaguerre::gauss_laguerre_points_200[i][0];
            double w_i = GaussLaguerre::gauss_laguerre_points_200[i][1];

            double R = pow(x_i, 0.5);

            double funct = mu * q * HBARC * gs * LS * gsl_sf_bessel_jl(L, q * bosc * R);

            if(tz2 < 0) // only protons
                funct += (1 + 2. * mu * q) * (double(L + 1) * gsl_sf_bessel_jl(L, q * bosc * R) - q * bosc * R * gsl_sf_bessel_jl(L + 1, q * bosc * R));

            double f_i = Norm * gsl_sf_laguerre_n(na, la + 0.5, x_i) * gsl_sf_laguerre_n(nb, lb + 0.5, x_i) * pow(x_i, 0.5 * (la + lb + 1)) * funct;

            I += w_i * f_i;
        }
    } 
    return I;
}

// The radial harmonic-oscillator wavefunction
double HO_gr(int n, int l, double x)
{
  double Norm = sqrt(2 * tgamma(n + 1) / tgamma(n + l + 1.5));

  return Norm * pow(x, l) * exp(-0.5 * x * x) * gsl_sf_laguerre_n(n, l + 0.5, x * x);
}

// The spherical Bessel function matrix element
double jL_ho(int na0, int la, int nb0, int lb, int L, double y)
{
  int na = na0 + 1; // Different conventions on n are used wrt e.g. Suhonen
  int nb = nb0 + 1; // 

  double f1 = pow(2.0, L) / gsl_sf_doublefact(2 * L + 1) * pow(y, 0.5 * L) * exp(-y) * sqrt(gsl_sf_fact(nb - 1) * gsl_sf_fact(na - 1) * gsl_sf_gamma(na + la + 0.5) * gsl_sf_gamma(nb + lb + 0.5));

  double j_ho = 0.0;

  for (int ka = 0; ka < na; ka++)
  {
    double f2 = pow(-1, ka) / gsl_sf_fact(ka) / gsl_sf_fact(na - 1 - ka) / gsl_sf_gamma(la + ka + 1.5);
    double f3 = 0.0;

    for (int kb = 0; kb < nb; kb++)
    {
      f3 += pow(-1, kb) / gsl_sf_fact(kb) / gsl_sf_fact(nb - 1 - kb) / gsl_sf_gamma(lb + kb + 1.5) * gsl_sf_gamma(0.5 * (L + lb + la + 2 * ka + 2 * kb + 3.0)) * gsl_sf_hyperg_1F1(0.5 * (L - lb - la - 2 * ka - 2 * kb), L + 1.5, y);
    }
    j_ho += f2 * f3;
  }
  j_ho *= f1;

  return j_ho;
}

// Kernels operator
Operator Mix0(ModelSpace& modelspace, const Operator& LL, const Operator& RR)
{
  // Initialize operator
  Operator LdotR_op(modelspace, 0, 0, 0, 2);

  auto& Lmat = LL.OneBody;
  auto& Rmat = RR.OneBody;

  int Ll = LL.GetJRank();
  int Lr = RR.GetJRank();

  if (Ll != Lr) 
  {
    std::cout << "Trying to mix different multipole channels: exit" << std::endl;
    exit(EXIT_FAILURE);
  }

  int L = Ll;

  // Filling the one-body part
  for (int i : modelspace.all_orbits)
  {
    Orbit& oi = modelspace.GetOrbit(i);
    double ji = 0.5 * oi.j2;

    for (int j : LdotR_op.OneBodyChannels.at({oi.l, oi.j2, oi.tz2})) // Channels accessible to the scalar operator
    {
      Orbit& oj = modelspace.GetOrbit(j);
      double jj = 0.5 * oj.j2;

      double me = 0.;

      for (int k : LL.OneBodyChannels.at({oi.l, oi.j2, oi.tz2})) // Channels accessible to the tensor operator
      {
        Orbit& ok = modelspace.GetOrbit(k);
        double jk = 0.5 * ok.j2;

        me += 0.5 * (Lmat(k, i) * Rmat(k, j) + Rmat(k, i) * Lmat(k, j));
      }
      LdotR_op.OneBody(i,j) = me / (2 * ji + 1);
    }
  }

  // Filling the two-body part
  int nchan = modelspace.GetNumberTwoBodyChannels();

  for (int ch = 0; ch < nchan; ++ch)
  {
    TwoBodyChannel& tbc = modelspace.GetTwoBodyChannel(ch);

    int nkets = tbc.GetNumberKets();
    int J     = tbc.J;

    for (int ibra = 0; ibra < nkets; ++ibra)
    {
      Ket & bra = tbc.GetKet(ibra);

      int i = bra.p;
      int j = bra.q;

      Orbit & oi = modelspace.GetOrbit(i);
      Orbit & oj = modelspace.GetOrbit(j);

      double ji = oi.j2 * 0.5;
      double jj = oj.j2 * 0.5;

      for (int iket = ibra; iket < nkets; ++iket)
      {
        Ket & ket = tbc.GetKet(iket);

        int k = ket.p;
        int l = ket.q;

        Orbit & ok = modelspace.GetOrbit(k);
        Orbit & ol = modelspace.GetOrbit(l);

        double jk = ok.j2 * 0.5;
        double jl = ol.j2 * 0.5;

        if (oi.tz2 + oj.tz2 != ok.tz2 + ol.tz2) continue; // Only charge-conserving transitions are allowed up to now

        double Aijkl = 0.5 * modelspace.GetSixJ(ji, jl, L, jk, jj, J) * (Lmat(k, j) * Rmat(i, l) + Rmat(k, j) * Lmat(i, l));
        double Ajikl = 0.5 * modelspace.GetSixJ(jj, jl, L, jk, ji, J) * (Lmat(k, i) * Rmat(j, l) + Rmat(k, i) * Lmat(j, l));

        double val = 0.;

        if (tbc.Tz == 0) // proton-neutron channel
        {
               if (oj.tz2 == ok.tz2) val = Aijkl;                                     // pnpn
          else if (oi.tz2 == ok.tz2) val = modelspace.phase(ji + jj + J + 1) * Ajikl; // pnnp

          val *= -2.;
        }
        else // pppp or nnnn
        {
          val = Aijkl - modelspace.phase(ji + jj + J) * Ajikl;

          if (i == j) val /= sqrt(2.0);
          if (k == l) val /= sqrt(2.0);

          val *= -2.;
        }
        LdotR_op.TwoBody.SetTBME(ch, ibra, iket, val);
      }
    }
  }
  return LdotR_op;
}

Operator Mix1(ModelSpace& modelspace, const Operator& H, const Operator& LL, const Operator& RR)
{
  // Initialize operator
  Operator SR(modelspace, 0, 0, 0, 2);

  auto& Rmat = RR.OneBody;
  auto& Lmat = LL.OneBody;
  auto& h    = H.OneBody;
  auto& V    = H.TwoBody;

  int Ll = LL.GetJRank();
  int Lr = RR.GetJRank();

  if (Ll != Lr) 
  {
    std::cout << "Trying to mix different multipole channels: exit" << std::endl;
    exit(EXIT_FAILURE);
  }

  int K = Ll;

  // cut on matrix elements in loops
  double prec = 1e-8;

  // Filling the one-body part

  SR.OneBody -= h * Lmat * Rmat.t() + h * Rmat * Lmat.t();
  SR.OneBody += Lmat * h * Rmat.t() + Rmat * h * Lmat.t();
  SR.OneBody += Lmat.t() * h * Rmat + Rmat.t() * h * Lmat;
  SR.OneBody -= Lmat.t() * Rmat * h + Rmat.t() * Lmat * h;

  // reduce operator
  for (int a : modelspace.all_orbits) {
    Orbit& oa = modelspace.GetOrbit(a);
    double ja = 0.5 * oa.j2;

    for (int b : SR.OneBodyChannels.at({oa.l, oa.j2, oa.tz2})) {
      Orbit& ob = modelspace.GetOrbit(b);
      double jb = 0.5 * ob.j2;

      SR.OneBody(a, b) *= 0.25 / (2 * ja + 1);
    }
  }

  // Filling the two-body part
  int nchan = modelspace.GetNumberTwoBodyChannels();

  #pragma omp parallel for schedule(dynamic,1) 
  for (int ch = 0; ch < nchan; ++ch)
  {
    TwoBodyChannel& tbc = modelspace.GetTwoBodyChannel(ch);

    int nkets = tbc.GetNumberKets();
    int J     = tbc.J;

    for (int ibra = 0; ibra < nkets; ++ibra)
    {
      Ket & bra = tbc.GetKet(ibra);

      int a = bra.p;
      int b = bra.q;

      Orbit & oa = modelspace.GetOrbit(a);
      Orbit & ob = modelspace.GetOrbit(b);

      double ja = oa.j2 * 0.5;
      double jb = ob.j2 * 0.5;

      for (int iket = ibra; iket < nkets; ++iket)
      {
        Ket & ket = tbc.GetKet(iket);

        int c = ket.p;
        int d = ket.q;

        Orbit & oc = modelspace.GetOrbit(c);
        Orbit & od = modelspace.GetOrbit(d);

        double jc = oc.j2 * 0.5;
        double jd = od.j2 * 0.5;

        ///////////////////////////// W1 /////////////////////////////

        double W1abcd = 0.;
        double W1abdc = 0.;

        for (int e : LL.OneBodyChannels.at({oc.l, oc.j2, oc.tz2}))
        {
          Orbit& oe = modelspace.GetOrbit(e);
          double je = 0.5 * oe.j2;

          double meL1 = Lmat(e, c);
          double meR1 = Rmat(e, c);
          if(abs(meL1) < prec && abs(meR1) < prec) continue;

          for (int f : LL.OneBodyChannels.at({od.l, od.j2, od.tz2}))
          {
            Orbit& of = modelspace.GetOrbit(f);
            double jf = 0.5 * of.j2;

            double meL2 = Lmat(f, d);
            double meR2 = Rmat(f, d);
            if(abs(meL2) < prec && abs(meR2) < prec) continue;

            double abcd = 0.5 * V.GetTBME_J(J, a, b, e, f) * (meL1 * meR2 + meR1 * meL2);
            if(abs(abcd) < prec) continue;

            int Jmin = std::max(std::abs(od.j2 - oe.j2) / 2, std::abs(J - K));
            int Jmax = std::min((od.j2 + oe.j2) / 2, J + K);

            for (int JJ = Jmin; JJ <= std::min(Jmax, modelspace.TwoBodyJmax); ++JJ)
              if (!(d == e && JJ % 2 != 0))
                W1abcd += (2 * JJ + 1) * modelspace.phase(J + JJ) * modelspace.GetSixJ(J, K, JJ, jd, je, jf) * modelspace.GetSixJ(J, K, JJ, je, jd, jc) * abcd;
          }
        }

        for (int e : LL.OneBodyChannels.at({od.l, od.j2, od.tz2}))
        {
          Orbit& oe = modelspace.GetOrbit(e);
          double je = 0.5 * oe.j2;

          double meL1 = Lmat(e, d);
          double meR1 = Rmat(e, d);
          if(abs(meL1) < prec && abs(meR1) < prec) continue;

          for (int f : LL.OneBodyChannels.at({oc.l, oc.j2, oc.tz2}))
          {
            Orbit& of = modelspace.GetOrbit(f);
            double jf = 0.5 * of.j2;

            double meL2 = Lmat(f, c);
            double meR2 = Rmat(f, c);
            if(abs(meL2) < prec && abs(meR2) < prec) continue;

            double abdc = 0.5 * V.GetTBME_J(J, a, b, e, f) * (meR1 * meL2 + meL1 * meR2);
            if(abs(abdc) < prec) continue;

            int Jmin = std::max(std::abs(oc.j2 - oe.j2) / 2, std::abs(J - K));
            int Jmax = std::min((oc.j2 + oe.j2) / 2, J + K);

            for (int JJ = Jmin; JJ <= std::min(Jmax, modelspace.TwoBodyJmax); ++JJ)
              if (!(c == e && JJ % 2 != 0))
                W1abdc += (2 * JJ + 1) * modelspace.phase(J + JJ) * modelspace.GetSixJ(J, K, JJ, jc, je, jf) * modelspace.GetSixJ(J, K, JJ, je, jc, jd) * abdc;
          }
        }

        double W1 = W1abcd - modelspace.phase(jc + jd + J) * W1abdc;

        ///////////////////////////// W2 /////////////////////////////

        double W2abcd = 0.;
        double W2abdc = 0.;

        for (int f : LL.OneBodyChannels.at({od.l, od.j2, od.tz2}))
        {
          Orbit& of = modelspace.GetOrbit(f);
          double jf = 0.5 * of.j2;

          double meL1 = Lmat(d, f);
          double meR1 = Rmat(d, f);
          if(abs(meL1) < prec && abs(meR1) < prec) continue;

          for (int e : LL.OneBodyChannels.at({of.l, of.j2, of.tz2}))
          {
            Orbit& oe = modelspace.GetOrbit(e);
            double je = 0.5 * oe.j2;

            double meL2 = Lmat(e, f);
            double meR2 = Rmat(e, f);
            if(abs(meL2) < prec && abs(meR2) < prec) continue;

            double abcd = 0.5 * V.GetTBME_J(J, a, b, c, e) * (meL2 * meR1 + meR2 * meL1);
            if(abs(abcd) < prec) continue;

            int Jmin = std::max(std::abs(oc.j2 - of.j2) / 2, std::abs(J - K));
            int Jmax = std::min((oc.j2 + of.j2) / 2, J + K);

            for (int JJ = Jmin; JJ <= std::min(Jmax, modelspace.TwoBodyJmax); ++JJ)
              if (!(c == f && JJ % 2 != 0))
                W2abcd += (2 * JJ + 1) * modelspace.GetSixJ(J, K, JJ, jf, jc, je) * modelspace.GetSixJ(J, K, JJ, jf, jc, jd) * abcd;
          }
        }

        for (int f : LL.OneBodyChannels.at({oc.l, oc.j2, oc.tz2}))
        {
          Orbit& of = modelspace.GetOrbit(f);
          double jf = 0.5 * of.j2;

          double meL1 = Lmat(c, f);
          double meR1 = Rmat(c, f);
          if(abs(meL1) < prec && abs(meR1) < prec) continue;

          for (int e : LL.OneBodyChannels.at({of.l, of.j2, of.tz2}))
          {
            Orbit& oe = modelspace.GetOrbit(e);
            double je = 0.5 * oe.j2;

            double meL2 = Lmat(e, f);
            double meR2 = Rmat(e, f);
            if(abs(meL2) < prec && abs(meR2) < prec) continue;

            double abdc = 0.5 * V.GetTBME_J(J, a, b, d, e) * (meR2 * meL1 + meL2 * meR1);
            if(abs(abdc) < prec) continue;

            int Jmin = std::max(std::abs(od.j2 - of.j2) / 2, std::abs(J - K));
            int Jmax = std::min((od.j2 + of.j2) / 2, J + K);

            for (int JJ = Jmin; JJ <= std::min(Jmax, modelspace.TwoBodyJmax); ++JJ)
              if (!(d == f && JJ % 2 != 0))
                W2abdc += (2 * JJ + 1) * modelspace.GetSixJ(J, K, JJ, jf, jd, je) * modelspace.GetSixJ(J, K, JJ, jf, jd, jc) * abdc;
          }
        }

        double W2 = W2abcd - modelspace.phase(jc + jd + J) * W2abdc;

        ///////////////////////////// W3 /////////////////////////////

        double W3abcd = 0.;
        double W3abdc = 0.;
        double W3bacd = 0.;
        double W3badc = 0.;

        for (int f : LL.OneBodyChannels.at({oc.l, oc.j2, oc.tz2}))
        {
          Orbit& of = modelspace.GetOrbit(f);
          double jf = 0.5 * of.j2;

          double meL1 = Lmat(f, c);
          double meR1 = Rmat(f, c);
          if(abs(meL1) < prec && abs(meR1) < prec) continue;

          for (int e : LL.OneBodyChannels.at({oa.l, oa.j2, oa.tz2}))
          {
            Orbit& oe = modelspace.GetOrbit(e);
            double je = 0.5 * oe.j2;

            double meL2 = Lmat(e, a);
            double meR2 = Rmat(e, a);
            if(abs(meL2) < prec && abs(meR2) < prec) continue;

            int Jmin = std::max(std::abs(ob.j2 - oe.j2), std::abs(od.j2 - of.j2)) / 2;
            int Jmax = std::min((ob.j2 + oe.j2), (od.j2 + of.j2)) / 2;

            for (int JJ = Jmin; JJ <= std::min(Jmax, modelspace.TwoBodyJmax); ++JJ)
              W3abcd += (2 * JJ + 1) * modelspace.GetSixJ(J, K, JJ, je, jb, ja) * modelspace.GetSixJ(J, K, JJ, jf, jd, jc) * V.GetTBME_J(JJ, b, e, d, f) * (meL2 * meR1 + meR2 * meL1);
          }
        }

        for (int f : LL.OneBodyChannels.at({od.l, od.j2, od.tz2}))
        {
          Orbit& of = modelspace.GetOrbit(f);
          double jf = 0.5 * of.j2;

          double meL1 = Lmat(f, d);
          double meR1 = Rmat(f, d);
          if(abs(meL1) < prec && abs(meR1) < prec) continue;

          for (int e : LL.OneBodyChannels.at({oa.l, oa.j2, oa.tz2}))
          {
            Orbit& oe = modelspace.GetOrbit(e);
            double je = 0.5 * oe.j2;

            double meL2 = Lmat(e, a);
            double meR2 = Rmat(e, a);
            if(abs(meL2) < prec && abs(meR2) < prec) continue;

            int Jmin = std::max(std::abs(ob.j2 - oe.j2), std::abs(oc.j2 - of.j2)) / 2;
            int Jmax = std::min((ob.j2 + oe.j2), (oc.j2 + of.j2)) / 2;

            for (int JJ = Jmin; JJ <= std::min(Jmax, modelspace.TwoBodyJmax); ++JJ)
              W3abdc += (2 * JJ + 1) * modelspace.GetSixJ(J, K, JJ, je, jb, ja) * modelspace.GetSixJ(J, K, JJ, jf, jc, jd) * V.GetTBME_J(JJ, b, e, c, f) * (meL2 * meR1 + meR2 * meL1);
          }
        }

        for (int f : LL.OneBodyChannels.at({oc.l, oc.j2, oc.tz2}))
        {
          Orbit& of = modelspace.GetOrbit(f);
          double jf = 0.5 * of.j2;

          double meL1 = Lmat(f, c);
          double meR1 = Rmat(f, c);
          if(abs(meL1) < prec && abs(meR1) < prec) continue;

          for (int e : LL.OneBodyChannels.at({ob.l, ob.j2, ob.tz2}))
          {
            Orbit& oe = modelspace.GetOrbit(e);
            double je = 0.5 * oe.j2;

            double meL2 = Lmat(e, b);
            double meR2 = Rmat(e, b);
            if(abs(meL2) < prec && abs(meR2) < prec) continue;

            int Jmin = std::max(std::abs(oa.j2 - oe.j2), std::abs(od.j2 - of.j2)) / 2;
            int Jmax = std::min((oa.j2 + oe.j2), (od.j2 + of.j2)) / 2;

            for (int JJ = Jmin; JJ <= std::min(Jmax, modelspace.TwoBodyJmax); ++JJ)
              W3bacd += (2 * JJ + 1) * modelspace.GetSixJ(J, K, JJ, je, ja, jb) * modelspace.GetSixJ(J, K, JJ, jf, jd, jc) * V.GetTBME_J(JJ, a, e, d, f) * (meL2 * meR1 + meR2 * meL1);
          }
        }

        for (int f : LL.OneBodyChannels.at({od.l, od.j2, od.tz2}))
        {
          Orbit& of = modelspace.GetOrbit(f);
          double jf = 0.5 * of.j2;

          double meL1 = Lmat(f, d);
          double meR1 = Rmat(f, d);
          if(abs(meL1) < prec && abs(meR1) < prec) continue;

          for (int e : LL.OneBodyChannels.at({ob.l, ob.j2, ob.tz2}))
          {
            Orbit& oe = modelspace.GetOrbit(e);
            double je = 0.5 * oe.j2;

            double meL2 = Lmat(e, b);
            double meR2 = Rmat(e, b);
            if(abs(meL2) < prec && abs(meR2) < prec) continue;

            int Jmin = std::max(std::abs(oa.j2 - oe.j2), std::abs(oc.j2 - of.j2)) / 2;
            int Jmax = std::min((oa.j2 + oe.j2), (oc.j2 + of.j2)) / 2;

            for (int JJ = Jmin; JJ <= std::min(Jmax, modelspace.TwoBodyJmax); ++JJ)
              W3badc += (2 * JJ + 1) * modelspace.GetSixJ(J, K, JJ, je, ja, jb) * modelspace.GetSixJ(J, K, JJ, jf, jc, jd) * V.GetTBME_J(JJ, a, e, c, f) * (meL2 * meR1 + meR2 * meL1);
          }
        }

        double W3 = W3abcd - modelspace.phase(jc + jd + J) * W3abdc - modelspace.phase(ja + jb + J) * W3bacd + modelspace.phase(ja + jb + J) * modelspace.phase(jc + jd + J) * W3badc;

        ///////////////////////////// W4 /////////////////////////////

        double W4abcd = 0.;
        double W4abdc = 0.;

        for (int e : LL.OneBodyChannels.at({oa.l, oa.j2, oa.tz2}))
        {
          Orbit& oe = modelspace.GetOrbit(e);
          double je = 0.5 * oe.j2;

          double meL1 = Lmat(e, a);
          double meR1 = Rmat(e, a);
          if(abs(meL1) < prec && abs(meR1) < prec) continue;

          for (int f : LL.OneBodyChannels.at({ob.l, ob.j2, ob.tz2}))
          {
            Orbit& of = modelspace.GetOrbit(f);
            double jf = 0.5 * of.j2;

            double meL2 = Lmat(f, b);
            double meR2 = Rmat(f, b);
            if(abs(meL2) < prec && abs(meR2) < prec) continue;

            double abcd = 0.5 * V.GetTBME_J(J, c, d, e, f) * (meL1 * meR2 + meR1 * meL2);
            if(abs(abcd) < prec) continue;

            int Jmin = std::max(std::abs(ob.j2 - oe.j2) / 2, std::abs(J - K));
            int Jmax = std::min((ob.j2 + oe.j2) / 2, J + K);

            for (int JJ = Jmin; JJ <= std::min(Jmax, modelspace.TwoBodyJmax); ++JJ)
              if (!(b == e && JJ % 2 != 0))
                W4abcd += (2 * JJ + 1) * modelspace.phase(J + JJ) * modelspace.GetSixJ(J, K, JJ, jb, je, jf) * modelspace.GetSixJ(J, K, JJ, je, jb, ja) * abcd;
          }
        }

        for (int e : LL.OneBodyChannels.at({ob.l, ob.j2, ob.tz2}))
        {
          Orbit& oe = modelspace.GetOrbit(e);
          double je = 0.5 * oe.j2;

          double meL1 = Lmat(e, b);
          double meR1 = Rmat(e, b);
          if(abs(meL1) < prec && abs(meR1) < prec) continue;

          for (int f : LL.OneBodyChannels.at({oa.l, oa.j2, oa.tz2}))
          {
            Orbit& of = modelspace.GetOrbit(f);
            double jf = 0.5 * of.j2;

            double meL2 = Lmat(f, a);
            double meR2 = Rmat(f, a);
            if(abs(meL2) < prec && abs(meR2) < prec) continue;

            double abdc = 0.5 * V.GetTBME_J(J, c, d, e, f) * (meR1 * meL2 + meL1 * meR2);
            if(abs(abdc) < prec) continue;

            int Jmin = std::max(std::abs(oa.j2 - oe.j2) / 2, std::abs(J - K));
            int Jmax = std::min((oa.j2 + oe.j2) / 2, J + K);

            for (int JJ = Jmin; JJ <= std::min(Jmax, modelspace.TwoBodyJmax); ++JJ)
              if (!(a == e && JJ % 2 != 0))
                W4abdc += (2 * JJ + 1) * modelspace.phase(J + JJ) * modelspace.GetSixJ(J, K, JJ, ja, je, jf) * modelspace.GetSixJ(J, K, JJ, je, ja, jb) * abdc;
          }
        }

        double W4 = W4abcd - modelspace.phase(ja + jb + J) * W4abdc;

        ///////////////////////////// W5 /////////////////////////////

        double W5abcd = 0.;
        double W5abdc = 0.;

        for (int f : LL.OneBodyChannels.at({ob.l, ob.j2, ob.tz2}))
        {
          Orbit& of = modelspace.GetOrbit(f);
          double jf = 0.5 * of.j2;

          double meL1 = Lmat(b, f);
          double meR1 = Rmat(b, f);
          if(abs(meL1) < prec && abs(meR1) < prec) continue;

          for (int e : LL.OneBodyChannels.at({of.l, of.j2, of.tz2}))
          {
            Orbit& oe = modelspace.GetOrbit(e);
            double je = 0.5 * oe.j2;

            double meL2 = Lmat(e, f);
            double meR2 = Rmat(e, f);
            if(abs(meL2) < prec && abs(meR2) < prec) continue;

            double abcd = 0.5 * V.GetTBME_J(J, c, d, a, e) * (meL2 * meR1 + meR2 * meL1);
            if(abs(abcd) < prec) continue;

            int Jmin = std::max(std::abs(oa.j2 - of.j2) / 2, std::abs(J - K));
            int Jmax = std::min((oa.j2 + of.j2) / 2, J + K);

            for (int JJ = Jmin; JJ <= std::min(Jmax, modelspace.TwoBodyJmax); ++JJ)
              if (!(a == f && JJ % 2 != 0))
                W5abcd += (2 * JJ + 1) * modelspace.GetSixJ(J, K, JJ, jf, ja, je) * modelspace.GetSixJ(J, K, JJ, jf, ja, jb) * abcd;
          }
        }

        for (int f : LL.OneBodyChannels.at({oa.l, oa.j2, oa.tz2}))
        {
          Orbit& of = modelspace.GetOrbit(f);
          double jf = 0.5 * of.j2;

          double meL1 = Lmat(a, f);
          double meR1 = Rmat(a, f);
          if(abs(meL1) < prec && abs(meR1) < prec) continue;

          for (int e : LL.OneBodyChannels.at({of.l, of.j2, of.tz2}))
          {
            Orbit& oe = modelspace.GetOrbit(e);
            double je = 0.5 * oe.j2;

            double meL2 = Lmat(e, f);
            double meR2 = Rmat(e, f);
            if(abs(meL2) < prec && abs(meR2) < prec) continue;

            double abdc = 0.5 * V.GetTBME_J(J, c, d, b, e) * (meR2 * meL1 + meL2 * meR1);
            if(abs(abdc) < prec) continue;

            int Jmin = std::max(std::abs(ob.j2 - of.j2) / 2, std::abs(J - K));
            int Jmax = std::min((ob.j2 + of.j2) / 2, J + K);

            for (int JJ = Jmin; JJ <= std::min(Jmax, modelspace.TwoBodyJmax); ++JJ)
              if (!(b == f && JJ % 2 != 0))
                W5abdc += (2 * JJ + 1) * modelspace.GetSixJ(J, K, JJ, jf, jb, je) * modelspace.GetSixJ(J, K, JJ, jf, jb, ja) * abdc;
          }
        }

        double W5 = W5abcd - modelspace.phase(ja + jb + J) * W5abdc;

        ///////////////////////////// TOTAL ///////////////////////////

        double W = 0.5 * (W3 - (W1 + W2 + W4 + W5));

        if (a == b) W /= sqrt(2.0);
        if (c == d) W /= sqrt(2.0);

        SR.TwoBody.SetTBME(ch, ibra, iket, W);
      }
    }
  }
  return SR;
}

Operator SetOperator(ModelSpace& modelspace, double q, int L, std::string field)
{
  Operator Op = Operator(modelspace, 0, 0, 0, 2);

  if (field == "TE")
    Op = TE_Op(modelspace, L, q);
  else if (field == "C")
    Op = C_Op(modelspace, L, q);
  else if (field == "IS")
    Op = IS_Op(modelspace, L, q);
  else if (field == "PS")
  {
    Operator tmp = IS_Op(modelspace, L, q);
    Operator T   = imsrg_util::KineticEnergy_Op(modelspace);
    
    Op = tmp + Commutator::Commutator(T, tmp);
  }
  else if (field == "IV")
    Op = IV_Op(modelspace, L, q);

  return Op;
}

Operator SetSub(ModelSpace& modelspace, double q, std::string field)
{
  Operator Op = Operator(modelspace, 0, 0, 0, 2);

  if (field == "C")
    Op = C_Sub(modelspace, q);
  else if (field == "IS")
    Op = IS_Sub(modelspace, q);
  else if (field == "PS")
  {
    Operator tmp = IS_Sub(modelspace, q);
    Operator T   = imsrg_util::KineticEnergy_Op(modelspace);
    
    Op = tmp + Commutator::Commutator(T, tmp);
  }

  return Op;
}

void computeKernel(double qL, double qR, Operator H, int L, std::string fL, std::string fR, imsrg_response::KernelParams par)
{
  ModelSpace modelspace = *par.modelspace;
  HFMBPT     hf         = *par.hf;

  ReadWrite rw;

  // Initialising left and right operators

  Operator OpL = SetOperator(modelspace, qL, L, fL);
  Operator OpR = SetOperator(modelspace, qR, L, fR);

  double sub_L0 = 0.;
  double sub_R0 = 0.;
  double sub_Ls = 0.;
  double sub_Rs = 0.;

  Operator OpSubL = Operator(modelspace, 0, 0, 0, 2);
  Operator OpSubR = Operator(modelspace, 0, 0, 0, 2);

  if (L == 0) // In the monopole case one needs to set the operator to subtract
  {
    OpSubL = SetSub(modelspace, qL, fL);
    OpSubR = SetSub(modelspace, qR, fR);
  }

  Operator MixMom0 = Mix0(modelspace, OpL, OpR);
  Operator MixMom1 = Mix1(modelspace, H, OpL, OpR);

  MixMom0 = hf.TransformToHFBasis(MixMom0).DoNormalOrdering();
  MixMom1 = hf.TransformToHFBasis(MixMom1).DoNormalOrdering();

  if (L == 0)
  {
    OpSubL = hf.TransformToHFBasis(OpSubL).DoNormalOrdering();
    OpSubR = hf.TransformToHFBasis(OpSubR).DoNormalOrdering();
  }

  double mom0_0 = MixMom0.ZeroBody;
  double mom1_0 = MixMom1.ZeroBody;

  if (L == 0)
  {
    sub_L0 = OpSubL.ZeroBody;
    sub_R0 = OpSubR.ZeroBody;

    mom0_0 -= sub_L0 * sub_R0;
  }

  for (int i = 0; i < par.N_Magnus; i++)
  {
    int eMax = modelspace.Emax;

    Operator Magnus_i = Operator(modelspace, 0, 0, 0, 2);

    Magnus_i.SetAntiHermitian();

    rw.Read_me1j (par.omefile + "_" + std::to_string(i) + ".me1j.gz",  Magnus_i, eMax, eMax);
    rw.Read_me2jp(par.omefile + "_" + std::to_string(i) + ".me2jp.gz", Magnus_i, eMax, 2 * eMax, eMax);

    MixMom0 = Commutator::BCH_Transform(MixMom0, Magnus_i);
    MixMom1 = Commutator::BCH_Transform(MixMom1, Magnus_i);

    if (L == 0)
    {
      OpSubL = Commutator::BCH_Transform(OpSubL, Magnus_i);
      OpSubR = Commutator::BCH_Transform(OpSubR, Magnus_i);
    }
  }

  double mom0_s = MixMom0.ZeroBody;
  double mom1_s = MixMom1.ZeroBody;

  if (L == 0)
  {
    sub_Ls = OpSubL.ZeroBody;
    sub_Rs = OpSubR.ZeroBody;

    mom0_s -= sub_Ls * sub_Rs;
  }

  auto ss = boost::format{"%s/L=%i_%s_%.3f_%s_%.3f.dat"} % par.kerdir % L % fL % qL % fR % qR;
  std::string filename = ss.str();

  printKernel(std::cout, fL, qL, fR, qR, mom0_0, mom1_0, mom0_s, mom1_s);

  // Print to file
  std::ofstream file(filename);
  if (file)
  {
    printKernel(file, fL, qL, fR, qR, mom0_0, mom1_0, mom0_s, mom1_s);
    file.close();
    std::cout << "Data written to " << filename << std::endl;
  }
  else
  {
    std::cout << "Error opening file " << filename << std::endl;
  }
}

// Printing functions

void printKernel(std::ostream& out, std::string fL, double qL, std::string fR, double qR, double mom0_0, double mom1_0, double mom0_s, double mom1_s)
{
  out << std::setw(7)  << "lda L"
      << std::setw(10) << "qL"
      << std::setw(7)  << "lda R"
      << std::setw(10) << "qR"
      << std::setw(16) << "mom0_HF"
      << std::setw(16) << "mom1_HF"
      << std::setw(16) << "mom0_imsrg"
      << std::setw(16) << "mom1_imsrg" << std::endl;

  out << std::fixed << std::setprecision(3);
  out << std::setw(7)  << fL
      << std::setw(10) << qL
      << std::setw(7)  << fR
      << std::setw(10) << qR;

  out << std::fixed << std::setprecision(8);
  out << std::fixed << std::scientific;  // Use scientific notation
  out << std::setw(16) << mom0_0
      << std::setw(16) << mom1_0
      << std::setw(16) << mom0_s
      << std::setw(16) << mom1_s << std::endl;
}

}