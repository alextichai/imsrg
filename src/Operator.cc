
#include "Operator.hh"
#include "AngMom.hh"
#include <cmath>
#include <iostream>
#include <iomanip>
#ifndef SQRT2
  #define SQRT2 1.4142135623730950488
#endif

using namespace std;

//===================================================================================
//===================================================================================
//  START IMPLEMENTATION OF OPERATOR METHODS
//===================================================================================
//===================================================================================

//double  Operator::bch_transform_threshold = 1e-6;
double  Operator::bch_transform_threshold = 1e-9;
double  Operator::bch_product_threshold = 1e-4;
map<string, double> Operator::timer;


Operator::~Operator()
{
//   cout << "calling Operator destructor" << endl;
}

/////////////////// CONSTRUCTORS /////////////////////////////////////////
Operator::Operator()
 :   modelspace(NULL), 
    rank_J(0), rank_T(0), parity(0), particle_rank(2),
    hermitian(true), antihermitian(false), nChannels(0)
{
}


// Create a zero-valued operator in a given model space
Operator::Operator(ModelSpace& ms, int Jrank, int Trank, int p, int part_rank) : 
    modelspace(&ms), ZeroBody(0), OneBody(ms.GetNumberOrbits(), ms.GetNumberOrbits(),arma::fill::zeros),
    TwoBody(&ms,Jrank,Trank,p),  ThreeBody(&ms),
    rank_J(Jrank), rank_T(Trank), parity(p), particle_rank(part_rank),
    E3max(ms.GetN3max()),
    hermitian(true), antihermitian(false),  
    nChannels(ms.GetNumberTwoBodyChannels()) 
{
  SetUpOneBodyChannels();
  if (particle_rank >=3) ThreeBody.Allocate();
}



Operator::Operator(ModelSpace& ms) :
    modelspace(&ms), ZeroBody(0), OneBody(ms.GetNumberOrbits(), ms.GetNumberOrbits(),arma::fill::zeros),
    TwoBody(&ms),  ThreeBody(&ms),
    rank_J(0), rank_T(0), parity(0), particle_rank(2),
    E3max(ms.GetN3max()),
    hermitian(true), antihermitian(false),  
    nChannels(ms.GetNumberTwoBodyChannels())
{
  SetUpOneBodyChannels();
}

Operator::Operator(const Operator& op)
: modelspace(op.modelspace),  ZeroBody(op.ZeroBody),
  OneBody(op.OneBody), TwoBody(op.TwoBody) ,ThreeBody(op.ThreeBody),
  rank_J(op.rank_J), rank_T(op.rank_T), particle_rank(op.particle_rank),
  E2max(op.E2max), E3max(op.E3max), 
  hermitian(op.hermitian), antihermitian(op.antihermitian),
  nChannels(op.nChannels), OneBodyChannels(op.OneBodyChannels)
{
//   cout << "Calling copy constructor for Operator" << endl;
}

Operator::Operator(Operator&& op)
: modelspace(op.modelspace), ZeroBody(op.ZeroBody),
  OneBody(move(op.OneBody)), TwoBody(move(op.TwoBody)) , ThreeBody(move(op.ThreeBody)),
  rank_J(op.rank_J), rank_T(op.rank_T), particle_rank(op.particle_rank),
  E2max(op.E2max), E3max(op.E3max), 
  hermitian(op.hermitian), antihermitian(op.antihermitian),
  nChannels(op.nChannels), OneBodyChannels(op.OneBodyChannels)
{
//   cout << "Calling move constructor for Operator" << endl;
}

/////////// COPY METHOD //////////////////////////
void Operator::Copy(const Operator& op)
{
   modelspace    = op.modelspace;
   nChannels     = op.nChannels;
   hermitian     = op.hermitian;
   antihermitian = op.antihermitian;
   rank_J        = op.rank_J;
   rank_T        = op.rank_T;
   parity        = op.parity;
   particle_rank = op.particle_rank;
   E2max         = op.E2max;
   E3max         = op.E3max;
   ZeroBody      = op.ZeroBody;
   OneBody       = op.OneBody;
   TwoBody       = op.TwoBody;
   ThreeBody     = op.ThreeBody;
   OneBodyChannels = op.OneBodyChannels;
}

/////////////// OVERLOADED OPERATORS =,+,-,*,etc ////////////////////
Operator& Operator::operator=(const Operator& rhs)
{
//   cout << "Using copy assignment" << endl;
   Copy(rhs);
   return *this;
}

Operator& Operator::operator=(Operator&& rhs)
{
//   cout << "Using move assignment" << endl;
   modelspace    = rhs.modelspace;
   nChannels     = rhs.nChannels;
   hermitian     = rhs.hermitian;
   antihermitian = rhs.antihermitian;
   rank_J        = rhs.rank_J;
   rank_T        = rhs.rank_T;
   parity        = rhs.parity;
   particle_rank = rhs.particle_rank;
   E2max         = rhs.E2max;
   E3max         = rhs.E3max;
   ZeroBody      = rhs.ZeroBody;
   OneBody       = move(rhs.OneBody);
   TwoBody       = move(rhs.TwoBody);
   ThreeBody     = move(rhs.ThreeBody);
   OneBodyChannels = move(rhs.OneBodyChannels);
   return *this;
}

// multiply operator by a scalar
Operator& Operator::operator*=(const double rhs)
{
   ZeroBody *= rhs;
   OneBody *= rhs;
   TwoBody *= rhs;
   return *this;
}

Operator Operator::operator*(const double rhs) const
{
   Operator opout = Operator(*this);
   opout *= rhs;
   return opout;
}

// Add non-member operator so we can multiply an operator
// by a scalar from the lhs, i.e. s*O = O*s
Operator operator*(const double lhs, const Operator& rhs)
{
   return rhs * lhs;
}
Operator operator*(const double lhs, const Operator&& rhs)
{
   return rhs * lhs;
}


// divide operator by a scalar
Operator& Operator::operator/=(const double rhs)
{
   return *this *=(1.0/rhs);
}

Operator Operator::operator/(const double rhs) const
{
   Operator opout = Operator(*this);
   opout *= (1.0/rhs);
   return opout;
}

// Add operators
Operator& Operator::operator+=(const Operator& rhs)
{
   ZeroBody += rhs.ZeroBody;
   OneBody  += rhs.OneBody;
   TwoBody  += rhs.TwoBody;
   return *this;
}

Operator Operator::operator+(const Operator& rhs) const
{
   return ( Operator(*this) += rhs );
}

// Subtract operators
Operator& Operator::operator-=(const Operator& rhs)
{
   ZeroBody -= rhs.ZeroBody;
   OneBody -= rhs.OneBody;
   TwoBody -= rhs.TwoBody;
   return *this;
}

Operator Operator::operator-(const Operator& rhs) const
{
   return ( Operator(*this) -= rhs );
}

Operator Operator::operator-() const
{
   return (*this)*-1.0;
}



void Operator::PrintTimes()
{
   cout << "==== TIMES ====" << endl;
   for ( auto it : timer )
   {
     cout << it.first << ":  " << it.second  << endl;
   }
}


void Operator::SetUpOneBodyChannels()
{
  for ( int i=0; i<modelspace->GetNumberOrbits(); ++i )
  {
    Orbit& oi = modelspace->GetOrbit(i);
    int lmin = max( oi.l - rank_J - parity, (oi.l+parity)%2);
    int lmax = min( oi.l + rank_J, modelspace->Nmax);
    for (int l=lmin; l<=lmax; l+=2)
    {
      int j2min = max(max(oi.j2 - 2*rank_J, 2*l-1),1);
      int j2max = min(oi.j2 + 2*rank_J, 2*l+1);
      for (int j2=j2min; j2<=j2max; j2+=2)
      {
        int tz2min = max( oi.tz2 - 2*rank_T, -1);
        int tz2max = min( oi.tz2 + 2*rank_T, 1);
        for (int tz2=tz2min; tz2<=tz2max; tz2+=2)
        {
          OneBodyChannels[ {l, j2, tz2} ].push_back(i);
        }
      }
    }
  }
}


////////////////// MAIN INTERFACE METHODS //////////////////////////

Operator Operator::DoNormalOrdering()
{
   if (particle_rank==3)
   {
      return DoNormalOrdering3();
   }
   else
   {
      return DoNormalOrdering2();
   }
}

//*************************************************************
///  Normal ordering of a 2body operator
///  currently this only handles scalar operators
//*************************************************************
Operator Operator::DoNormalOrdering2()
{
   Operator opNO = *this;


   for (auto& k : modelspace->holes) // loop over hole orbits
   {
      opNO.ZeroBody += (modelspace->GetOrbit(k).j2+1) * OneBody(k,k);
   }


   index_t norbits = modelspace->GetNumberOrbits();

   for ( auto& itmat : TwoBody.MatEl )
   {
      int ch_bra = itmat.first[0];
      int ch_ket = itmat.first[1];
      auto& matrix = itmat.second;
      
      TwoBodyChannel &tbc_ket = modelspace->GetTwoBodyChannel(ch_ket);
      int J_ket = tbc_ket.J;

      // Zero body part
      arma::vec diagonals = matrix.diag();
      auto hh = tbc_ket.GetKetIndex_hh();
      opNO.ZeroBody += arma::sum( diagonals.elem(hh) ) * (2*J_ket+1);

      // One body part
      for (index_t a=0;a<norbits;++a)
      {
         Orbit &oa = modelspace->GetOrbit(a);
         index_t bstart = IsNonHermitian() ? 0 : a; // If it's neither hermitian or anti, we need to do the full sum
         for ( auto& b : modelspace->OneBodyChannels.at({oa.l,oa.j2,oa.tz2}) ) // OneBodyChannels should be moved to the operator, to accommodate tensors
         {
            if (b < bstart) continue;
            for (auto& h : modelspace->holes)  // C++11 syntax
            {
               opNO.OneBody(a,b) += (2*J_ket+1.0)/(oa.j2+1)  * TwoBody.GetTBME(ch_bra,ch_ket,a,h,b,h);
            }
         }
      }
   } // loop over channels

   if (hermitian) opNO.Symmetrize();
   if (antihermitian) opNO.AntiSymmetrize();

   return opNO;
}



//*******************************************************************************
///   Normal ordering of a three body operator. Start by generating the normal ordered
///   two body piece, then use DoNormalOrdering2() to get the rest. (Note that there
///   are some numerical factors).
///   The normal ordered two body piece is 
///   \f[ \Gamma^J_{ijkl} = V^J_{ijkl} + \sum_a n_a  \sum_K \frac{2K+1}{2J+1} V^{(3)JJK}_{ijakla} \f]
///   Right now, this is only set up for scalar operators, but I don't anticipate
///   handling 3body tensor operators in the near future.
//*******************************************************************************
Operator Operator::DoNormalOrdering3()
{
   Operator opNO3 = Operator(*modelspace);
//   #pragma omp parallel for
   for ( auto& itmat : opNO3.TwoBody.MatEl )
   {
      int ch = itmat.first[0]; // assume ch_bra = ch_ket for 3body...
      TwoBodyChannel& tbc = modelspace->GetTwoBodyChannel(ch);
      arma::mat& Gamma = (arma::mat&) itmat.second;
      for (int ibra=0; ibra<tbc.GetNumberKets(); ++ibra)
      {
         Ket & bra = tbc.GetKet(ibra);
         int i = bra.p;
         int j = bra.q;
         Orbit & oi = modelspace->GetOrbit(i);
         Orbit & oj = modelspace->GetOrbit(j);
         for (int iket=ibra; iket<tbc.GetNumberKets(); ++iket)
         {
            Ket & ket = tbc.GetKet(iket);
            int k = ket.p;
            int l = ket.q;
            Orbit & ok = modelspace->GetOrbit(k);
            Orbit & ol = modelspace->GetOrbit(l);
            for (auto& a : modelspace->holes)
            {
               Orbit & oa = modelspace->GetOrbit(a);
               if ( (2*(oi.n+oj.n+oa.n)+oi.l+oj.l+oa.l)>E3max) continue;
               if ( (2*(ok.n+ol.n+oa.n)+ok.l+ol.l+oa.l)>E3max) continue;
               int kmin2 = abs(2*tbc.J-oa.j2);
               int kmax2 = 2*tbc.J+oa.j2;
               for (int K2=kmin2; K2<=kmax2; K2+=2)
               {
                  Gamma(ibra,iket) += (K2+1) * ThreeBody.GetME_pn(tbc.J,tbc.J,K2,i,j,a,k,l,a);
               }
            }
            Gamma(ibra,iket) /= (2*tbc.J+1);
         }
      }
   }
   opNO3.Symmetrize();
   Operator opNO2 = opNO3.DoNormalOrdering2();
   opNO2.ScaleZeroBody(1./3.);
   opNO2.ScaleOneBody(1./2.);

   // Also normal order the 1 and 2 body pieces
   opNO2 += DoNormalOrdering2();
   return opNO2;

}


Operator Operator::UndoNormalOrdering()
{
   Operator opNO = *this;
   cout << "Undoing Normal ordering. Initial ZeroBody = " << opNO.ZeroBody << endl;

   for (auto& k : modelspace->holes) // loop over hole orbits
   {
      opNO.ZeroBody -= (modelspace->GetOrbit(k).j2+1) * OneBody(k,k);
   }

   index_t norbits = modelspace->GetNumberOrbits();

   for ( auto& itmat : TwoBody.MatEl )
   {
      int ch_bra = itmat.first[0];
      int ch_ket = itmat.first[1];
      auto& matrix = itmat.second;
      
      TwoBodyChannel &tbc_ket = modelspace->GetTwoBodyChannel(ch_ket);
      int J_ket = tbc_ket.J;

      // Zero body part
      arma::vec diagonals = matrix.diag();
      auto hh = tbc_ket.GetKetIndex_hh();
      opNO.ZeroBody += arma::sum( diagonals.elem(hh) ) * (2*J_ket+1);

      // One body part
      for (index_t a=0;a<norbits;++a)
      {
         Orbit &oa = modelspace->GetOrbit(a);
         index_t bstart = IsNonHermitian() ? 0 : a; // If it's neither hermitian or anti, we need to do the full sum
         for ( auto& b : modelspace->OneBodyChannels.at({oa.l,oa.j2,oa.tz2}) ) // OneBodyChannels should be moved to the operator, to accommodate tensors
         {
            if (b < bstart) continue;
            for (auto& h : modelspace->holes)  // C++11 syntax
            {
               opNO.OneBody(a,b) -= (2*J_ket+1.0)/(oa.j2+1)  * TwoBody.GetTBME(ch_bra,ch_ket,a,h,b,h);
            }
         }
      }
   } // loop over channels

   if (hermitian) opNO.Symmetrize();
   if (antihermitian) opNO.AntiSymmetrize();

   cout << "Zero-body piece is now " << opNO.ZeroBody << endl;
   return opNO;

}

/*
Operator Operator::DoNormalOrderingCore()
{
   Operator opNO = *this;
   cout << "Normal Ordering wrt the core. Initial ZeroBody = " << opNO.ZeroBody << endl;


   for (auto& k : modelspace->hole_qspace) // loop over core orbits
   {
      opNO.ZeroBody += (modelspace->GetOrbit(k).j2+1) * OneBody(k,k);
   }


   index_t norbits = modelspace->GetNumberOrbits();

   for ( auto& itmat : TwoBody.MatEl )
   {
      int ch_bra = itmat.first[0];
      int ch_ket = itmat.first[1];
      auto& matrix = itmat.second;
      
//      TwoBodyChannel &tbc_bra = modelspace->GetTwoBodyChannel(ch_bra);
      TwoBodyChannel &tbc_ket = modelspace->GetTwoBodyChannel(ch_ket);
      int J_ket = tbc_ket.J;

      // Zero body part
      arma::vec diagonals = matrix.diag();
//      auto hh = tbc_ket.GetKetIndex_hh();
      auto cc = tbc_ket.GetKetIndex_c_c();
      opNO.ZeroBody += arma::sum( diagonals.elem(cc) ) * (2*J_ket+1);

      // One body part
//      for (long long unsigned int a=0;a<norbits;++a)
      for (index_t a=0;a<norbits;++a)
      {
         Orbit &oa = modelspace->GetOrbit(a);
         index_t bstart = IsNonHermitian() ? 0 : a; // If it's neither hermitian or anti, we need to do the full sum
         for ( auto& b : modelspace->OneBodyChannels.at({oa.l,oa.j2,oa.tz2}) ) // OneBodyChannels should be moved to the operator, to accommodate tensors
         {
            if (b < bstart) continue;
            for (auto& h : modelspace->hole_qspace)  // C++11 syntax
            {
               opNO.OneBody(a,b) += (2*J_ket+1.0)/(oa.j2+1)  * TwoBody.GetTBME(ch_bra,ch_ket,a,h,b,h);
            }
         }
      }
   } // loop over channels

   if (hermitian) opNO.Symmetrize();
   if (antihermitian) opNO.AntiSymmetrize();

   cout << "Zero body piece is now " << opNO.ZeroBody << endl;

   return opNO;
}
*/

ModelSpace* Operator::GetModelSpace()
{
   return modelspace;
}


void Operator::Erase()
{
  EraseZeroBody();
  EraseOneBody();
  TwoBody.Erase();
  if (particle_rank >=3)
    ThreeBody.Erase();
}

void Operator::EraseOneBody()
{
   OneBody.zeros();
}

void Operator::EraseTwoBody()
{
 TwoBody.Erase();
}


void Operator::ScaleZeroBody(double x)
{
   ZeroBody *= x;
}

void Operator::ScaleOneBody(double x)
{
   OneBody *= x;
}

void Operator::ScaleTwoBody(double x)
{
   TwoBody.Scale(x);
}

void Operator::Eye()
{
   ZeroBody = 1;
   OneBody.eye();
   TwoBody.Eye();
}


//***********************************************
/// Calculates the kinetic energy operator in the 
/// harmonic oscillator basis.
/// \f[ t_{ab} = \frac{1}{2}\hbar\omega
/// \delta_{\ell_a \ell_b} \delta_{j_aj_b} \delta_{t_{za}t_{zb}}
/// \left\{
/// \begin{array}{ll}
/// 2n_a + \ell_a + \frac{3}{2} &: n_a=n_b\\
/// \sqrt{n_{a}(n_{a}+\ell_a + \frac{1}{2})} &: n_a=n_b+1\\
/// \end{array} \right. \f]
//***********************************************
void Operator::CalculateKineticEnergy()
{
   OneBody.zeros();
   int norbits = modelspace->GetNumberOrbits();
   double hw = modelspace->GetHbarOmega();
   for (int a=0;a<norbits;++a)
   {
      Orbit & oa = modelspace->GetOrbit(a);
      OneBody(a,a) = 0.5 * hw * (2*oa.n + oa.l +3./2); 
      for (int b=a+1;b<norbits;++b)  // make this better once OneBodyChannel is implemented
      {
         Orbit & ob = modelspace->GetOrbit(b);
         if (oa.l == ob.l and oa.j2 == ob.j2 and oa.tz2 == ob.tz2)
         {
            if (oa.n == ob.n+1)
               OneBody(a,b) = 0.5 * hw * sqrt( (oa.n)*(oa.n + oa.l +1./2));
            else if (oa.n == ob.n-1)
               OneBody(a,b) = 0.5 * hw * sqrt( (ob.n)*(ob.n + ob.l +1./2));
            OneBody(b,a) = OneBody(a,b);
         }
      }
   }
}










//*****************************************************************************************
/// X.BCH_Transform(Y) returns \f$ Z = e^{Y} X e^{-Y} \f$.
/// We use the [Baker-Campbell-Hausdorff formula](http://en.wikipedia.org/wiki/Baker-Campbell-Hausdorff_formula)
/// \f[ Z = X + [Y,X] + \frac{1}{2!}[Y,[Y,X]] + \frac{1}{3!}[Y,[Y,[Y,X]]] + \ldots \f]
/// with all commutators truncated at the two-body level.
Operator Operator::BCH_Transform(  Operator &Omega)
{
   double t = omp_get_wtime();
   int max_iter = 40;
   int warn_iter = 12;
   double nx = Norm();
   double ny = Omega.Norm();
   Operator OpOut = *this;
   Operator OpNested = *this;
   double epsilon = nx * exp(-2*ny) * bch_transform_threshold / (2*ny);
   for (int i=1; i<max_iter; ++i)
   {
      OpNested = Omega.Commutator(OpNested);
      OpNested /= i;

      OpOut += OpNested;

      if (OpNested.Norm() < epsilon *(i+1))
      {
        timer["BCH_Transform"] += omp_get_wtime() - t;
        return OpOut;
      }
      if (i == warn_iter)
      {
         cout << "Warning: BCH_Transform not converged after " << warn_iter << " nested commutators" << endl;
      }

   }
   cout << "Warning: BCH_Transform didn't coverge after "<< max_iter << " nested commutators" << endl;
   timer["BCH_Transform"] += omp_get_wtime() - t;
   return OpOut;
}


//*****************************************************************************************
// Baker-Campbell-Hausdorff formula
//  returns Z, where
//  exp(Z) = exp(X) * exp(Y).
//  Z = X + Y + 1/2[X, Y]
//     + 1/12 [X,[X,Y]] + 1/12 [Y,[Y,X]]
//     - 1/24 [Y,[X,[X,Y]]]
//     - 1/720 [Y,[Y,[Y,[Y,X]]]] - 1/720 [X,[X,[X,[X,Y]]]]
//     + ...

//*****************************************************************************************
/// X.BCH_Product(Y) returns \f$Z\f$ such that \f$ e^{Z} = e^{X}e^{Y}\f$
/// by employing the [Baker-Campbell-Hausdorff formula](http://en.wikipedia.org/wiki/Baker-Campbell-Hausdorff_formula)
/// \f[ Z = X + Y + \frac{1}{2}[X,Y] + \frac{1}{12}([X,[X,Y]]+[Y,[Y,X]]) + \ldots \f]
//*****************************************************************************************
Operator Operator::BCH_Product(  Operator &Y)
{
   double t = omp_get_wtime();
   Operator& X = *this;
   double nx = X.Norm();
   double ny = Y.Norm();
   if (nx < 1e-7) return Y;
   if (ny < 1e-7) return X;

   Operator Z = X.Commutator(Y);
   Z *= 0.5;
   double nxy = Z.Norm();

   if ( nxy < (nx+ny)*bch_product_threshold )
   {
     Z += X;
     Z += Y;
     timer["BCH_Product"] += omp_get_wtime() - t;
     return Z;
   }

   Y -= X;
   Z += (1./6)* Z.Commutator( Y );
   Z += Y;
   X *=2;
   Z += X;

   timer["BCH_Product"] += omp_get_wtime() - t;
   return Z;
}

/// Obtain the Frobenius norm of the operator, which here is 
/// defined as 
/// \f[ \|X\| = \sqrt{\|X_{(1)}\|^2 +\|X_{(2)}\|^2 } \f]
/// and
/// \f[ \|X_{(1)}\|^2 = \sum\limits_{ij} X_{ij}^2 \f]
double Operator::Norm() const
{
   double n1 = OneBodyNorm();
   double n2 = TwoBody.Norm();
   return sqrt(n1*n1+n2*n2);
}

double Operator::OneBodyNorm() const
{
   return arma::norm(OneBody,"fro");
}



double Operator::TwoBodyNorm() const
{
  return TwoBody.Norm();
}

void Operator::Symmetrize()
{
   OneBody = arma::symmatu(OneBody);
   TwoBody.Symmetrize();
}

void Operator::AntiSymmetrize()
{
   int norb = modelspace->GetNumberOrbits();
   for (int i=0;i<norb;++i)
   {
      for(int j=i+1;j<norb;++j)
      {
        OneBody(j,i) = -OneBody(i,j);
      }
   }
   TwoBody.AntiSymmetrize();
}

Operator Operator::Commutator( Operator& opright)
{
   timer["N_Commutators"] += 1;
   if (rank_J==0)
   {
      if (opright.rank_J==0)
      {
         return CommutatorScalarScalar(opright); // [S,S]
      }
      else
      {
         return CommutatorScalarTensor(opright); // [S,T]
      }
   }
   else if(opright.rank_J==0)
   {
      return (-1)*opright.CommutatorScalarTensor(*this); // [T,S]
   }
   else
   {
      cout << "In Tensor-Tensor because rank_J = " << rank_J << "  and opright.rank_J = " << opright.rank_J << endl;
      cout << " Tensor-Tensor commutator not yet implemented." << endl;
      return *this;
   }
}


Operator Operator::CommutatorScalarScalar( Operator& opright) 
{
   Operator out = opright;
   out.EraseZeroBody();
   out.EraseOneBody();
   out.EraseTwoBody();

   if ( (IsHermitian() and opright.IsHermitian()) or (IsAntiHermitian() and opright.IsAntiHermitian()) ) out.SetAntiHermitian();
   else if ( (IsHermitian() and opright.IsAntiHermitian()) or (IsAntiHermitian() and opright.IsHermitian()) ) out.SetHermitian();
   else out.SetNonHermitian();

   if ( not out.IsAntiHermitian() )
   {
      comm110ss(opright, out);
      if (particle_rank>1 and opright.particle_rank>1)
        comm220ss(opright, out) ;
   }

    double t = omp_get_wtime();
   comm111ss(opright, out);
    timer["comm111ss"] += omp_get_wtime() - t;

    t = omp_get_wtime();
//   comm111st(opright, out);  // << equivalent in scalar case
   comm121ss(opright, out);
//   comm121st(opright, out);  // << equivalent in scalar case
    timer["comm121ss"] += omp_get_wtime() - t;

    t = omp_get_wtime();
   comm122ss(opright, out); //  This is the slow one for some reason.
    timer["comm122ss"] += omp_get_wtime() - t;

   if (particle_rank>1 and opright.particle_rank>1)
   {
    t = omp_get_wtime();
    comm222_pp_hh_221ss(opright, out);
    timer["comm222_pp_hh_221ss"] += omp_get_wtime() - t;
     
////   comm222_pp_hh_221st(opright, out); // << equivalent in scalar case

    t = omp_get_wtime();
    comm222_phss(opright, out);
//   comm222_phst_pandya(opright, out);
////   comm222_phst(opright, out);
    timer["comm222_phss"] += omp_get_wtime() - t;
   }


   if ( out.IsHermitian() )
   {
      out.Symmetrize();
   }
   else if (out.IsAntiHermitian() )
   {
      out.AntiSymmetrize();
   }


   return out;
}


// Calculate [S,T]
Operator Operator::CommutatorScalarTensor( Operator& opright) 
{
   Operator out = opright; // This ensures the commutator has the same tensor rank as opright
   out.EraseZeroBody();
   out.EraseOneBody();
   out.EraseTwoBody();

   if ( (IsHermitian() and opright.IsHermitian()) or (IsAntiHermitian() and opright.IsAntiHermitian()) ) out.SetAntiHermitian();
   else if ( (IsHermitian() and opright.IsAntiHermitian()) or (IsAntiHermitian() and opright.IsHermitian()) ) out.SetHermitian();
   else out.SetNonHermitian();

   comm111st(opright, out);
   comm121st(opright, out);

   comm122st(opright, out);
   comm222_pp_hh_221st(opright, out);
   comm222_phst(opright, out);

   return out;
}



///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
// Below is the implementation of the commutators in the various channels
///////////////////////////////////////////////////////////////////////////////////////////

//*****************************************************************************************
//                ____Y    __         ____X
//          X ___(_)             Y___(_) 
//
//  [X1,Y1](0) = Sum_ab (2j_a+1) x_ab y_ba  (n_a-n_b) 
//             = Sum_a  (2j_a+1)  (xy-yx)_aa n_a
//
// -- AGREES WITH NATHAN'S RESULTS
/// \f[
///  [X_{1)},Y_{(1)}]_{(0)} = \sum_{a} n_a (2j_a+1) \left(X_{(1)}Y_{(1)}-Y_{(1)}X_{(1)}\right)_{aa}
/// \f]
void Operator::comm110ss( Operator& opright, Operator& out) 
{
  if (IsHermitian() and opright.IsHermitian()) return ; // I think this is the case
  if (IsAntiHermitian() and opright.IsAntiHermitian()) return ; // I think this is the case

   arma::mat xyyx = OneBody*opright.OneBody - opright.OneBody*OneBody;
   for ( auto& a : modelspace->holes) 
   {
      out.ZeroBody += (modelspace->GetOrbit(a).j2+1) * xyyx(a,a);
   }
}


//*****************************************************************************************
//         __Y__       __X__
//        ()_ _()  -  ()_ _()
//           X           Y
//
//  [ X^(2), Y^(2) ]^(0) = 1/2 Sum_abcd  Sum_J (2J+1) x_abcd y_cdab (n_a n_b nbar_c nbar_d)
//                       = 1/2 Sum_J (2J+1) Sum_abcd x_abcd y_cdab (n_a n_b nbar_c nbar_d)  
//                       = 1/2 Sum_J (2J+1) Sum_ab  (X*P_pp*Y)_abab  P_hh
//
//  -- AGREES WITH NATHAN'S RESULTS (within < 1%)
/// \f[
/// [X_{(2)},Y_{(2)}]_{(0)} = \frac{1}{2} \sum_{J} (2J+1) \sum_{abcd} (n_a n_b \bar{n}_c \bar{n}_d) \tilde{X}_{abcd}^{J} \tilde{Y}_{cdab}^{J}
/// \f]
/// may be rewritten as
/// \f[
/// [X_{(2)},Y_{(2)}]_{(0)} = 2 \sum_{J} (2J+1) Tr(X_{hh'pp'}^{J} Y_{pp'hh'}^{J})
/// \f] where we obtain a factor of four from converting two unrestricted sums to restricted sums, i.e. \f$\sum_{ab} \rightarrow \sum_{a\leq b} \f$,
/// and using the normalized TBME.
void Operator::comm220ss(  Operator& opright, Operator& out) 
{
   if (IsHermitian() and opright.IsHermitian()) return; // I think this is the case
   if (IsAntiHermitian() and opright.IsAntiHermitian()) return; // I think this is the case

   for (int ch=0;ch<nChannels;++ch)
   {
      TwoBodyChannel& tbc = modelspace->GetTwoBodyChannel(ch);
      auto hh = tbc.GetKetIndex_hh();
      auto pp = tbc.GetKetIndex_pp();
      arma::mat& X = TwoBody.GetMatrix(ch);
      arma::mat& Y = opright.TwoBody.GetMatrix(ch);
      out.ZeroBody += 2 * (2*tbc.J+1) * arma::trace( X.submat(hh,pp) * Y.submat(pp,hh) );
   }
}

//*****************************************************************************************
//
//        |____. Y          |___.X
//        |        _        |
//  X .___|            Y.___|              [X1,Y1](1)  =  XY - YX
//        |                 |
//
// -- AGREES WITH NATHAN'S RESULTS
/// \f[
/// [X_{(1)},Y_{(1)}]_{(1)} = X_{(1)}Y_{(1)} - Y_{(1)}X_{(1)}
/// \f]
void Operator::comm111ss( Operator & opright, Operator& out) 
{
   out.OneBody += OneBody*opright.OneBody - opright.OneBody*OneBody;
}

//*****************************************************************************************
//                                       |
//      i |              i |             |
//        |    ___.Y       |__X__        |
//        |___(_)    _     |   (_)__.    |  [X2,Y1](1)  =  1/(2j_i+1) sum_ab(n_a-n_b)y_ab 
//      j | X            j |        Y    |        * sum_J (2J+1) x_biaj^(J)  
//                                       |      
//---------------------------------------*        = 1/(2j+1) sum_a n_a sum_J (2J+1)
//                                                  * sum_b y_ab x_biaj - yba x_aibj
//
//                     (note: I think this should actually be)
//                                                = sum_ab (n_a nbar_b) sum_J (2J+1)/(2j_i+1)
//                                                      * y_ab xbiag - yba x_aibj
//
// -- AGREES WITH NATHAN'S RESULTS 
/// Returns \f$ [X_{(1)},Y_{(2)}] - [Y_{(1)},X_{(2)}] \f$, where
/// \f[
/// [X_{(1)},Y_{(2)}]_{ij} = \frac{1}{2j_i+1}\sum_{ab} (n_a \bar{n}_b) \sum_{J} (2J+1) (X_{ab} Y^J_{biaj} - X_{ba} Y^J_{aibj})
/// \f]
void Operator::comm121ss( Operator& opright, Operator& out) 
{
   int norbits = modelspace->GetNumberOrbits();
   #pragma omp parallel for 
   for (int i=0;i<norbits;++i)
   {
      Orbit &oi = modelspace->GetOrbit(i);
      int jmin = out.IsNonHermitian() ? 0 : i;
      for (int j : modelspace->OneBodyChannels.at({oi.l,oi.j2,oi.tz2}) ) 
      {
          if (j<jmin) continue; // only calculate upper triangle
          for (auto& a : modelspace->holes)  // C++11 syntax
          {
             Orbit &oa = modelspace->GetOrbit(a);
             for (auto& b : modelspace->particles)
             {
                Orbit &ob = modelspace->GetOrbit(b);
                out.OneBody(i,j) += (ob.j2+1) *  OneBody(a,b) * opright.TwoBody.GetTBMEmonopole(b,i,a,j) ;
                out.OneBody(i,j) -= (oa.j2+1) *  OneBody(b,a) * opright.TwoBody.GetTBMEmonopole(a,i,b,j) ;

                // comm211 part
                out.OneBody(i,j) -= (ob.j2+1) *  opright.OneBody(a,b) * TwoBody.GetTBMEmonopole(b,i,a,j) ;
                out.OneBody(i,j) += (oa.j2+1) *  opright.OneBody(b,a) * TwoBody.GetTBMEmonopole(a,i,b,j) ;
             }
          }
      }
   }
}



//*****************************************************************************************
//
//      i |              i |            [X2,Y2](1)  =  1/(2(2j_i+1)) sum_J (2J+1) 
//        |__Y__           |__X__           * sum_abc (nbar_a*nbar_b*n_c + n_a*n_b*nbar_c)
//        |    /\          |    /\          * (x_ciab y_abcj - y_ciab xabcj)
//        |   (  )   _     |   (  )                                                                                      
//        |____\/          |____\/       = 1/(2(2j+1)) sum_J (2J+1)
//      j | X            j |  Y            *  sum_c ( Pp*X*Phh*Y*Pp - Pp*Y*Phh*X*Pp)  - (Ph*X*Ppp*Y*Ph - Ph*Y*Ppp*X*Ph)_cicj
//                                     
//
// -- AGREES WITH NATHAN'S RESULTS 
//   No factor of 1/2 because the matrix multiplication corresponds to a restricted sum (a<=b) 
// \f[
// [X_{(2)},Y_{(2)}]_{ij} = \frac{1}{2(2j_i+1)}\sum_{J}(2J+1)\sum_{c}
// \left( \mathcal{P}_{pp} (X \mathcal{P}_{hh} Y^{J} 
// - Y^{J} \mathcal{P}_{hh} X^{J}) \mathcal{P}_{pp}
//  - \mathcal{P}_{hh} (X^{J} \mathcal{P}_{pp} Y^{J} 
//  -  Y^{J} \mathcal{P}_{pp} X^{J}) \mathcal{P}_{hh} \right)_{cicj}
// \f]
/// \f[
/// [X_{(2)},Y_{(2)}]_{ij} = \frac{1}{2(2j_i+1)}\sum_{J}(2J+1)\sum_{abc} (\bar{n}_a\bar{n}_bn_c + n_an_b\bar{n}_c)
///  (X^{J}_{ciab} Y^{J}_{abcj} - Y^{J}_{ciab}X^{J}_{abcj})
/// \f]
/// This may be rewritten as
/// \f[
/// [X_{(2)},Y_{(2)}]_{ij} = \frac{1}{2j_i+1} \sum_{c} \sum_{J} (2J+1) \left( n_c \mathcal{M}^{J}_{pp,icjc} + \bar{n}_c\mathcal{M}^{J}_{hh,icjc} \right)
/// \f]
/// With the intermediate matrix \f[ \mathcal{M}^{J}_{pp} \equiv \frac{1}{2} (X^{J}\mathcal{P}_{pp} Y^{J} - Y^{J}\mathcal{P}_{pp}X^{J}) \f]
/// and likewise for \f$ \mathcal{M}^{J}_{hh} \f$
void Operator::comm221ss( Operator& opright, Operator& out) 
{

   int norbits = modelspace->GetNumberOrbits();
   TwoBodyME Mpp = opright.TwoBody;
   TwoBodyME Mhh = opright.TwoBody;

   #pragma omp parallel for schedule(dynamic,1)
   for (int ch=0;ch<nChannels;++ch)
   {
      TwoBodyChannel& tbc = modelspace->GetTwoBodyChannel(ch);

      auto& LHS = (arma::mat&) TwoBody.GetMatrix(ch,ch);
      auto& RHS = (arma::mat&) opright.TwoBody.GetMatrix(ch,ch);
      arma::mat& Matrixpp =  Mpp.GetMatrix(ch,ch);
      arma::mat& Matrixhh =  Mpp.GetMatrix(ch,ch);
      
      Matrixpp = ( LHS.rows(tbc.GetKetIndex_pp()) * RHS.cols(tbc.GetKetIndex_pp()));
      Matrixpp -= Matrixpp.t();
      Matrixhh = ( LHS.rows(tbc.GetKetIndex_hh()) * RHS.cols(tbc.GetKetIndex_hh()));
      Matrixhh -= Matrixhh.t();

      // If commutator is hermitian or antihermitian, we only
      // need to do half the sum. Add this.
      for (int i=0;i<norbits;++i)
      {
         Orbit &oi = modelspace->GetOrbit(i);
         for (int j : modelspace->OneBodyChannels.at({oi.l,oi.j2,oi.tz2}) )
         {
            double cijJ = 0;
            // Sum c over holes and include the nbar_a * nbar_b terms
            for (auto& c : modelspace->holes)
            {
               cijJ +=   Mpp.GetTBME(ch,i,c,j,c);
            // Sum c over particles and include the n_a * n_b terms
            }
            for (auto& c : modelspace->particles)
            {
               cijJ +=  Mhh.GetTBME(ch,i,c,j,c);
            }
            cijJ *= (2*tbc.J+1.0)/(oi.j2 +1.0);
            #pragma omp critical
            out.OneBody(i,j) +=  cijJ;
         } // for j
      } // for i
   } //for ch
}





//*****************************************************************************************
//
//    |     |               |      |           [X2,Y1](2) = sum_a ( Y_ia X_ajkl + Y_ja X_iakl - Y_ak X_ijal - Y_al X_ijka )
//    |     |___.Y          |__X___|         
//    |     |         _     |      |          
//    |_____|               |      |_____.Y        
//    |  X  |               |      |            
//
// -- AGREES WITH NATHAN'S RESULTS
/// Returns \f$ [X_{(1)},Y_{(2)}]_{(2)} - [Y_{(1)},X_{(2)}]_{(2)} \f$, where
/// \f[
/// [X_{(1)},Y_{(2)}]^{J}_{ijkl} = \sum_{a} ( X_{ia}Y^{J}_{ajkl} + X_{ja}Y^{J}_{iakl} - X_{ak} Y^{J}_{ijal} - X_{al} Y^{J}_{ijka} )
/// \f]
/// here, all TBME are unnormalized, i.e. they should have a tilde.
void Operator::comm122ss( Operator& opright, Operator& opout ) 
{
   auto& L1 = OneBody;
   auto& R1 = opright.OneBody;

//   for (int ch=0; ch<nChannels; ++ch)
   int n_nonzero = modelspace->SortedTwoBodyChannels.size();
   #pragma omp parallel for schedule(dynamic,1)
   for (int ich=0; ich<n_nonzero; ++ich)
   {
      int ch = modelspace->SortedTwoBodyChannels[ich];
      TwoBodyChannel& tbc = modelspace->GetTwoBodyChannel(ch);
      auto& L2 = TwoBody.GetMatrix(ch,ch);
      auto& R2 = opright.TwoBody.GetMatrix(ch,ch);
      arma::mat& OUT = opout.TwoBody.GetMatrix(ch,ch);


      int npq = tbc.GetNumberKets();
//      int norbits = modelspace->GetNumberOrbits();
      for (int indx_ij = 0;indx_ij<npq; ++indx_ij)
      {
         Ket & bra = tbc.GetKet(indx_ij);
         int i = bra.p;
         int j = bra.q;
         double pre_ij = i==j ? SQRT2 : 1;
         Orbit& oi = modelspace->GetOrbit(i);
         Orbit& oj = modelspace->GetOrbit(j);
         arma::Row<double> L2_ij = L2.row(indx_ij); // trying this to better use the cache. not sure if it helps.
         arma::Row<double> R2_ij = R2.row(indx_ij);
         int klmin = opout.IsNonHermitian() ? 0 : indx_ij;
         for (int indx_kl=klmin;indx_kl<npq; ++indx_kl)
         {
            Ket & ket = tbc.GetKet(indx_kl);
            int k = ket.p;
            int l = ket.q;
            double pre_kl = k==l ? SQRT2 : 1;
            Orbit& ok = modelspace->GetOrbit(k);
            Orbit& ol = modelspace->GetOrbit(l);
            arma::vec L2_kl = L2.unsafe_col(indx_kl);
            arma::vec R2_kl = R2.unsafe_col(indx_kl);

            double cijkl = 0;


            for (int a : modelspace->OneBodyChannels.at({oi.l,oi.j2,oi.tz2}) )
            {
                 int indx_aj = tbc.GetLocalIndex(min(a,j),max(a,j));
                 if (indx_aj < 0) continue;
                 double pre_aj = a>j ? tbc.GetKet(indx_aj).Phase(tbc.J) : (a==j ? SQRT2 : 1);
                 cijkl += pre_kl * pre_aj  * ( L1(i,a) * R2_kl(indx_aj) - R1(i,a) * L2_kl(indx_aj) );
            }

            for (int a : modelspace->OneBodyChannels.at({oj.l,oj.j2,oj.tz2}) )
            {
                 int indx_ia = tbc.GetLocalIndex(min(i,a),max(i,a));
                 if (indx_ia < 0) continue;
                 double pre_ia = i>a ? tbc.GetKet(indx_ia).Phase(tbc.J) : (i==a ? SQRT2 : 1);
                 cijkl += pre_kl * pre_ia * ( L1(j,a) * R2_kl(indx_ia) - R1(j,a) * L2_kl(indx_ia) );
             }

            for (int a : modelspace->OneBodyChannels.at({ok.l,ok.j2,ok.tz2}) )
            {
                int indx_al = tbc.GetLocalIndex(min(a,l),max(a,l));
                if (indx_al < 0) continue;
                double pre_al = a>l ? tbc.GetKet(indx_al).Phase(tbc.J) : (a==l ? SQRT2 : 1);
                cijkl += pre_ij * pre_al * ( R1(a,k) * L2_ij(indx_al) - L1(a,k) * R2_ij(indx_al) );
            }

            for (int a : modelspace->OneBodyChannels.at({ol.l,ol.j2,ol.tz2}) )
            {
               int indx_ka = tbc.GetLocalIndex(min(k,a),max(k,a));
               if (indx_ka < 0) continue;
               double pre_ka = k>a ? tbc.GetKet(indx_ka).Phase(tbc.J) : (k==a ? SQRT2 : 1);
               cijkl += pre_ij * pre_ka * ( R1(a,l) * L2_ij(indx_ka) - L1(a,l) * R2_ij(indx_ka) );
            }

            double norm = bra.delta_pq()==ket.delta_pq() ? 1+bra.delta_pq() : SQRT2;
            OUT(indx_ij,indx_kl) += cijkl / norm;
         }
      }
   }

}





//*****************************************************************************************
//
//  |     |      |     |   
//  |__Y__|      |__x__|   [X2,Y2](2)_pp(hh) = 1/2 sum_ab (X_ijab Y_abkl - Y_ijab X_abkl)(1 - n_a - n_b)
//  |     |  _   |     |                = 1/2 [ X*(P_pp-P_hh)*Y - Y*(P_pp-P_hh)*X ]
//  |__X__|      |__Y__|   
//  |     |      |     |   
//
// -- AGREES WITH NATHAN'S RESULTS
//   No factor of 1/2 because the matrix multiplication corresponds to a restricted sum (a<=b) 
/// Calculates the part of the commutator \f$ [X_{(2)},Y_{(2)}]_{(2)} \f$ which involves particle-particle
/// or hole-hole intermediate states.
/// \f[
/// [X_{(2)},Y_{(2)}]^{J}_{ijkl} = \frac{1}{2} \sum_{ab} (\bar{n}_a\bar{n}_b - n_an_b) (X^{J}_{ijab}Y^{J}_{ablk} - Y^{J}_{ijab}X^{J}_{abkl})
/// \f]
/// This may be written as
/// \f[
/// [X_{(2)},Y_{(2)}]^{J} = \mathcal{M}^{J}_{pp} - \mathcal{M}^{J}_{hh}
/// \f]
/// With the intermediate matrices
/// \f[
/// \mathcal{M}^{J}_{pp} \equiv \frac{1}{2}(X^{J} \mathcal{P}_{pp} Y^{J} - Y^{J} \mathcal{P}_{pp} X^{J})
/// \f]
/// and likewise for \f$ \mathcal{M}^{J}_{hh} \f$.
void Operator::comm222_pp_hhss( Operator& opright, Operator& opout ) 
{
//   #pragma omp parallel for schedule(dynamic,5)
   for (int ch=0; ch<nChannels; ++ch)
   {
      TwoBodyChannel& tbc = modelspace->GetTwoBodyChannel(ch);
      arma::mat& LHS = (arma::mat&) TwoBody.GetMatrix(ch,ch);
      arma::mat& RHS = (arma::mat&) opright.TwoBody.GetMatrix(ch,ch);
      arma::mat& OUT = (arma::mat&) opout.TwoBody.GetMatrix(ch,ch);

      arma::mat Mpp = (LHS.rows(tbc.GetKetIndex_pp()) * RHS.cols(tbc.GetKetIndex_pp()));
      arma::mat Mhh = (LHS.rows(tbc.GetKetIndex_hh()) * RHS.cols(tbc.GetKetIndex_hh()));
      OUT += Mpp - Mpp.t() - Mhh + Mhh.t();
   }
}







/// Since comm222_pp_hhss() and comm221ss() both require the ruction of 
/// the intermediate matrices \f$\mathcal{M}_{pp} \f$ and \f$ \mathcal{M}_{hh} \f$, we can combine them and
/// only calculate the intermediates once.
void Operator::comm222_pp_hh_221ss( Operator& opright, Operator& opout )  
{

//   int herm = opout.IsHermitian() ? 1 : -1;
   int norbits = modelspace->GetNumberOrbits();

   TwoBodyME Mpp = opout.TwoBody;
   TwoBodyME Mhh = opout.TwoBody;

   double t = omp_get_wtime();
   // Don't use omp, because the matrix multiplication is already
   // parallelized by armadillo.
//   for (int ch=0;ch<nChannels;++ch)
   for (int ch : modelspace->SortedTwoBodyChannels)
   {
      TwoBodyChannel& tbc = modelspace->GetTwoBodyChannel(ch);

      auto& LHS = TwoBody.GetMatrix(ch,ch);
      auto& RHS = opright.TwoBody.GetMatrix(ch,ch);
      arma::mat& OUT =  opout.TwoBody.GetMatrix(ch,ch);

      arma::mat & Matrixpp = Mpp.GetMatrix(ch,ch);
      arma::mat & Matrixhh = Mhh.GetMatrix(ch,ch);

      arma::uvec& kets_pp = tbc.GetKetIndex_pp();
      arma::uvec& kets_hh = tbc.GetKetIndex_hh();
      
      Matrixpp =  LHS.cols(kets_pp) * RHS.rows(kets_pp);
      Matrixhh =  LHS.cols(kets_hh) * RHS.rows(kets_hh);

      if (opout.IsHermitian())
      {
         Matrixpp +=  Matrixpp.t();
         Matrixhh +=  Matrixhh.t();
      }
      else if (opout.IsAntiHermitian()) // i.e. LHS and RHS are both hermitian
      {
         Matrixpp -=  Matrixpp.t();
         Matrixhh -=  Matrixhh.t();
      }
      else
      {
        Matrixpp -=  RHS.cols(kets_pp) * LHS.rows(kets_pp);
        Matrixhh -=  RHS.cols(kets_hh) * LHS.rows(kets_hh);
      }

      // The two body part
      OUT += Matrixpp - Matrixhh;
   } //for ch
   timer["pphh TwoBody bit"] += omp_get_wtime() - t;

   t = omp_get_wtime();
   // The one body part
   #pragma omp parallel for schedule(dynamic,1)
   for (int i=0;i<norbits;++i)
   {
      Orbit &oi = modelspace->GetOrbit(i);
      int jmin = opout.IsNonHermitian() ? 0 : i;
      for (int j : modelspace->OneBodyChannels.at({oi.l,oi.j2,oi.tz2}) )
      {
         if (j<jmin) continue;
         double cijJ = 0;
         for (int ch=0;ch<nChannels;++ch)
         {
            TwoBodyChannel& tbc = modelspace->GetTwoBodyChannel(ch);
            double Jfactor = (2*tbc.J+1.0);
            // Sum c over holes and include the nbar_a * nbar_b terms
            for (auto& c : modelspace->holes)
            {
               cijJ += Mpp.GetTBME(ch,c,i,c,j) * Jfactor;
            // Sum c over particles and include the n_a * n_b terms
            }
            for (auto& c : modelspace->particles)
            {
               cijJ += Mhh.GetTBME(ch,c,i,c,j) * Jfactor;
            }
         }
         opout.OneBody(i,j) += cijJ /(oi.j2+1.0);
      } // for j
   } // for i
   timer["pphh One Body bit"] += omp_get_wtime() - t;
}



//**************************************************************************
//
//  X^J_ij`kl` = - sum_J' { i j J } (2J'+1) X^J'_ilkj
//                        { k l J'}
// SCALAR VARIETY
/// The scalar Pandya transformation is defined as
/// \f[
///  \bar{X}^{J}_{i\bar{j}k\bar{l}} = - \sum_{J'} (2J'+1)
///  \left\{ \begin{array}{lll}
///  j_i  &  j_j  &  J \\
///  j_k  &  j_l  &  J' \\
///  \end{array} \right\}
///  X^{J}_{ilkj}
/// \f]
/// where the overbar indicates time-reversed orbits.
/// This function is designed for use with comm222_phss() and so it takes in
/// two arrays of matrices, one for hp terms and one for ph terms.
//void Operator::DoPandyaTransformation(TwoBodyME& TwoBody_CC_hp, TwoBodyME& TwoBody_CC_ph)
//void Operator::DoPandyaTransformation(vector<arma::mat>& TwoBody_CC_hp, vector<arma::mat>& TwoBody_CC_ph)
void Operator::DoPandyaTransformation(vector<arma::mat>& TwoBody_CC_hp, vector<arma::mat>& TwoBody_CC_ph)
{
   // loop over cross-coupled channels
//   for (int ch_cc=0; ch_cc<nChannels; ++ch_cc)
   int n_nonzero = modelspace->SortedTwoBodyChannels_CC.size();
   int herm = IsHermitian() ? 1 : -1;
   #pragma omp parallel for schedule(dynamic,1) if (not modelspace->SixJ_is_empty())
   for (int ich=0; ich<n_nonzero; ++ich)
   {
      int ch_cc = modelspace->SortedTwoBodyChannels_CC[ich];
      TwoBodyChannel& tbc_cc = modelspace->GetTwoBodyChannel_CC(ch_cc);
      int nKets_cc = tbc_cc.GetNumberKets();
      arma::uvec& kets_ph = tbc_cc.GetKetIndex_ph();
      int nph_kets = kets_ph.n_rows;
      int J_cc = tbc_cc.J;

      TwoBody_CC_hp[ch_cc] = arma::mat(nph_kets,   2*nKets_cc, arma::fill::zeros);
      TwoBody_CC_ph[ch_cc] = arma::mat(nph_kets,   2*nKets_cc, arma::fill::zeros);

      // loop over cross-coupled ph bras <ac| in this channel
      for (int ibra=0; ibra<nph_kets; ++ibra)
      {
         Ket & bra_cc = tbc_cc.GetKet( kets_ph[ibra] );
         int a = bra_cc.p;
         int b = bra_cc.q;
         Orbit & oa = modelspace->GetOrbit(a);
         Orbit & ob = modelspace->GetOrbit(b);
         double ja = oa.j2*0.5;
         double jb = ob.j2*0.5;

         // loop over cross-coupled kets |bd> in this channel
         // we go to 2*nKets to include |bd> and |db>
//         for (int iket_cc=0; iket_cc<2*nKets_cc; ++iket_cc)
         for (int iket_cc=0; iket_cc<nKets_cc; ++iket_cc)
         {
            Ket & ket_cc = tbc_cc.GetKet(iket_cc%nKets_cc);
            int c = iket_cc < nKets_cc ? ket_cc.p : ket_cc.q;
            int d = iket_cc < nKets_cc ? ket_cc.q : ket_cc.p;
            Orbit & oc = modelspace->GetOrbit(c);
            Orbit & od = modelspace->GetOrbit(d);
            double jc = oc.j2*0.5;
            double jd = od.j2*0.5;


            int jmin = max(abs(ja-jd),abs(jc-jb));
            int jmax = min(ja+jd,jc+jb);
            double sm = 0;
            for (int J_std=jmin; J_std<=jmax; ++J_std)
            {
               double sixj = modelspace->GetSixJ(ja,jb,J_cc,jc,jd,J_std);
               if (abs(sixj) < 1e-8) continue;
               double tbme = TwoBody.GetTBME_J(J_std,a,d,c,b);
               sm -= (2*J_std+1) * sixj * tbme ;
            }
            TwoBody_CC_hp[ch_cc](ibra,iket_cc) = sm;
            TwoBody_CC_ph[ch_cc](ibra,iket_cc+nKets_cc) = herm* modelspace->phase(ja+jb+jc+jd) * sm;


            // Exchange (a <-> b) to account for the (n_a - n_b) term
            // Get Tz,parity and range of J for <bd || ca > coupling
            jmin = max(abs(jb-jd),abs(jc-ja));
            jmax = min(jb+jd,jc+ja);
            sm = 0;
            for (int J_std=jmin; J_std<=jmax; ++J_std)
            {
               double sixj = modelspace->GetSixJ(jb,ja,J_cc,jc,jd,J_std);
               if (abs(sixj) < 1e-8) continue;
               double tbme = TwoBody.GetTBME_J(J_std,b,d,c,a);
               sm -= (2*J_std+1) * sixj * tbme ;
            }
            TwoBody_CC_ph[ch_cc](ibra,iket_cc) = sm;
            TwoBody_CC_hp[ch_cc](ibra,iket_cc+nKets_cc) = herm* modelspace->phase(ja+jb+jc+jd) * sm;

         }
      }
   }
}


//void Operator::InversePandyaTransformation(vector<arma::mat>& W, vector<arma::mat>& opout)
void Operator::AddInversePandyaTransformation(vector<arma::mat>& W)
{
    // Do the inverse Pandya transform
    // Only go parallel if we've previously calculated the SixJ's. Otherwise, it's not thread safe.
//   for (int ch=0;ch<nChannels;++ch)
   int n_nonzeroChannels = modelspace->SortedTwoBodyChannels.size();
   #pragma omp parallel for schedule(dynamic,1) if (not modelspace->SixJ_is_empty())
   for (int ich = 0; ich < n_nonzeroChannels; ++ich)
   {
      int ch = modelspace->SortedTwoBodyChannels[ich];
      TwoBodyChannel& tbc = modelspace->GetTwoBodyChannel(ch);
      int J = tbc.J;
      int nKets = tbc.GetNumberKets();

//      opout[ch] = arma::mat(nKets,nKets,arma::fill::zeros);
      for (int ibra=0; ibra<nKets; ++ibra)
      {
         Ket & bra = tbc.GetKet(ibra);
         int i = bra.p;
         int j = bra.q;
         Orbit & oi = modelspace->GetOrbit(i);
         Orbit & oj = modelspace->GetOrbit(j);
         double ji = oi.j2/2.;
         double jj = oj.j2/2.;
         int ketmin = IsHermitian() ? ibra : ibra+1;
         for (int iket=ketmin; iket<nKets; ++iket)
         {
            Ket & ket = tbc.GetKet(iket);
            int k = ket.p;
            int l = ket.q;
            Orbit & ok = modelspace->GetOrbit(k);
            Orbit & ol = modelspace->GetOrbit(l);
            double jk = ok.j2/2.;
            double jl = ol.j2/2.;

            double commij = 0;
            double commji = 0;

            int parity_cc = (oi.l+ol.l)%2;
            int Tz_cc = abs(oi.tz2+ol.tz2)/2;
            int jmin = max(abs(int(ji-jl)),abs(int(jk-jj)));
            int jmax = min(int(ji+jl),int(jk+jj));

            for (int Jprime=jmin; Jprime<=jmax; ++Jprime)
            {
               double sixj = modelspace->GetSixJ(ji,jj,J,jk,jl,Jprime);
               if (abs(sixj)<1e-8) continue;
               int ch_cc = modelspace->GetTwoBodyChannelIndex(Jprime,parity_cc,Tz_cc);
               TwoBodyChannel_CC& tbc = modelspace->GetTwoBodyChannel_CC(ch_cc);
               int indx_il = tbc.GetLocalIndex(min(i,l),max(i,l));
               int indx_kj = tbc.GetLocalIndex(min(j,k),max(j,k));
               if (i>l) indx_il += tbc.GetNumberKets();
               if (k>j) indx_kj += tbc.GetNumberKets();
               double me1 = W[ch_cc](indx_il,indx_kj);
               commij += (2*Jprime+1) * sixj * me1;
            }

            // now loop over the cross coupled TBME's
            parity_cc = (oi.l+ok.l)%2;
            Tz_cc = abs(oi.tz2+ok.tz2)/2;
            jmin = max(abs(int(jj-jl)),abs(int(jk-ji)));
            jmax = min(int(jj+jl),int(jk+ji));
            for (int Jprime=jmin; Jprime<=jmax; ++Jprime)
            {
               double sixj = modelspace->GetSixJ(jj,ji,J,jk,jl,Jprime);
               if (abs(sixj)<1e-8) continue;
               int ch_cc = modelspace->GetTwoBodyChannelIndex(Jprime,parity_cc,Tz_cc);
               TwoBodyChannel_CC& tbc = modelspace->GetTwoBodyChannel_CC(ch_cc);
               int indx_jl = tbc.GetLocalIndex(min(j,l),max(j,l));
               int indx_ki = tbc.GetLocalIndex(min(i,k),max(i,k));
               if (j>l) indx_jl += tbc.GetNumberKets();
               if (k>i) indx_ki += tbc.GetNumberKets();
               double me1 = W[ch_cc](indx_jl,indx_ki);
               commji += (2*Jprime+1) *  sixj * me1;
            }

            double norm = bra.delta_pq()==ket.delta_pq() ? 1+bra.delta_pq() : SQRT2;
            TwoBody.GetMatrix(ch,ch)(ibra,iket) += (commij - modelspace->phase(ji+jj-J)*commji) / norm;
//            opout[ch](ibra,iket) = (commij - modelspace->phase(ji+jj-J)*commji) / norm;
         }
      }
   }
 
}



//*****************************************************************************************
//
//  THIS IS THE BIG UGLY ONE.     
//                                             
//   |          |      |          |           
//   |     __Y__|      |     __X__|            
//   |    /\    |      |    /\    |
//   |   (  )   |  _   |   (  )   |            
//   |____\/    |      |____\/    |            
//   |  X       |      |  Y       |            
//           
//            
// -- This appears to agree with Nathan's results
//
/// Calculates the part of \f$ [X_{(2)},Y_{(2)}]_{ijkl} \f$ which involves ph intermediate states, here indicated by \f$ Z^{J}_{ijkl} \f$
/// \f[
/// Z^{J}_{ijkl} = \sum_{ab}(n_a-n_b)\sum_{J'} (2J'+1)
/// \left[
///  \left\{ \begin{array}{lll}
///  j_i  &  j_j  &  J \\
///  j_k  &  j_l  &  J' \\
///  \end{array} \right\}
/// \left( \bar{X}^{J'}_{i\bar{l}a\bar{b}}\bar{Y}^{J'}_{a\bar{b}k\bar{j}} - 
///   \bar{Y}^{J'}_{i\bar{l}a\bar{b}}\bar{X}^{J'}_{a\bar{b}k\bar{j}} \right)
///  -(-1)^{j_i+j_j-J}
///  \left\{ \begin{array}{lll}
///  j_j  &  j_i  &  J \\
///  j_k  &  j_l  &  J' \\
///  \end{array} \right\}
/// \left( \bar{X}^{J'}_{j\bar{l}a\bar{b}}\bar{Y}^{J'}_{a\bar{b}k\bar{i}} - 
///   \bar{Y}^{J'}_{j\bar{l}a\bar{b}}\bar{X}^{J'}_{a\bar{b}k\bar{i}} \right)
/// \right]
/// \f]
/// This is implemented by defining an intermediate matrix
/// \f[
/// \bar{W}^{J}_{i\bar{l}k\bar{j}} \equiv \sum_{ab}(n_a\bar{n}_b)
/// \left[ \left( \bar{X}^{J'}_{i\bar{l}a\bar{b}}\bar{Y}^{J'}_{a\bar{b}k\bar{j}} - 
///   \bar{Y}^{J'}_{i\bar{l}a\bar{b}}\bar{X}^{J'}_{a\bar{b}k\bar{j}} \right)
/// -\left( \bar{X}^{J'}_{i\bar{l}b\bar{a}}\bar{Y}^{J'}_{b\bar{a}k\bar{j}} - 
///    \bar{Y}^{J'}_{i\bar{l}b\bar{a}}\bar{X}^{J'}_{b\bar{a}k\bar{j}} \right)\right]
/// \f]
/// The Pandya-transformed matrix elements are obtained with DoPandyaTransformation().
/// The commutator is then given by
/// \f[
/// Z^{J}_{ijkl} = \sum_{J'} (2J'+1)
/// \left[
///  \left\{ \begin{array}{lll}
///  j_i  &  j_j  &  J \\
///  j_k  &  j_l  &  J' \\
///  \end{array} \right\}
///  \bar{W}^{J'}_{i\bar{l}k\bar{j}}
///  -(-1)^{j_i+j_j-J}
///  \left\{ \begin{array}{lll}
///  j_j  &  j_i  &  J \\
///  j_k  &  j_l  &  J' \\
///  \end{array} \right\}
///  \bar{W}^{J'}_{j\bar{l}k\bar{i}}
///  \right]
///  \f]
///
void Operator::comm222_phss( Operator& Y, Operator& Z ) 
{

   Operator& X = *this;
   // Create Pandya-transformed hp and ph matrix elements
   vector<arma::mat> X_bar_hp (nChannels );
   vector<arma::mat> X_bar_ph (nChannels );
   vector<arma::mat> Y_bar_hp (nChannels );
   vector<arma::mat> Y_bar_ph (nChannels );

   double t = omp_get_wtime();
   X.DoPandyaTransformation(X_bar_hp, X_bar_ph );
   Y.DoPandyaTransformation(Y_bar_hp, Y_bar_ph );
   timer["DoPandyaTransformation"] += omp_get_wtime() - t;

   // Construct the intermediate matrix W_bar
   t = omp_get_wtime();
   vector<arma::mat> W_bar (nChannels );
//   int hx = X.IsHermitian() ? 1 : -1; // I don't remember why this is here, but the minus for anti-hermitian X makes sense...

   for (int ch : modelspace->SortedTwoBodyChannels_CC )
   {
      if ( X.IsHermitian() )
         W_bar[ch] =  X_bar_hp[ch].t() * Y_bar_hp[ch] - X_bar_ph[ch].t() * Y_bar_ph[ch] ;
      else
         W_bar[ch] =  X_bar_ph[ch].t() * Y_bar_ph[ch] - X_bar_hp[ch].t() * Y_bar_hp[ch] ;

      if ( Z.IsHermitian() )
        W_bar[ch] += W_bar[ch].t();
      else
        W_bar[ch] -= W_bar[ch].t();
   }
   timer["Build W_bar"] += omp_get_wtime() - t;

   // Perform inverse Pandya transform on W_bar to get Z
   t = omp_get_wtime();
   Z.AddInversePandyaTransformation(W_bar);
   timer["InversePandyaTransformation"] += omp_get_wtime() - t;

}






//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
////////////   BEGIN SCALAR-TENSOR COMMUTATORS      //////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////


//*****************************************************************************************
//
//        |____. Y          |___.X
//        |        _        |
//  X .___|            Y.___|              [X1,Y1](1)  =  XY - YX
//        |                 |
//
// This is no different from the scalar-scalar version
void Operator::comm111st( Operator & opright, Operator& out) 
{
   out.OneBody += OneBody*opright.OneBody - opright.OneBody*OneBody;
}

//*****************************************************************************************
//                                       |
//      i |              i |             |
//        |    ___.Y       |__X__        |
//        |___(_)    _     |   (_)__.    |  [X2,Y1](1)  =  1/(2j_i+1) sum_ab(n_a-n_b)y_ab 
//      j | X            j |        Y    |        * sum_J (2J+1) x_biaj^(J)  
//                                       |      
//---------------------------------------*        = 1/(2j+1) sum_a n_a sum_J (2J+1)
//                                                  * sum_b y_ab x_biaj - yba x_aibj
//
// X is scalar one-body, Y is tensor two-body
// There must be a better way to do this looping. 
//
void Operator::comm121st( Operator& Y, Operator& Z) 
{
   int norbits = modelspace->GetNumberOrbits();
   int Lambda = Z.GetJRank();
   Operator& X = *this;
//   #pragma omp parallel for // for starters, don't do it parallel
   for (int i=0;i<norbits;++i)
   {
      Orbit &oi = modelspace->GetOrbit(i);
      double ji = oi.j2/2.0;
      for (int j : X.OneBodyChannels.at({oi.l,oi.j2,oi.tz2}) ) 
      {
          Orbit &oj = modelspace->GetOrbit(j);
          double jj = oj.j2/2.0;
//          if (j<jmin) continue; // only calculate upper triangle
          double& Zij = Z.OneBody(i,j);
          for (auto& a : modelspace->holes)  // C++11 syntax
          {
             Orbit &oa = modelspace->GetOrbit(a);
             double ja = oa.j2/2.0;
               for (auto& b : modelspace->particles)
               {
                  Orbit &ob = modelspace->GetOrbit(b);
                  double jb = ob.j2/2.0;
                  if (ob.j2 == oa.j2 and ob.l == oa.l and ob.tz2 == oa.tz2)
                  {
                    int J1min = min(abs(ji-ja),abs(jj-ja));
                    int J1max = max(ji,jj) + ja;
                    for (int J1=J1min; J1<=J1max; ++J1)
                    {
                      int phasefactor = modelspace->phase(jj+ja+J1+Lambda);
                      int J2min = max(abs(Lambda - J1),J1min);
                      int J2max = min(Lambda + J1,J1max);
                      for (int J2=J2min; J2<=J2max; ++J2)
                      {
                        double prefactor = phasefactor * sqrt((2*J1+1)*(2*J2+1)) * modelspace->GetSixJ(J1,J2,Lambda,jj,ji,ja);
                        if (J1>=abs(ja-ji) and J1<=ja+ji and J2>=abs(ja-jj) and J2<=ja+jj )
                          Zij +=  prefactor * ( X.OneBody(a,b) * Y.TwoBody.GetTBME_J(J1,J2,b,i,a,j) );
                        if (J1>=abs(ja-jj) and J1<=ja+jj and J2>=abs(ja-ji) and J2<=ja+ji )
                          Zij -= X.OneBody(b,a) * Y.TwoBody.GetTBME_J(J1,J2,a,i,b,j)  ;
                      }
                    }
                  }

                  // Now, X is scalar two-body and Y is tensor one-body
                  if ( (abs(ja-jb)>Lambda) or (ja+jb<Lambda) ) continue;
                  int J1min = max(abs(ji-ja),abs(jj-jb));
                  int J1max = min(ji+ja,jj+jb);
                  for (int J1=J1min; J1<=J1max; ++J1)
                  {
                    double prefactor = modelspace->phase(ji+jb+J1) * (2*J1+1) * modelspace->GetSixJ(ja,jb,Lambda,ji,jj,J1);
                    Zij += prefactor * X.TwoBody.GetTBME_J(J1,J1,b,i,a,j) * Y.OneBody(a,b);
                  }

                  J1min = max(abs(ji-jb),abs(jj-ja));
                  J1max = min(ji+jb,jj+ja);
                  for (int J1=J1min; J1<=J1max; ++J1)
                  {
                    double prefactor = modelspace->phase(ji+ja+J1) * (2*J1+1) * modelspace->GetSixJ(jb,ja,Lambda,ji,jj,J1);
                    Zij += prefactor * X.TwoBody.GetTBME_J(J1,J1,a,i,b,j) * Y.OneBody(b,a);
                  }
               }
               
               
             }
          }
      }
   
}




//*****************************************************************************************
//
//      i |              i |            [X2,Y2](1)  =  1/(2(2j_i+1)) sum_J (2J+1) 
//        |__Y__           |__X__           * sum_abc (nbar_a*nbar_b*n_c + n_a*n_b*nbar_c)
//        |    /\          |    /\          * (x_ciab y_abcj - y_ciab xabcj)
//        |   (  )   _     |   (  )                                                                                      
//        |____\/          |____\/       = 1/(2(2j+1)) sum_J (2J+1)
//      j | X            j |  Y            *  sum_c ( Pp*X*Phh*Y*Pp - Pp*Y*Phh*X*Pp)  - (Ph*X*Ppp*Y*Ph - Ph*Y*Ppp*X*Ph)_cicj
//                                     
//
// -- AGREES WITH NATHAN'S RESULTS 
//   No factor of 1/2 because the matrix multiplication corresponds to a restricted sum (a<=b) 

// Combined with the comm222pphh because they use the same intermediates.





//*****************************************************************************************
//
//    |     |               |      |           [X2,Y1](2) = sum_a ( Y_ia X_ajkl + Y_ja X_iakl - Y_ak X_ijal - Y_al X_ijka )
//    |     |___.Y          |__X___|         
//    |     |         _     |      |          
//    |_____|               |      |_____.Y        
//    |  X  |               |      |            
//
// -- AGREES WITH NATHAN'S RESULTS
// Right now, this is the slowest one...
// Agrees with previous code in the scalar-scalar limit
void Operator::comm122st( Operator& Y, Operator& Z ) 
{
//   int herm = Z.IsHermitian() ? 1 : -1;
   int norbits = modelspace->GetNumberOrbits();
   int Lambda = Z.rank_J;
   Operator& X = *this;

//   #pragma omp parallel for schedule(dynamic,5)
    for ( auto& itmat : Y.TwoBody.MatEl )
    {
     int ch_bra = itmat.first[0];
     int ch_ket = itmat.first[1];

      TwoBodyChannel& tbc_bra = modelspace->GetTwoBodyChannel(ch_bra);
      TwoBodyChannel& tbc_ket = modelspace->GetTwoBodyChannel(ch_ket);
      int J1 = tbc_bra.J;
      int J2 = tbc_ket.J;
      int nbras = tbc_bra.GetNumberKets();
      int nkets = tbc_ket.GetNumberKets();
      double hatfactor = sqrt((2*J1+1)*(2*J2+1));
      arma::mat& OUT = (arma::mat&) Z.TwoBody.GetMatrix(ch_bra,ch_ket);

      for (int ibra = 0;ibra<nbras; ++ibra)
      {
         Ket & bra = tbc_bra.GetKet(ibra);
         int i = bra.p;
         int j = bra.q;
         Orbit& oi = modelspace->GetOrbit(i);
         Orbit& oj = modelspace->GetOrbit(j);
         double ji = oi.j2/2.0;
         double jj = oj.j2/2.0;
         for (int iket=0;iket<nkets; ++iket)
         {
            Ket & ket = tbc_ket.GetKet(iket);
            int k = ket.p;
            int l = ket.q;
            Orbit& ok = modelspace->GetOrbit(k);
            Orbit& ol = modelspace->GetOrbit(l);
            double jk = ok.j2/2.0;
            double jl = ol.j2/2.0;

            double cijkl = 0;
            for (int a=0;a<norbits;++a)
            {
              Orbit& oa = modelspace->GetOrbit(a);
              double ja = oa.j2/2.0;

               cijkl += X.OneBody(i,a) * Y.TwoBody.GetTBME(ch_bra,ch_ket,a,j,k,l);
               cijkl += X.OneBody(j,a) * Y.TwoBody.GetTBME(ch_bra,ch_ket,i,a,k,l);
               cijkl -= X.OneBody(a,k) * Y.TwoBody.GetTBME(ch_bra,ch_ket,i,j,a,l);
               cijkl -= X.OneBody(a,l) * Y.TwoBody.GetTBME(ch_bra,ch_ket,i,j,k,a);


               double prefactor = hatfactor * modelspace->phase(ji+jj+J2+Lambda) * modelspace->GetSixJ(J2,J1,Lambda,ji,ja,jj);
               cijkl -= prefactor * Y.OneBody(i,a) * X.TwoBody.GetTBME(ch_bra,ch_bra,a,j,k,l) ;

               prefactor = hatfactor * modelspace->phase(ji+ja+J1+Lambda) * modelspace->GetSixJ(J2,J1,Lambda,jj,ja,ji);
               cijkl -= prefactor * Y.OneBody(j,a) * X.TwoBody.GetTBME(ch_bra,ch_bra,i,a,k,l);

               prefactor = hatfactor * modelspace->phase(jl+ja+J2+Lambda) * modelspace->GetSixJ(J1,J2,Lambda,jk,ja,jl);
               cijkl += prefactor * Y.OneBody(a,k) * X.TwoBody.GetTBME(ch_bra,ch_bra,i,j,a,l) ;

               prefactor = hatfactor * modelspace->phase(jl+jk+J1+Lambda) * modelspace->GetSixJ(J1,J2,Lambda,jl,ja,jk);
               cijkl += prefactor * Y.OneBody(a,l) * X.TwoBody.GetTBME(ch_bra,ch_bra,i,j,k,a) ;

            }
//            double norm = bra.delta_pq()==ket.delta_pq() ? 1+bra.delta_pq() : SQRT2;
//            cijkl /= norm;
            #pragma omp critical
            {
            OUT(ibra,iket) += cijkl;
//            if (ibra != iket)
//               OUT(iket,ibra) += herm * cijkl;
            }
         }
      }
   }
}





//*****************************************************************************************
//
//  |     |      |     |   
//  |__Y__|      |__x__|   [X2,Y2](2)_pp(hh) = 1/2 sum_ab (X_ijab Y_abkl - Y_ijab X_abkl)(1 - n_a - n_b)
//  |     |  _   |     |                = 1/2 [ X*(P_pp-P_hh)*Y - Y*(P_pp-P_hh)*X ]
//  |__X__|      |__Y__|   
//  |     |      |     |              ( note that   1-n_a-n_b  =  nbar_a nbar_b - n_an_b )
//
// -- AGREES WITH NATHAN'S RESULTS
//   No factor of 1/2 because the matrix multiplication corresponds to a restricted sum (a<=b) 
// Combined with comm221 because they have the same intermediates







// Since comm222_pp_hh and comm211 both require the ruction of 
// the intermediate matrices Mpp and Mhh, we can combine them and
// only calculate the intermediates once.
// X is a scalar, Y is a tensor
void Operator::comm222_pp_hh_221st( Operator& Y, Operator& Z )  
{

   Operator& X = *this;
   int Lambda = Z.GetJRank();
//   int herm = Z.IsHermitian() ? 1 : -1;
   int norbits = modelspace->GetNumberOrbits();

   TwoBodyME Mpp = Z.TwoBody;
   TwoBodyME Mhh = Z.TwoBody;

//   #pragma omp parallel for schedule(dynamic,5)
     for ( auto& itmat : Y.TwoBody.MatEl )
     {
      int ch_bra = itmat.first[0];
      int ch_ket = itmat.first[1];
      TwoBodyChannel& tbc_bra = modelspace->GetTwoBodyChannel(ch_bra);
      TwoBodyChannel& tbc_ket = modelspace->GetTwoBodyChannel(ch_ket);

      auto& LHS1 = X.TwoBody.GetMatrix(ch_bra,ch_bra);
      auto& LHS2 = X.TwoBody.GetMatrix(ch_ket,ch_ket);

      auto& RHS  =  itmat.second;
      arma::mat& OUT2 =    Z.TwoBody.GetMatrix(ch_bra,ch_ket);

      arma::mat& Matrixpp =  Mpp.GetMatrix(ch_bra,ch_ket);
      arma::mat& Matrixhh =  Mhh.GetMatrix(ch_bra,ch_ket);
     
      arma::uvec& bras_pp = tbc_bra.GetKetIndex_pp();
      arma::uvec& bras_hh = tbc_bra.GetKetIndex_hh();
      arma::uvec& kets_pp = tbc_ket.GetKetIndex_pp();
      arma::uvec& kets_hh = tbc_ket.GetKetIndex_hh();
      
      Matrixpp =  LHS1.cols(bras_pp) * RHS.rows(bras_pp) - RHS.cols(kets_pp)*LHS2.rows(kets_pp);
      Matrixhh =  LHS1.cols(bras_hh) * RHS.rows(bras_hh) - RHS.cols(kets_hh)*LHS2.rows(kets_hh);
 

      // Now, the two body part is easy
      OUT2 += Matrixpp - Matrixhh;


      // The one body part takes some additional work
      int J1 = tbc_bra.J;
      int J2 = tbc_ket.J;
      double hatfactor = sqrt( (2*J1+1)*(2*J2+1) );

      for (int i=0;i<norbits;++i)
      {
         Orbit &oi = modelspace->GetOrbit(i);
         double ji = oi.j2/2.0;
         for (int j : Z.OneBodyChannels.at({oi.l, oi.j2, oi.tz2}) )
         {
            Orbit &oj = modelspace->GetOrbit(j);
            double jj = oj.j2/2.0;
            double cijJ = 0;
            // Sum c over holes and include the nbar_a * nbar_b terms
            for (auto& c : modelspace->holes)
            {
               Orbit &oc = modelspace->GetOrbit(c);
               double jc = oc.j2/2.0;
               if ( not AngMom::Triangle(jc,ji,J1) or not AngMom::Triangle(jc,jj,J1)) continue;
               double sixj = modelspace->GetSixJ(J1, J2, Lambda, jj, ji, jc);
               cijJ +=   sixj * modelspace->phase(jj + jc + J1 + Lambda) * Mpp.GetTBME(ch_bra,ch_ket,c,i,j,c);
            // Sum c over particles and include the n_a * n_b terms
            }
            for (auto& c : modelspace->particles)
            {
               Orbit &oc = modelspace->GetOrbit(c);
               double jc = oc.j2/2.0;
               if ( not AngMom::Triangle(jc,ji,J1) or not AngMom::Triangle(jc,jj,J1)) continue;
               double sixj = modelspace->GetSixJ(J1, J2, Lambda, jj, ji, jc);
               cijJ -=   sixj * modelspace->phase(jj + jc + J1 + Lambda) * Mhh.GetTBME(ch_bra,ch_ket,c,i,j,c);
            }
            #pragma omp critical
            Z.OneBody(i,j) += cijJ *hatfactor ;
         } // for j
       } // for i
   } //for itmat
}






//**************************************************************************
//
//  X^J_ij`kl` = - sum_J' { i j J } (2J'+1) X^J'_ilkj
//                        { k l J'}
// TENSOR VARIETY
/// The scalar Pandya transformation is defined as
/// \f[
///  \bar{X}^{J}_{i\bar{j}k\bar{l}} = - \sum_{J'} (2J'+1)
///  \left\{ \begin{array}{lll}
///  j_i  &  j_j  &  J \\
///  j_k  &  j_l  &  J' \\
///  \end{array} \right\}
///  X^{J}_{ilkj}
/// \f]
/// where the overbar indicates time-reversed orbits.
/// This function is designed for use with comm222_phss() and so it takes in
/// two arrays of matrices, one for hp terms and one for ph terms.
//void Operator::DoTensorPandyaTransformation(vector<arma::mat>& TwoBody_CC_hp, vector<arma::mat>& TwoBody_CC_ph)
void Operator::DoTensorPandyaTransformation(map<array<int,2>,arma::mat>& TwoBody_CC_hp, map<array<int,2>,arma::mat>& TwoBody_CC_ph)
{
   int Lambda = rank_J;
   // loop over cross-coupled channels
//   #pragma omp parallel for schedule(dynamic,1) if (not modelspace->SixJ_is_empty())
   for ( int ch_bra_cc : modelspace->SortedTwoBodyChannels_CC )
   {
      TwoBodyChannel& tbc_bra_cc = modelspace->GetTwoBodyChannel_CC(ch_bra_cc);
      int Jbra_cc = tbc_bra_cc.J;
      arma::uvec& bras_ph = tbc_bra_cc.GetKetIndex_ph();
      int nph_bras = bras_ph.n_rows;
      for ( int ch_ket_cc : modelspace->SortedTwoBodyChannels_CC )
      {
        TwoBodyChannel& tbc_ket_cc = modelspace->GetTwoBodyChannel_CC(ch_ket_cc);
        int Jket_cc = tbc_ket_cc.J;
        if ( (Jbra_cc+Jket_cc < rank_J) or abs(Jbra_cc-Jket_cc)>rank_J ) continue;
        if ( (tbc_bra_cc.parity + tbc_ket_cc.parity + parity)%2>0 ) continue;

        int nKets_cc = tbc_ket_cc.GetNumberKets();

        // Need to make these maps to account for ch_bra and ch_ket.
        TwoBody_CC_hp[{ch_bra_cc,ch_ket_cc}] = arma::mat(nph_bras,   2*nKets_cc, arma::fill::zeros);
        TwoBody_CC_ph[{ch_bra_cc,ch_ket_cc}] = arma::mat(nph_bras,   2*nKets_cc, arma::fill::zeros);

        // loop over cross-coupled ph bras <ac| in this channel
        for (int ibra=0; ibra<nph_bras; ++ibra)
        {
           Ket & bra_cc = tbc_bra_cc.GetKet( bras_ph[ibra] );
           int a = bra_cc.p;
           int b = bra_cc.q;
           Orbit & oa = modelspace->GetOrbit(a);
           Orbit & ob = modelspace->GetOrbit(b);
           double ja = oa.j2*0.5;
           double jb = ob.j2*0.5;

           // loop over kets |bd> in this channel
           // we go to 2*nKets to include |bd> and |db>
           for (int iket_cc=0; iket_cc<2*nKets_cc; ++iket_cc)
           {
              Ket & ket_cc = tbc_ket_cc.GetKet(iket_cc%nKets_cc);
              int c = iket_cc < nKets_cc ? ket_cc.p : ket_cc.q;
              int d = iket_cc < nKets_cc ? ket_cc.q : ket_cc.p;
              Orbit & oc = modelspace->GetOrbit(c);
              Orbit & od = modelspace->GetOrbit(d);
              double jc = oc.j2*0.5;
              double jd = od.j2*0.5;


              int j1min = abs(ja-jd);
              int j2min = abs(jb-jc);
              int j1max = ja+jd;
              int j2max = jc+jb;
              double sm = 0;
              for (int J1=j1min; J1<=j1max; ++J1)
              {
                for (int J2=j2min; J2<=j2max; ++J2)
                {
                  double ninej = modelspace->GetNineJ(ja,jd,Jbra_cc,jb,jc,Jket_cc,J1,J2,Lambda);
                  if (abs(ninej) < 1e-8) continue;
                  double hatfactor = sqrt( (2*J1+1)*(2*J2+1)*(2*Jbra_cc+1)*(2*Jket_cc+1) );
                  double tbme = TwoBody.GetTBME_J(J1,J2,a,d,c,b);
                  sm -= hatfactor * modelspace->phase(jb+jd+Jket_cc+J2) * ninej * tbme ;
                }
              }
              TwoBody_CC_hp[{ch_bra_cc,ch_ket_cc}](ibra,iket_cc) = sm;


              // Exchange (a <-> b) to account for the (n_a - n_b) term
              // Get Tz,parity and range of J for <bd || ca > coupling
              j1min = abs(jb-jd);
              j2min = abs(jc-ja);
              j1max = jb+jd;
              j2max = jc+ja;
              sm = 0;
              for (int J1=j1min; J1<=j1max; ++J1)
              {
                for (int J2=j2min; J2<=j2max; ++J2)
                {
                  double ninej = modelspace->GetNineJ(jb,jd,Jbra_cc,ja,jc,Jket_cc,J1,J2,Lambda);
                  if (abs(ninej) < 1e-8) continue;
                  double hatfactor = sqrt( (2*J1+1)*(2*J2+1)*(2*Jbra_cc+1)*(2*Jket_cc+1) );
                  double tbme = TwoBody.GetTBME_J(J1,J2,b,d,c,a);
                  sm -= hatfactor * modelspace->phase(ja+jd+Jket_cc+J2) * ninej * tbme ;
                }
              }
              TwoBody_CC_ph[{ch_bra_cc,ch_ket_cc}](ibra,iket_cc) = sm;


           }
        }
    }
   }
}


void Operator::InverseTensorPandyaTransformation(map<array<int,2>,arma::mat>& W, map<array<int,2>,arma::mat>& opout, bool hermitian)
{
    // Do the inverse Pandya transform
    // Only go parallel if we've previously calculated the SixJ's. Otherwise, it's not thread safe.
//   for (int ch=0;ch<nChannels;++ch)
   int n_nonzeroChannels = modelspace->SortedTwoBodyChannels.size();
   #pragma omp parallel for schedule(dynamic,1) if (not modelspace->SixJ_is_empty())
   for (int ich = 0; ich < n_nonzeroChannels; ++ich)
   {
      int ch = modelspace->SortedTwoBodyChannels[ich];
      TwoBodyChannel& tbc = modelspace->GetTwoBodyChannel(ch);
      int J = tbc.J;
      int nKets = tbc.GetNumberKets();

      opout[{ch,ch}] = arma::mat(nKets,nKets,arma::fill::zeros);
      for (int ibra=0; ibra<nKets; ++ibra)
      {
         Ket & bra = tbc.GetKet(ibra);
         int i = bra.p;
         int j = bra.q;
         Orbit & oi = modelspace->GetOrbit(i);
         Orbit & oj = modelspace->GetOrbit(j);
         double ji = oi.j2/2.;
         double jj = oj.j2/2.;
         int ketmin = hermitian ? ibra : ibra+1;
         for (int iket=ketmin; iket<nKets; ++iket)
         {
            Ket & ket = tbc.GetKet(iket);
            int k = ket.p;
            int l = ket.q;
            Orbit & ok = modelspace->GetOrbit(k);
            Orbit & ol = modelspace->GetOrbit(l);
            double jk = ok.j2/2.;
            double jl = ol.j2/2.;

            double commij = 0;
            double commji = 0;

            int parity_cc = (oi.l+ol.l)%2;
            int Tz_cc = abs(oi.tz2+ol.tz2)/2;
            int jmin = max(abs(int(ji-jl)),abs(int(jk-jj)));
            int jmax = min(int(ji+jl),int(jk+jj));

            for (int Jprime=jmin; Jprime<=jmax; ++Jprime)
            {
               double sixj = modelspace->GetSixJ(ji,jj,J,jk,jl,Jprime);
               if (abs(sixj)<1e-8) continue;
               int ch_cc = modelspace->GetTwoBodyChannelIndex(Jprime,parity_cc,Tz_cc);
               TwoBodyChannel_CC& tbc = modelspace->GetTwoBodyChannel_CC(ch_cc);
               int indx_il = tbc.GetLocalIndex(min(i,l),max(i,l));
               int indx_kj = tbc.GetLocalIndex(min(j,k),max(j,k));
               if (i>l) indx_il += tbc.GetNumberKets();
               if (k>j) indx_kj += tbc.GetNumberKets();
               double me1 = W[{ch_cc,ch_cc}](indx_il,indx_kj);
               commij += (2*Jprime+1) * sixj * me1;
            }

            // now loop over the cross coupled TBME's
            parity_cc = (oi.l+ok.l)%2;
            Tz_cc = abs(oi.tz2+ok.tz2)/2;
            jmin = max(abs(int(jj-jl)),abs(int(jk-ji)));
            jmax = min(int(jj+jl),int(jk+ji));
            for (int Jprime=jmin; Jprime<=jmax; ++Jprime)
            {
               double sixj = modelspace->GetSixJ(jj,ji,J,jk,jl,Jprime);
               if (abs(sixj)<1e-8) continue;
               int ch_cc = modelspace->GetTwoBodyChannelIndex(Jprime,parity_cc,Tz_cc);
               TwoBodyChannel_CC& tbc = modelspace->GetTwoBodyChannel_CC(ch_cc);
               int indx_jl = tbc.GetLocalIndex(min(j,l),max(j,l));
               int indx_ki = tbc.GetLocalIndex(min(i,k),max(i,k));
               if (j>l) indx_jl += tbc.GetNumberKets();
               if (k>i) indx_ki += tbc.GetNumberKets();
               double me1 = W[{ch_cc,ch_cc}](indx_jl,indx_ki);
               commji += (2*Jprime+1) *  sixj * me1;
            }

            double norm = bra.delta_pq()==ket.delta_pq() ? 1+bra.delta_pq() : SQRT2;
            opout[{ch,ch}](ibra,iket) = (commij - modelspace->phase(ji+jj-J)*commji) / norm;;
         }
      }
   }
 
}




//*****************************************************************************************
//
//  THIS IS THE BIG UGLY ONE.     
//                                             
//   |          |      |          |           
//   |     __Y__|      |     __X__|            
//   |    /\    |      |    /\    |
//   |   (  )   |  _   |   (  )   |            
//   |____\/    |      |____\/    |            
//   |  X       |      |  Y       |            
//           
//            
// -- This appears to agree with Nathan's results
//
/// Calculates the part of \f$ [X_{(2)},Y_{(2)}]_{ijkl} \f$ which involves ph intermediate states, here indicated by \f$ Z^{J}_{ijkl} \f$
/// \f[
/// Z^{J}_{ijkl} = \sum_{ab}(n_a-n_b)\sum_{J'} (2J'+1)
/// \left[
///  \left\{ \begin{array}{lll}
///  j_i  &  j_j  &  J \\
///  j_k  &  j_l  &  J' \\
///  \end{array} \right\}
/// \left( \bar{X}^{J'}_{i\bar{l}a\bar{b}}\bar{Y}^{J'}_{a\bar{b}k\bar{j}} - 
///   \bar{Y}^{J'}_{i\bar{l}a\bar{b}}\bar{X}^{J'}_{a\bar{b}k\bar{j}} \right)
///  -(-1)^{j_i+j_j-J}
///  \left\{ \begin{array}{lll}
///  j_j  &  j_i  &  J \\
///  j_k  &  j_l  &  J' \\
///  \end{array} \right\}
/// \left( \bar{X}^{J'}_{j\bar{l}a\bar{b}}\bar{Y}^{J'}_{a\bar{b}k\bar{i}} - 
///   \bar{Y}^{J'}_{j\bar{l}a\bar{b}}\bar{X}^{J'}_{a\bar{b}k\bar{i}} \right)
/// \right]
/// \f]
/// This is implemented by defining an intermediate matrix
/// \f[
/// \bar{W}^{J}_{i\bar{l}k\bar{j}} \equiv \sum_{ab}(n_a\bar{n}_b)
/// \left[ \left( \bar{X}^{J'}_{i\bar{l}a\bar{b}}\bar{Y}^{J'}_{a\bar{b}k\bar{j}} - 
///   \bar{Y}^{J'}_{i\bar{l}a\bar{b}}\bar{X}^{J'}_{a\bar{b}k\bar{j}} \right)
/// -\left( \bar{X}^{J'}_{i\bar{l}b\bar{a}}\bar{Y}^{J'}_{b\bar{a}k\bar{j}} - 
///    \bar{Y}^{J'}_{i\bar{l}b\bar{a}}\bar{X}^{J'}_{b\bar{a}k\bar{j}} \right)\right]
/// \f]
/// The Pandya-transformed matrix elements are obtained with DoPandyaTransformation().
/// The commutator is then given by
/// \f[
/// Z^{J}_{ijkl} = \sum_{J'} (2J'+1)
/// \left[
///  \left\{ \begin{array}{lll}
///  j_i  &  j_j  &  J \\
///  j_k  &  j_l  &  J' \\
///  \end{array} \right\}
///  \bar{W}^{J'}_{i\bar{l}k\bar{j}}
///  -(-1)^{j_i+j_j-J}
///  \left\{ \begin{array}{lll}
///  j_j  &  j_i  &  J \\
///  j_k  &  j_l  &  J' \\
///  \end{array} \right\}
///  \bar{W}^{J'}_{j\bar{l}k\bar{i}}
///  \right]
///  \f]
///
void Operator::comm222_phst( Operator& Y, Operator& Z ) 
{

   Operator& X = *this;
   // Create Pandya-transformed hp and ph matrix elements
   vector<arma::mat> X_bar_hp (nChannels );
   vector<arma::mat> X_bar_ph (nChannels );

//   vector<arma::mat> Y_bar_hp (nChannels );
//   vector<arma::mat> Y_bar_ph (nChannels );
   vector<arma::mat> Y_bar_hp ( Y.TwoBody.MatEl.size() );
   vector<arma::mat> Y_bar_ph ( Y.TwoBody.MatEl.size() );

   double t = omp_get_wtime();
   X.DoPandyaTransformation(X_bar_hp, X_bar_ph );
//   Y.DoTensorPandyaTransformation(Y_bar_hp, Y_bar_ph );
   timer["DoTensorPandyaTransformation"] += omp_get_wtime() - t;

   t = omp_get_wtime();
   // Construct the intermediate matrix W_bar
   vector<arma::mat> W_bar (nChannels );
   int hx = X.IsHermitian() ? 1 : -1;
   int hy = Y.IsHermitian() ? 1 : -1;

   int nch = modelspace->SortedTwoBodyChannels_CC.size();
//   #pragma omp parallel for
   for (int ich=0; ich<nch; ++ich)
   {
      int ch = modelspace->SortedTwoBodyChannels_CC[ich];
      W_bar[ch] =  hx*( X_bar_hp[ch].t() * Y_bar_hp[ch] - X_bar_ph[ch].t() * Y_bar_ph[ch]) ;
      if (hx*hy<0)
        W_bar[ch] += W_bar[ch].t();
      else
        W_bar[ch] -= W_bar[ch].t();
   }
   timer["Build W_bar"] += omp_get_wtime() - t;

   vector<arma::mat> W (nChannels );
   t = omp_get_wtime();
   //InverseTensorPandyaTransformation(W_bar, W, Z.IsHermitian());
   timer["InverseTensorPandyaTransformation"] += omp_get_wtime() - t;
   for (int ch=0; ch<nChannels; ++ch)
   {
     Z.TwoBody.GetMatrix(ch,ch) += W[ch];
   }

}









