#include <math.h>

#include "TestRayleighWave.h"
#include "F77_FUNC.h"

extern "C" {
double F77_FUNC(rvel,RVEL)(double *lambda, double *mu);
}

TestRayleighWave:: TestRayleighWave( double rho, double cs, double cp, int nwl, double xmax ) : 
  m_rho(rho), m_cs(cs), m_cp(cp), m_alpha(0.0) 
// m_omega must be assigned before the exact solution can be evaluated
{
// calculate wave length and wave number
  double Lwave = xmax/(cos(m_alpha)*nwl);
  m_omega = 2*M_PI/Lwave;

// calculate the phase velocity
  double xi;
  
  m_mu = m_cs*m_cs*m_rho;
  m_lambda = m_cp*m_cp*m_rho-2*m_mu;
  xi = F77_FUNC(rvel,RVEL)( &m_lambda, &m_mu );
  m_cr = xi*m_cs;
}