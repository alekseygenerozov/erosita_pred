import cgs_const as cgs
from astropy.io import ascii
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as IUS

from scipy.optimize import newton, brentq
import matplotlib.pyplot as plt
import matplotlib as mpl
import labelLine
from astropy.io import fits
import os


hnu_min_ros=0.2
hnu_max_ros=2.0
ms=cgs.M_sun

##flim should not be a fixed number--It depends on the spectrum 
dh=3.0/0.7*1e9*cgs.pc
# flim=1.06e-12
omega_m=0.3
omega_l=0.7
#--------------------------------------------------------------------------------------------------_#
##Shankar mass function
loc=os.path.dirname(__file__)+'/../dat/'
phi_dat=ascii.read(loc+"shankar_mass_func.txt")
phi_dat=phi_dat[phi_dat['z']==0.02]
phi_interp=IUS(phi_dat['logM'],  phi_dat['logP'])
#--------------------------------------------------------------------------------------------------_#
##ZAMS stellar radii mist 
rdat = np.genfromtxt(loc+"r_zams.csv", delimiter=',')
rinterp = IUS(rdat[:,0], rdat[:,1])
#--------------------------------------------------------------------------------------------------_#
##eROSITA calibration
ff=fits.open(loc+'eros_200nmAl_200nmPI_sdtq.fits')
ff2=fits.open(loc+'eros_100nmAl_200nmPI_sdtq.fits')
dat=ff[1].data
ords=0.5*(dat['ENERG_LO']+dat['ENERG_HI'])
dat2=ff2[1].data
ords2=0.5*(dat2['ENERG_LO']+dat['ENERG_HI'])
e_resp=IUS(ords, 5.0*dat['SPECRESP']+2.0*dat2['SPECRESP'])
#--------------------------------------------------------------------------------------------------_#
abs_dat=np.genfromtxt(loc+'abs2.csv')
abs_dat=IUS(abs_dat[:,0], abs_dat[:,1])
t_integration=240.
#--------------------------------------------------------------------------------------------------_#
def abs1(en):
	'''
	Absorption as a function of energy 
	'''
	return abs_dat(en)

def rs(M):
	#return rinterp(M/cgs.M_sun)*cgs.R_sun
	return cgs.R_sun

def phi(M):
	return 10.**phi_interp(np.log10(M/cgs.M_sun))/(1.0e6*cgs.pc)**(3.0)

def log_integral(x1, x2, xs, ys):
	'''
	Compute int_{u1}^{u2} e^u y(u) du, where u is log(x), u1=log(x1), and u2=log(x2). 
	y(u) is conputed from an interpolating spline constructed from xs and ys
	'''
	us=np.log(xs)
	return IUS(us, ys*np.exp(us)).integral(np.log(x1), np.log(x2))

def rg(M):
	return cgs.G*M/cgs.c**2.0

def Ledd(M):
	return 4.0*np.pi*cgs.G*M*cgs.mp*cgs.c/(0.4*cgs.mp)

def rt(M):
	return (M/ms)**(1./3.)*rs(ms)

def Mhill(a):
	'''
	Newtonian estimate for the Hill's mass
	'''
	return (risco(a)*rg(ms)/rs(ms))**(-1.5)*ms

def EE(z):
	return (omega_m*(1+z)**3.0+omega_l)**0.5

def J(z):
	ords=np.linspace(0,z,500)
	absc=1.0/EE(ords)

	return IUS(ords, absc).integral(0,z)

def dVc(z):
	return 2.0*np.pi*dh**3.0*J(z)**2.0/EE(z)

def Z1(a):
	return 1 + (1 - a**2)**(1./3.)*((1 - a)**(1./3.) + (1 + a)**(1./3.))

def Z2(a):
	return (3.0*a**2.0+Z1(a)**2.0)**0.5

def risco(a):
	return 3.0+Z2(a)-np.sign(a)*np.sqrt((3-Z1(a))*(3.0+Z1(a)+2.0*Z2(a)))

def eta(a):
	return 1.0-np.sqrt(1.0-2.0/(3.0*risco(a)))

def to1(M):
	return 3.5e6*(M/1.0e6/cgs.M_sun)**0.5

def mdot(t, M, q):
	return ms/2/to1(M)*(q-1.0)*(t/to1(M))**(-q)

##Need to fix q-dependence here...
def Lo1(M, *, spin=0, **kwargs):
	return eta(spin)*mdot(to1(M), M, 1.2)*cgs.c**2.0

@np.vectorize
def Lo(M, *, spin=0, **kwargs):
	return min(Ledd(M), Lo1(M, spin=spin, **kwargs))

def tedd(M, q, *, spin=0, **kwargs):
	return to1(M)*(Ledd(M)/Lo1(M, spin=spin))**(-1.0/q)

@np.vectorize
def to(M, q, *, spin=0, **kwargs):
	return max(tedd(M, q, spin=spin, **kwargs), to1(M))

def flim(ss):
	nu_ords=np.linspace(hnu_min_ros, hnu_max_ros, 500)
	return 40.*1.0e3*cgs.eV/IUS(nu_ords, ss(nu_ords)*e_resp(nu_ords)*abs1(nu_ords)/nu_ords).integral(nu_ords[0], nu_ords[-1])/t_integration

@np.vectorize
def spec_disk(M, mdot1, nu, *, spin=0, **kwargs):
	rin=risco(spin)*rg(M)
	rout=2.0*rt(M)
	x=1.36
	teff_in=(3.*cgs.G*mdot1*M/(8.0*np.pi*x**3.0*rin**3.0)*(1.0-1.0/x**0.5)/cgs.sigma_sb)**(0.25)
	nu_br2=cgs.kb*teff_in/(1.0e3*cgs.eV)
	nu_br1=cgs.kb*teff_in*(rout/x/rin)**(-0.75)*(1.0-(1/x)**0.5)**(0.25)/(1.0-(rin/rout)**0.5)**0.25/(1.0e3*cgs.eV)
	norm=nu_br1/3. + (3*nu_br1*(-1 + (nu_br2/nu_br1)**(4./3.)))/4. + 5.0*nu_br2*(nu_br2/nu_br1)**(1./3.)

	if nu<nu_br1:
		return (nu/nu_br1)**2.0/norm
	elif nu>nu_br2:
		return (nu_br2/nu_br1)**(1./3.)*(nu/nu_br2)**2.0*np.exp(-(nu-nu_br2)/nu_br2)/norm
	else: 
		return (nu/nu_br1)**(1./3.)/norm

def K(z, spec):
	nu_ords=np.linspace(hnu_min_ros, hnu_max_ros, 500)
	int1=IUS(nu_ords, spec(nu_ords*(1+z))*abs1(nu_ords)*e_resp(nu_ords)/nu_ords).integral(nu_ords[0], nu_ords[-1])
	int2=IUS(nu_ords, spec(nu_ords)*abs1(nu_ords)*e_resp(nu_ords)/nu_ords).integral(nu_ords[0], nu_ords[-1])
	# print(int1,spec(nu_ords)*e_resp(nu_ords))
	return ((1.0+z)*int1/int2)**(-1.0)

def bol_correct(spec, nu_min, nu_max):
	nu_ords=np.logspace(np.log10(nu_min), np.log10(nu_max), 2000)
	int1=log_integral(nu_min, nu_max, nu_ords, spec(nu_ords))	

	return int1


def zlim(M, q, *, spin=0, **kwargs):
	##Should be to below???
	ss=lambda nu: spec_disk(M, mdot(to(M, q, spin=spin, **kwargs), M, q), nu, spin=spin, **kwargs)
	flim1=flim(ss)
	return newton(lambda z1:(1.0+z1)**2.0*J(z1)**2.0*K(z1, ss)-Lo(M)/4.0/np.pi/dh**2.0/flim1, 0.2)

def tend(M, q, z, *, spin=0, **kwargs):
	tt=to(M, q, spin=spin, **kwargs)
	return brentq(lambda tx: (Lo(M, spin=spin, **kwargs)/\
		(4.0*np.pi*dh**2.0*K(z, lambda nu: spec_disk(M, mdot(tx*to(M, q, spin=spin, **kwargs), M, q), nu, spin=spin, **kwargs))\
			*flim(lambda nu: spec_disk(M, mdot(tx*to(M, q, spin=spin, **kwargs), M, q), nu, spin=spin, **kwargs))*(1+z)**2.*J(z)**2.0))\
	**(1.0/q)-tx, 1, 1000)*tt

def ndot_tde(M):
	return 2.9e-5/cgs.year*(M/1.0e8/cgs.M_sun)**-0.4*phi(M)

def ttot(M, q, z, *, spin=0, **kwargs):
	return tend(M, q, z, spin=spin, **kwargs)-to(M, q, spin=spin, **kwargs)

def Ntot(M, q, *, spin=0, **kwargs):
	zlim11=zlim(M, q, spin=spin, **kwargs)
	zords=np.arange(0.01*zlim11, 0.99*zlim11, (zlim11-0.01*zlim11)/20.)
	absc=[ndot_tde(M)*ttot(M, q, zz, spin=spin, **kwargs)*dVc(zz) for zz in zords]
	return IUS(zords, absc).integral(zords[0], zords[-1])

def Ngrid(M, q, *, spin=0, **kwargs):
	zlim11=zlim(M, q, spin=spin, **kwargs)
	zords=np.arange(0.01*zlim11, 0.99*zlim11, (zlim11-0.01*zlim11)/20.)
	absc=[1.0e-5*cgs.year**-1/(1.0e6*cgs.pc)**3.*ttot(M, q, zz, spin=spin, **kwargs)*dVc(zz) for zz in zords]
	return IUS(zords, absc).integral(zords[0], zords[-1])

# mords=10.**np.arange(5, np.log10(Mhill(0)/cgs.M_sun), 0.05)*cgs.M_sun
# dat=[Ntot(mm, 1.2) for mm in mords]
# np.savetxt("spin_0_dat_new.csv", np.transpose([mords, dat]))

# mords=10.**np.arange(5, np.log10(Mhill(0.95)/cgs.M_sun), 0.05)*cgs.M_sun
# dat=[Ntot(mm, 1.2, spin=0.95) for mm in mords]
# np.savetxt("spin_0.95_dat_new.csv", np.transpose([mords, dat]))

# ##Had numerical difficulties with larger SMBHs
# mords=10.**np.arange(5, np.log10(Mhill(-0.95)/cgs.M_sun), 0.05)*cgs.M_sun
# dat=[Ntot(mm, 1.2, spin=-0.95) for mm in mords]
# np.savetxt("spin_-0.95_dat_new.csv", np.transpose([mords, dat]))


