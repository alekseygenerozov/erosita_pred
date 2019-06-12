import src.erosita_pred as eros
import numpy as np
import cgs_const as cgs

mords=10.**np.arange(5, np.log10(eros.Mhill(0)/cgs.M_sun), 0.05)*cgs.M_sun
dat=[eros.Ntot(mm, 1.2) for mm in mords]
np.savetxt("spin_0_dat_new.csv", np.transpose([mords, dat]))

mords=10.**np.arange(5, np.log10(eros.Mhill(0.95)/cgs.M_sun), 0.05)*cgs.M_sun
dat=[eros.Ntot(mm, 1.2, spin=0.95) for mm in mords]
np.savetxt("spin_0.95_dat_new.csv", np.transpose([mords, dat]))

# ##Had numerical difficulties with larger SMBHs
mords=10.**np.arange(5, np.log10(eros.Mhill(-0.95)/cgs.M_sun), 0.05)*cgs.M_sun
dat=[eros.Ntot(mm, 1.2, spin=-0.95) for mm in mords]
np.savetxt("spin_-0.95_dat_new.csv", np.transpose([mords, dat]))
