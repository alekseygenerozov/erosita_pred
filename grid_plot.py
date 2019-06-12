import src.erosita_pred as eros
import numpy as np
import cgs_const as cgs

import sys

Mbh=10.0**float(sys.argv[1])*cgs.M_sun
Ndet=eros.Ngrid(Mbh, 1.2, spin=float(sys.argv[2]))
f=open('./Ndet_{0}_{1}'.format(sys.argv[1], sys.argv[2]), 'w')
f.write('{0:e}'.format(Ndet))
f.close()