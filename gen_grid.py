import numpy as np
import os
import sys
import src.erosita_pred as eros
import cgs_const as cgs

pre=os.path.join(os.path.dirname(__file__))
a1=np.arange(-0.95, 0.96, 0.1)
a1=np.around(a1,2)
for aa in a1:
	Mgrid=np.arange(5, np.log10(eros.Mhill(aa)/cgs.M_sun), 0.05)
	Mgrid=np.around(Mgrid,2)
	for MM in Mgrid:
		temp=open('template.sh').read()
		# temp=temp.replace('xx1', '{0}'.format(e1))
		temp=temp.replace('xx', '{0}'.format(MM))
		# temp=temp.replace('ww1', '{0}'.format(a1))
		temp=temp.replace('ww', '{0}'.format(aa))
		f=open('M{0}_a{1}.sh'.format(MM, aa), 'w')
		f.write(temp)
		f.close()