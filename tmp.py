import numpy as np
import os
import sys

a1=sys.argv[1]
angs=np.linspace(0.0, 180.0, 20.)
angs=angs[1:-1]
e1s=np.arange(0, 1.01, 0.1)
e1s[0]=0.01
e1s[-1]=0.99

tags=['0.05_a', '0.05_b', '0.1_a', '0.1_b']

pre=os.path.join(os.path.dirname(__file__))
for e1 in e1s:
	for ang in angs:
		for tag in tags:
			temp=open('template_2orbit.sh').read()
			temp=temp.replace('xx1', '{0}'.format(e1))
			temp=temp.replace('xx', '--etest {0} --ein {0}'.format(e1))
			temp=temp.replace('ww1', '{0}'.format(a1))
			temp=temp.replace('ww', '--atest {0}'.format(a1))
			temp=temp.replace('yy1', '{0:.1f}'.format(ang))
			temp=temp.replace('yy', '--pomega {0:.1f}'.format(ang))
			temp=temp.replace('zz1', '{0}'.format(tag))
			temp=temp.replace('zz', ' --dtag {0}'.format(tag))
			temp=temp.replace('ppp', pre)
			f=open('e{0}_ang{1:.1f}_dt{2}.sh'.format(e1, ang, tag), 'w')
			f.write(temp)
			f.close()