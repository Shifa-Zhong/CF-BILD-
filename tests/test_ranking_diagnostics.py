'''Check positive rescaling and the truncated upper-tail calculation.'''
import sys
from pathlib import Path
import torch
import numpy as np
from scipy.special import log_ndtr
from scipy.stats import truncnorm
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cf_bild.acquisition import cell_width_expectation,joint_feasibility_probability

def main():
    mu=np.array([[.2,1.,3.],[-.1,-2.,4.],[.5,2.,1.]])
    sd=np.array([[.1,.5,1.],[.2,1.,.6],[.3,.4,.5]])
    ref=np.array([0.,-5.,0.]);th=np.array([.4,1.5,3.5]);w=np.array([100.,.5,2.])
    def widths(m,s,r):return cell_width_expectation(m,s,np.broadcast_to(r,m.shape),np.full_like(m,np.inf))
    np.testing.assert_allclose(widths(mu*w,sd*w,ref*w),widths(mu,sd,ref)*w,rtol=1e-12)
    np.testing.assert_allclose(joint_feasibility_probability(mu*w,sd*w,th*w),joint_feasibility_probability(mu,sd,th),rtol=1e-12)
    tail=np.exp(log_ndtr((mu[:,0]-1)/sd[:,0])-log_ndtr(mu[:,0]/sd[:,0]))
    np.testing.assert_allclose(tail,truncnorm.sf(1,-mu[:,0]/sd[:,0],np.inf,loc=mu[:,0],scale=sd[:,0]),rtol=1e-10,atol=1e-14)
    print('PASS positive-scale EI/probability transformation and upper-support formula')

if __name__=='__main__':main()
