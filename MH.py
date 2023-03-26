import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
class MetroHast():
    def __init__(self, post, sig):
        self.p = post
        self.sig = sig

    @staticmethod
    def PDFgauss(loc,scale,x):
        pdf = 1/np.sqrt(2*np.pi*scale*scale)*np.exp(-(x-loc)**2/2/scale/scale)
        return pdf
        
    def proposal(self, x0):
        x1 = np.random.normal(x0,self.sig)
        q_x0x1 = self.PDFgauss(loc=x1,scale=self.sig,x=x0)
        q_x1x0 = self.PDFgauss(loc=x0,scale=self.sig,x=x1)
        return x1, q_x0x1, q_x1x0

    def run(self,N,xinit=0):
        samples = []
        # initialize
        x0 = xinit
        for i in range(N):
            u  = np.random.uniform(0,1)
            # propose the move
            x1, q_x0x1, q_x1x0 = self.proposal(x0)
            # acceptance / rejection
            AR = np.min([1.0,self.p(x1)*q_x0x1 / self.p(x0) / q_x1x0])
            if u <= AR:
                samples.append(x1)
                x0 = x1
            else:
                samples.append(x0)
                x0 = x0
                
        return np.array(samples)