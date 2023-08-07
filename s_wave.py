import numpy as np
import math

from s_wave_base import *

def snell(theta1, V1, V2):
    '''Module function to implement snell's law to calculate a 
        reflected/refracted angle given the relevent wave velocities 
        and the angle of incidence.
        
        : param theta1 : Angle of incidence.
        : param V1     : Speed of incidence wave.
        : param V2     : Speed of reflection/refraction wave.
        :return        : Angle of reflection/refraction.
        '''
    
    if (np.sin(theta1)*V2)/V1 > 1:
        return np.pi/2
    else:
        return np.arcsin((math.sin(theta1)*V2)/V1)
  
class s_wave(s_wave_base):
  """ 
      A class to solve the seismic wave equation in an inhomogeneous environment
  """

  def __init__(self, Xmax, Zmax, Nx, Gamma=0.2, n_abs=30):
    """
      :  param Xmax       : X domain is [0, Xmax]
      :  param Zmax       : Z domain is [0, Zmax]
      :  param Nx         : Node number in X direction
      :  param Gamma      : Maximum damping value
      :  param n_abs      : Width of damping border (an integer)
    """
    super().__init__(Xmax, Zmax, Nx, Gamma, n_abs)
        

  def F(self, t, v_):
    """ Equation for seismic waves in inhomogeneous  media
      : param t : current time
      : param v_ : current function value (a vector of Shape [Nx, Nz, 4]).
      : return  : right hand side of the equation for v_.
    """
    # Initial exitation
    e = self.excite(t)
    if len(e) > 0:
      v_[self.excite_pos[0],self.excite_pos[1],:] = e

    eq = np.zeros([self.Nx, self.Nz, 4])

    # Scan the entire grid except edge nodes
    
    dv_xx = (v_[2:self.Nx, 1:self.Nz-1, 0] + v_[:self.Nx-2, 1:self.Nz-1, 0] - 2*v_[1:self.Nx-1, 1:self.Nz-1, 0])/self.dx**2
    dv_zz = (v_[1:self.Nx-1, 2:self.Nz, 0] + v_[1:self.Nx-1, :self.Nz-2, 0] - 2*v_[1:self.Nx-1, 1:self.Nz-1, 0])/self.dz**2
    dv_xz =(v_[2:self.Nx, 2:self.Nx, 0] + v_[:self.Nx-2, :self.Nz-2, 0] -v_[2:self.Nx, :self.Nz-2, 0] - v_[:self.Nx-2, 2:self.Nz, 0])/(4*self.dz*self.dx)
    
    dw_xx = (v_[2:self.Nx, 1:self.Nz-1, 1] + v_[0:self.Nx-2, 1:self.Nz-1, 1] - 2*v_[1:self.Nx-1, 1:self.Nz-1, 1])/self.dx**2
    dw_zz = (v_[1:self.Nx-1, 2:self.Nz, 1] + v_[1:self.Nx-1, :self.Nz-2, 1] - 2*v_[1:self.Nx-1, 1:self.Nz-1, 1])/self.dz**2
    dw_xz = (v_[2:self.Nx, 2:self.Nz, 1] + v_[:self.Nx-2, :self.Nz-2, 1]-v_[2:self.Nx, :self.Nz-2, 1] - v_[:self.Nx-2, 2:self.Nz, 1])/(4*self.dz*self.dx)
    
    eq[1:self.Nx-1, 1:self.Nz-1, 0] =  v_[1:self.Nx-1, 1:self.Nz-1, 2]
    eq[1:self.Nx-1, 1:self.Nz-1, 1] =  v_[1:self.Nx-1, 1:self.Nz-1, 3]
    eq[1:self.Nx-1, 1:self.Nz-1, 2] = ((self.lam[1:self.Nx-1, 1:self.Nz-1]+self.mu[1:self.Nx-1, 1:self.Nz-1])*(dv_xx+dw_xz)+self.mu[1:self.Nx-1, 1:self.Nz-1]*(dv_xx+dv_zz))/self.rho[1:self.Nx-1, 1:self.Nz-1]
    
    eq[1:self.Nx-1, 1:self.Nz-1, 3] = ((self.lam[1:self.Nx-1, 1:self.Nz-1]+self.mu[1:self.Nx-1, 1:self.Nz-1])*(dw_zz+dv_xz)+self.mu[1:self.Nx-1, 1:self.Nz-1]*(dw_xx+dw_zz) )/self.rho[1:self.Nx-1, 1:self.Nz-1]
    
    return(eq)


  def boundary(self, v_):
      
      
      v_[1:self.Nx-1, 0, 0] = v_[1:self.Nx-1, 1, 0] + (v_[2:self.Nx, 1, 1] - v_[:self.Nx-2, 1, 1]) * 0.5
      v_[1:self.Nx-1 ,0, 1] = v_[1:self.Nx-1, 1, 1] + 0.5*self.lam[1:self.Nx-1, 1]/\
          (self.lam[1:self.Nx-1, 1] + 2*self.mu[1:self.Nx-1, 1])*(v_[2:self.Nx, 1, 0] - v_[:self.Nx-2, 1, 0])

      # Absorption on the edges of the domain
      
      v_[0:self.Nx, 0:self.Nz, 2] *= 1-self.Gamma[0:self.Nx, 0:self.Nz]
      v_[:self.Nx,:self.Nz, 3] *= 1-self.Gamma[0:self.Nx, 0:self.Nz]
    
  def dist_offset(self, theta, path):
    '''Class function to calculate total horizontal offset of a path given an intial 
          angle of incidence theta, and the time in which the total path is travelled.
          
          : param theta : Inital angle of incidence.
          : param path  : list of the layer indices and wave types on the path [ [l1, W1], [l2, W2], ...]
          :return       : total horizontal offset, total time.
          '''
    
    # Initialise the thicknesses and velocities as empty numpy arrays.
    D = np.empty(len(path), dtype=float)
    V = np.empty(len(path), dtype=float)
    
    # Iterate through the class variable data to populate the arrays with the thicknesses and velocities.
    # V is decided depending on whether it is a P or S wave using an if statment.
    for i, p in enumerate(path):
        D[i] = self.data[p[0]][0]
        V[i] = self.data[p[0]][1] if p[1] == "P" else self.data[p[0]][2] 
    
    # Uses the numpy vectorize method to allows the function snell to be used element wise on an array.
    snell_vect = np.vectorize(snell)
    
    # Calculates array with the calculated angle of incidence for each layer.
    theta_k = snell_vect(theta, V, V[0])
    
    # Calculates array with horizontal displacement for each layer.
    x_k = np.tan(theta_k)*D
    
    # Calculates the total horizontal displacement and time
    x = x_k.sum()
    T = (D/(V*np.cos(theta_k))).sum()
    
    # Returns the x and T variables.
    return x, T

  def angle_and_delay(self, dx, path, err = 0.1):
      '''Class function to implement a bisection root finding algorithm to find the angle of incidence necessary
          to reach a specified horizontal displacement on a specific path.
          
          : param dx    : horizontal offset
          : param path  : list of the layer indices and wave types on the path [ [l1, W1], [l2, W2], ...]
          : param err   : maximum error in root approximation
          :return       : the incident angle, the true horizontal offset, and the time taken.
          '''

      def f(theta):
          '''Compute the difference between the actual 
              and required offset for a given theta'''
              
          return (self.dist_offset(theta,path))[0] - dx

      # Set initial values for angle bounds.
      a = 0
      b = np.pi/2

      # Check if there is a solution within the bounds.
      if f(a) * f(b) > 0:
          raise ValueError # bad input 

      # Iterate until a solution is found or maximum number of iterations is reached.
      nmax = 10000 # maximum number of iterations
      
      for n in range(nmax): 
          
          # Compute midpoint of current angle bounds.
          p = 0.5*(a + b)

          # Check if solution has been found
          if abs(f(p)) < err:
              break

          # Update angle bounds based on value of f at midpoint.
          if f(a) * f(p) > 0.0:
              a = p 
          else:
              b = p 

      # Compute distance offset for final angle.
      x_t = self.dist_offset(p,path)

      # Return final angle and distance offset.
      return p, x_t[0], x_t[1]


  def eval_detector_diff(self, d1, d2, dn, data_type):
      '''Class function to computes the difference based on a specified data_type between 
          two detector signals d1 and d2, at detector index dn.
          
          : param d1            : signal 1
          : param d2            : signal 2
          : param dn            : index of the detector 
          : param data_type     : type of difference
          :return               : list of type [[t0, x0], [t1, x1], .. [tn, xn]]
          '''
       
      def v(i):
            # v displacement difference at time ti.
            return [d1[i][0],(d1[i][1][dn] - d2[i][1][dn])]
      def w(i):
            # w displacement difference at time ti.
            return [d1[i][0],(d1[i][2][dn] - d2[i][2][dn])]
      def Mod(i):
            # Amplitude of the displacement difference at time ti.
            return [ d1[i][0],np.sqrt((d2[i][1][dn] - d1[i][1][dn])**2 + (d2[i][2][dn] - d1[i][2][dn])**2)]
      def phase(i):
            # Angle in the plane of the displacement difference at time ti.
            return [ d1[i][0],np.arctan2(d2[i][2][dn] - d1[i][2][dn], d2[i][1][dn] - d1[i][1][dn])]
    
        
      # Dictionary assigning each data_type to the relevant function.
      data_type_fcts = {
            "v": v,
            "w": w,
            "Mod": Mod,
            "Phase": phase
                }
        
      # Uses a list comprehension to populate x with [ti,xi].
      x = [data_type_fcts[data_type](i) for i in range(len(d1))]
        
      # Return the list of displacements x.
      return x
    

