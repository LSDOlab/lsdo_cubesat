"""
    Rotation matrix from Earth-centered inertial frame to
    Earth-fixed frame over time
    
    calculate quaternion then rotation matrix
"""
import numpy as np
from openmdao.api import ExplicitComponent

class RotMtxECIEFComp(ExplicitComponent):

    fact = np.pi / 3600.0 / 24.0
    
    def initialize(self):
        self.options.declare('num_times',types=int)
        
    def setup(self):
        num_times = self.options['num_times']
        
        self.add_input('t', np.zeros(num_times), units='s',
                        desc = 'Time')
                        
        self.add_output('Rot_ECI_EF',np.zeros((3,3,num_times)), units=None,
                        desc = 'Rotation matrix from Earth-centered inertial frame'
                        'to Earth-fixed frame over time')
    
    def compute(self,inputs,outputs):
        num_times = self.options['num_times']
        
        t = inputs['t']
        Rot_ECI_EF = outputs['Rot_ECI_EF']

#        q_E = np.zeros((4,num_times))
#        q_E[0,:] = np.cos(theta)
#        q_E[3,:] = np.sin(theta)
        
        A = np.zeros((4,3))
        B = np.zeros((4,3))
        
        for i in range(0,num_times):
        
#            A[0, :] = ( q_E[0, i], -q_E[3, i],  q_E[2, i])
#            A[1, :] = ( q_E[3, i],  q_E[0, i], -q_E[1, i])
#            A[2, :] = (-q_E[2, i],  q_E[1, i],  q_E[0, i])
#            A[3, :] = ( q_E[1, i],  q_E[2, i],  q_E[3, i])
#
#            B[0, :] = ( q_E[0, i],  q_E[3, i], -q_E[2, i])
#            B[1, :] = (-q_E[3, i],  q_E[0, i],  q_E[1, i])
#            B[2, :] = ( q_E[2, i], -q_E[1, i],  q_E[0, i])
#            B[3, :] = ( q_E[1, i],  q_E[2, i],  q_E[3, i])
            A[0,:] = ( np.cos(t[i]), np.sin(t[i]), 0)
            A[1,:] = (-np.sin(t[i]), np.cos(t[i]), 0)
            A[2,:] = (0, 0,  np.cos(t[i]))
            A[3,:] = (0, 0, -np.sin(t[i]))
            A = self.fact * A

            
            B[0,:] = ( np.cos(t[i]),-np.sin(t[i]), 0)
            B[1,:] = ( np.sin(t[i]), np.cos(t[i]), 0)
            B[2,:] = (0, 0,  np.cos(t[i]))
            B[3,:] = (0, 0, -np.sin(t[i]))
            B = self.fact * B


            Rot_ECI_EF[:, :, i] = np.dot(A.T, B)
            
    def compute_partials(self,inputs,outputs):
        num_times = self.options['num_times']
        
        t = inputs['t']

        theta = self.fact * t
        
        A = np.zeros((4,3))
        B = np.zeros((4,3))
        
        dA_dt = np.zeros((4,3))
        dB_dt = np.zeros((4,3))
        
        self.J = np.zeros((num_times,3,3))
        
        for i in range(0, num_times):
        
            A[0,:] = ( np.cos(theta[i]), np.sin(theta[i]), 0)
            A[1,:] = (-np.sin(theta[i]), np.cos(theta[i]), 0)
            A[2,:] = (0, 0,  np.cos(theta[i]))
            A[3,:] = (0, 0, -np.sin(theta[i]))
            
            B[0,:] = ( np.cos(theta[i]),-np.sin(theta[i]), 0)
            B[1,:] = ( np.sin(theta[i]), np.cos(theta[i]), 0)
            B[2,:] = (0, 0,  np.cos(theta[i]))
            B[3,:] = (0, 0, -np.sin(theta[i]))
            
            dA_dt[0,:] = (-np.sin(t[i]), -np.cos(t[i]), 0)
            dA_dt[1,:] = ( np.cos(t[i]), -np.sin(t[i]), 0)
            dA_dt[2,:] = ( 0, 0, -np.sin(t[i]))
            dA_dt[3,:] = ( 0, 0, -np.cos(t[i]))
            dA_dt = dA_dt * self.fact
            
            dB_dt[0,:] = (-np.sin(t[i]), -np.cos(t[i]), 0)
            dB_dt[1,:] = ( np.cos(t[i]), -np.sin(t[i]), 0)
            dB_dt[2,:] = ( 0, 0, -np.sin(t[i]))
            dB_dt[3,:] = ( 0, 0, -np.cos(t[i]))
            dB_dt = dB_dt * self.fact
            
            self.J[i,:,:] = np.dot(dA_dt.T, B) + np.dot(A.T, dB_dt)

if __name__ == '__main__':
    from openmdao.api import Problem, IndepVarComp


    num_times = 3

    prob = Problem()

    comp = IndepVarComp()
    comp.add_output('t', val=np.random.random(num_times))
    prob.model.add_subsystem('inputs_comp', comp, promotes=['*'])

    comp = RotMtxECIEFComp(
        num_times=num_times,
    )
    prob.model.add_subsystem('comp', comp, promotes=['*'])

    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    prob.check_partials(compact_print=True)
