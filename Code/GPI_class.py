import numpy as np
from tqdm import tqdm
from main import lissajous
from scipy.stats import multivariate_normal


class GPI():

    def __init__(self,traj,obs):

        self.ref = traj
        self.obstacles = obs
        self.time_step = 0.5
        self.num_x  = 8
        self.num_y = 8
        self.num_theta = 8
        self.nt = 100
        self.num_vel = 10
        self.num_omega = 10
        self.gamma = 0.95
        self.max_iters = 1000
        self.T = self.nt * self.num_x * self.num_y * self.num_theta
        self.xlim = np.array([-3,3])
        self.ylim = np.array([-3,3])
        self.theta_lim = np.array([-np.pi,np.pi])
        self.vel_lim = np.array([0,1])
        self.omega_lim = np.array([-1,1])
        self.error_state_space = self.error_state_space_matrix()
        self.control_space = self.control_space_matrix()
        self.stage_cost_matrix = self.generate_stage_cost()
        self.transition_state_idx, self.transition_probs = self.generate_transition_prob_matrix()
        self.optimal_policy = self.value_iteration()


    def error_state_space_matrix(self):

        e_x = np.linspace(-3,3,self.num_x)

        e_y = np.linspace(-3,3,self.num_y)

        e_theta = np.linspace(-np.pi,np.pi,self.num_theta)
   
        error_state_space = np.zeros((4,self.T),dtype=np.float16)
        k=0

        ## generating state space

        print("Starting to generate error state space....")

        for t in tqdm(range(self.nt)):
    
            for i in range(self.num_x):
                for j in range(self.num_y):
                    curr_ref = self.ref(t)
                    x,y = np.array([e_x[i],e_y[j]]) + curr_ref[:2]
                    if x <= 3 and x >= -3 and y <= 3 and y >= -3:  ## state space boundary check
                        if (x - self.obstacles[0,0])**2 + (y - self.obstacles[0,1])**2 >= (self.obstacles[0,2])**2:
                            if (x - self.obstacles[1,0])**2 + (y - self.obstacles[1,1])**2 >= self.obstacles[1,2]**2:
                                for m in range(self.num_theta):
                                    error_state_space[:,k] = np.array([t,e_x[i],e_y[j],e_theta[m]], dtype= np.float16)
                                    k = k+1

        print("State Space Generated..")
        np.save('error_state_space.npy',error_state_space)
        return error_state_space
    
    

    def control_space_matrix(self):
        vel = np.linspace(self.vel_lim[0],self.vel_lim[1],self.num_vel)
        omega = np.linspace(self.omega_lim[0],self.omega_lim[1],self.num_omega)
        control_space = np.array(np.meshgrid(vel,omega)).T.reshape(-1, 2)
        u = np.zeros((2,self.num_omega * self.num_vel))
        s = 0
        print("Starting to generate control space....")
        for i in tqdm(range(control_space.shape[0])):
            u[:,s] = control_space[i]
            s=s+1
        
        print("Control space generated..")
        np.save('control_space.npy',u)
        return u
    

    
    def generate_stage_cost(self):

        Q = 1*np.eye(2)
        R = 1*np.eye(2)
        q = 1
        L = np.zeros((self.T,self.control_space.shape[1]))

        print("Starting to generate stage cost....")

        for i in tqdm(range(self.T)):
            for j in range(self.control_space.shape[1]):
                L[i,j] = (self.error_state_space[1:3,i].T @ Q) @ self.error_state_space[1:3,i] + q * (1-np.cos(self.error_state_space[3,i]))**2 + (self.control_space[:,j].T @ R) @ self.control_space[:,j]
        print("Stage cost Matrix generated..")
        np.save('stage_cost_matrix.npy',L)
        return L
    

    def motion_model(self,x,u,ref_pos_curr, ref_pos_next, ref_theta_curr, ref_theta_next):
        x[0] = x[0] + u[0] * self.time_step * np.cos(x[2] + ref_theta_curr) + ref_pos_curr[0] - ref_pos_next[0] #+ np.random.normal(0,0.04)
        x[1] = x[1] + u[0] * self.time_step * np.sin(x[2] + ref_theta_curr) + ref_pos_curr[1] - ref_pos_next[1] #+ np.random.normal(0,0.04)
        x[2] = x[2] + u[1] * self.time_step + ref_theta_curr - ref_theta_next #+ np.random.normal(0,0.004)
        return x

       
    def generate_transition_prob_matrix(self):

        transition_probabilities_matrix = np.zeros((self.T,self.control_space.shape[1],6))
        transition_states_idx = np.zeros((self.T,self.control_space.shape[1],6))
        cov = np.diag(np.array([0.04,0.04,0.004]))

        print("Starting to generate transition probability matrix....")

        for i in tqdm(range(self.T)):
            curr_error_state = self.error_state_space[:,i]
            curr_time = curr_error_state[0]
            curr_ref = self.ref(curr_time)
            ref_pos_curr = curr_ref[:2]
            next_ref = self.ref(curr_time + 1)
            ref_pos_next = next_ref[:2]
            ref_theta_curr = curr_ref[2]
            ref_theta_next = next_ref[2]

            if curr_time != 99:
                next_time_error_states_idx = np.where(self.error_state_space[0,:] == curr_time + 1)[0]
            else:
                next_time_error_states_idx = np.where(self.error_state_space[0,:] == 0)[0]


            for j in range(self.control_space.shape[1]):
                temp = np.copy(curr_error_state[1:])
                next_error_state = self.motion_model(temp, self.control_space[:,j], ref_pos_curr, ref_pos_next,ref_theta_curr,ref_theta_next)
                next_error_state[2] = np.fmod(next_error_state[2] + np.pi, 2 * np.pi) - np.pi
                if next_error_state[2] > np.pi:
                    next_error_state[2] = next_error_state[2] - 2 * np.pi
                if next_error_state[2] < -np.pi:
                    next_error_state[2] = next_error_state[2] + 2 * np.pi

                mean = next_error_state
                trans_prob = np.zeros(self.T)
                trans_prob[next_time_error_states_idx] = multivariate_normal.pdf(self.error_state_space[1:,next_time_error_states_idx].T,mean,cov)
                best_6_idx = np.argpartition(trans_prob, -6)[-6:]
                best_6_probs = trans_prob[best_6_idx]/trans_prob[best_6_idx].sum()
                transition_probabilities_matrix[i,j,:] = best_6_probs
                transition_states_idx[i,j,:] = best_6_idx
        
        np.save('transition_states_idx.npy',transition_states_idx)
        np.save('transition_probabilities_matrix.npy',transition_probabilities_matrix)

        return transition_states_idx, transition_probabilities_matrix
    

    def value_iteration(self):

        P = np.load('transition_probabilities_matrix.npy')
        transition_idx = np.load('transition_states_idx.npy')

        Val_func = np.zeros(self.T)
        optimal_policy = np.zeros(self.T)

        L = np.load('stage_cost_matrix.npy')

        print("Starting Value Iteration...")

        for iter in tqdm(range(self.max_iters)):
            V_old = np.copy(Val_func)
            exp_val_cost = np.sum(P * Val_func[np.int64(transition_idx)],axis = 2)
            Q = L + self.gamma * exp_val_cost
            V_new = np.min(Q,axis =1)
            optimal_policy = np.argmin(Q,axis=1)
            Val_func = np.copy(V_new)
            diff = np.linalg.norm(V_new - V_old)
            print("Iter: {}, Difference: {}".format(iter,diff))
            if np.linalg.norm(V_new - V_old) <= 0.001:
                break
    
        print("Value Iteration completed")
        np.save('optimal_policy.npy', optimal_policy)
        return optimal_policy




if __name__ == '__main__':
    traj = lissajous
    obstacles = np.array([[-2,-2,0.5], [1,2,0.5]])
    gpi_solve = GPI(traj,obstacles)


