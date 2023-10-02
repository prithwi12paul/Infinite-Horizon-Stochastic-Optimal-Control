from casadi import *
import numpy as np


def CEC_controller(curr_iter, traj, curr_error, obstacles):

    opti = Opti()

    # define time horizon
    # T = 15
    T = 30  # time horizon
    e = opti.variable(3, T+1)
    u = opti.variable(2, T)

    e[0,0] = curr_error[0]
    e[1,0] = curr_error[1]
    e[2,0] = curr_error[2]

    # cost = 0
  
    gamma = 1
    Q = 30*np.eye(2)
    R = 30*np.eye(2)
    q = 20

    dt = 0.5

    ref_traj_curr = traj(curr_iter)
    # ref_traj_theta = ref_traj[-1,curr_iter:]

    # t=0
    # opt_control = []

    cost = 0

    for t in range(T):
        cost += (gamma ** (t)) * stage_cost(e[:,t], Q, R, q, u[:,t])
    
    terminal_cost = e[:,-1].T @ e[:,-1]
    cost = cost + terminal_cost
    opti.minimize(cost)


    ## constraints on the next state and the free space environment


    for t in range(T):

        cur_ref_traj = traj(curr_iter)
        next_ref_traj = traj(curr_iter + 1)
        ref_pos_curr = cur_ref_traj[:2]
        ref_pos_next = next_ref_traj[:2]
        ref_theta_curr = cur_ref_traj[2]
        ref_theta_next = next_ref_traj[2]

        e_next = motion_model(e[:,t], u[:,t], ref_pos_curr, ref_pos_next, ref_theta_curr, ref_theta_next)

        # e_next[2] = fmod(e_next[2],(2 * np.pi))
        # if e_next[2] > np.pi:
        #     e_next[2] -= 2 * np.pi

        # Using the fmod() function
        e_next[2] = fmod(e_next[2] + np.pi, 2 * np.pi) - np.pi
        e_next[2] = if_else(e_next[2] > np.pi, e_next[2] - 2 * np.pi, e_next[2])
        e_next[2] = if_else(e_next[2] < -np.pi, e_next[2] + 2 * np.pi, e_next[2])
        # e_next[2] = fmod(e_next[2]+np.pi, 2*np.pi)
        # e_next[2] = e_next[2] - np.pi 

        opti.subject_to(e[0,t+1] == e_next[0])
        opti.subject_to(e[1,t+1] == e_next[1])
        opti.subject_to(e[2,t+1] == e_next[2])
        # opti.subject_to(opti.bounded(0, u[0, t], 1))
        # opti.subject_to(opti.bounded(-1, u[1, t], 1))

        for i in range(obstacles.shape[0]):

            x_obs = obstacles[i,0]
            y_obs = obstacles[i,1]
            r_obs = obstacles[i,2]

            next_pos_x = e_next[0] + ref_pos_next[0]
            # curr_pos_x = e[0,t] + ref_pos_curr[0]
            next_pos_y = e_next[1] + ref_pos_next[1]
            # opti.subject_to(np.linalg.norm(np.array(next_pos_x,next_pos_y)-np.array(x_obs,y_obs)) > r_obs)
            opti.subject_to((next_pos_x - x_obs)**2 + (next_pos_y - y_obs)**2 > r_obs**2)
            
        opti.subject_to(opti.bounded(-3, next_pos_x, 3))
        opti.subject_to(opti.bounded(-3, next_pos_y, 3))
        # opti.subject_to(opti.bounded(-np.pi, e[2,t], np.pi))

        curr_iter += 1

    opti.subject_to(opti.bounded(0, u[0, :], 1))
    opti.subject_to(opti.bounded(-1, u[1, :], 1))

    opti.set_initial(e[0,0],curr_error[0])
    opti.set_initial(e[1,0],curr_error[1])
    opti.set_initial(e[2,0],curr_error[2])

    # opts = {'ipopt.print_level': 0, 'print_time': 0}
    p_opts = {"expand": True, 'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
    # s_opts = {"max_iter": 1000}
    opti.solver('ipopt', p_opts)
    sol = opti.solve()

    return sol.value(u[:,0])



def motion_model(x,u,ref_pos_curr, ref_pos_next, ref_theta_curr, ref_theta_next):
    dt = 0.5
    x[0] = x[0] + u[0]*dt*cos(x[2] + ref_theta_curr) + ref_pos_curr[0] - ref_pos_next[0]
    x[1] = x[1] + u[0]*dt*sin(x[2] + ref_theta_curr) + ref_pos_curr[1] - ref_pos_next[1]
    x[2] = x[2] + u[1]*dt + ref_theta_curr - ref_theta_next
    return x

def stage_cost(x,Q,R,q,u):
    # l = (mtimes((mtimes(x[:2].T,Q)),x[:2]) + q*(1-cos(x[2]))**2 + mtimes(mtimes(u.T,R),u))
    l = (x[:2].T @ Q) @ x[:2] + q*(1-cos(x[2]))**2 + (u.T @ R) @ u
    return l
