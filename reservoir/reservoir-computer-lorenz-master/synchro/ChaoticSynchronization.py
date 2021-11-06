#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Sandbox for exploring the phenomenon of chaotic synchronization in the
Lorenz system

Description of the Lorenz equations:
These are highly simplified equations that describe aspects
of fluid motion in a shallow layer.

From wiki:
More specifically, the Lorenz equations are derived from the Oberbeck-Boussinesq
approximation to the equations describing fluid circulation in a shallow layer
of fluid, heated uniformly from below and cooled uniformly from above. This fluid
circulation is known as Rayleigh-Bénard convection. The fluid is assumed to
circulate in two dimensions (vertical and horizontal) with periodic rectangular
boundary conditions.
The Lorenz equations also arise in simplified models for lasers, dynamos,
thermosyphons, brushless DC motors, electric circuits, chemical reactions and
forward osmosis.

Lorenz attractor equations:
dx/dt = σ(y−x)dx/dt     = σ(y−x)
dy/dt = x(ρ−z) − ydy/dt = x(ρ−z)−y
dz/dt                   = xy − βz

x - proportional to the intensity of convection motion.
y - proportional to the temperature difference between the ascending and
    descending currents.
z - proportional to the distortion of the vertical temperature profile
    from linearity.

"""

from logger import log
import sys
import numpy as np
import multiprocessing as mp
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def lorenz(x, y, z, sigma=10, rho=28, beta=2.667):
    """
    Calculates the dt in the lorenz system
    """
    x_dot = sigma * (y - x)
    y_dot = x * (rho - z) - y
    z_dot = (x * y) - (beta * z)
    return x_dot, y_dot, z_dot

def initLorenzRandom(x_mean=0.0, y_mean=0.0, z_mean=0.0,
                x_variance = 20.0, y_variance = 20.0, z_variance = 20.0):
    """
    Used to randomly assign initial values to the attractor
    """
    # Set to random values
    x_init = x_mean + random.uniform(-x_variance, x_variance)
    y_init = y_mean + random.uniform(-y_variance, y_variance)
    z_init = z_mean + random.uniform(-z_variance, z_variance)
    return [x_init,y_init,z_init]


def rossler(x, y, z, a=0.2, b=0.2, c=5.7):
    """
    Calculates the dt in the Rossler system
    """
    x_dot = (-1.0 * y) - z
    y_dot = x + (a * y)
    z_dot = b + (z * (x - c))
    return x_dot, y_dot, z_dot

def initRosslerRandom(x_mean=0.0, y_mean=0.0, z_mean=5.0,
                x_variance = 10.0, y_variance = 10.0, z_variance = 10.0):
    """
    Used to randomly assign initial values to the attractor
    """
    # Set to random values
    x_init = x_mean + random.uniform(-x_variance, x_variance)
    y_init = y_mean + random.uniform(-y_variance, y_variance)
    z_init = z_mean + random.uniform(-z_variance, z_variance)
    return [x_init,y_init,z_init]


def chua(x, y, z, a=15.6, b=32.0, c=0.01):
    """
    Calculates the dt in Chua's circuit system
    """
    x_dot = a * (y - chuaFunc(x))
    y_dot = x - y + z
    z_dot = ((-1.0 * b) * y) - (c * z)
    return x_dot, y_dot, z_dot

def chuaFunc(x, m0=-1.14285714285714, m1=-0.71428571428571):
    if (x <= -1): g = (m1 * x) + m1 - m0
    if (x > -1) and (x < 1): g = m0 * x
    if (x >= 1): g = (m1 * x) + m0 - m1
    return x + g

def initChuaRandom(x_mean=0.0, y_mean=0.0, z_mean=0.0,
                x_variance = 2.0, y_variance = 0.5, z_variance = 2.0):
    """
    Used to randomly assign initial values to the attractor
    """
    # Set to random values
    x_init = x_mean + random.uniform(-x_variance, x_variance)
    y_init = y_mean + random.uniform(-y_variance, y_variance)
    z_init = z_mean + random.uniform(-z_variance, z_variance)
    return [x_init,y_init,z_init]


def plotSynch(steps=15000, dt=0.001, kx=0.01, ky=0.01, kz=0.00,
              init_timesteps=1000, synch_max=3000,
              sigma_o=10, rho_o=28, beta_o=2.667,
              sigma_m=10, rho_m=28, beta_m=2.667,
              obs_error=0.0, attractor='lorenz'):
    """
    Creates plots the synchronization process
    steps - number of timesteps to integrate
    dt - the timestep size
    k<x,y,z> - coupling stregth in the dimensions
    synch_max - timestep to stop coupling
    sigma/rho/beta - set the parameters of the lorenz systems
    obs_error - add observational error (fraction)
    """
    axis = np.linspace(0,steps,steps)

    # Need one more for the initial values
    xo_vals = np.empty(steps)
    yo_vals = np.empty(steps)
    zo_vals = np.empty(steps)
    xm_vals = np.empty(steps)
    ym_vals = np.empty(steps)
    zm_vals = np.empty(steps)
    if (attractor == 'lorenz'):
        xo_vals[0], yo_vals[0], zo_vals[0] = initLorenzRandom()
        xm_vals[0], ym_vals[0], zm_vals[0] = initLorenzRandom()
    elif (attractor == 'rossler'):
        xo_vals[0], yo_vals[0], zo_vals[0] = initRosslerRandom()
        xm_vals[0], ym_vals[0], zm_vals[0] = initRosslerRandom()
    elif (attractor == 'chua'):
        xo_vals[0], yo_vals[0], zo_vals[0] = initChuaRandom()
        xm_vals[0], ym_vals[0], zm_vals[0] = initChuaRandom()
    else:
        log.out.error("Attractor: " + attractor + " not defined.")
        sys.exit(-1)
    # Stepping through "time".
    for i in range(0,steps-1):
        # Calculate the derivatives of the X, Y, Z state

        if (attractor == 'lorenz'):
            xo_dot, yo_dot, zo_dot = lorenz(xo_vals[i], yo_vals[i], zo_vals[i],
                                            sigma=sigma_o, rho=rho_o, beta=beta_o)
            xm_dot, ym_dot, zm_dot = lorenz(xm_vals[i], ym_vals[i], zm_vals[i],
                                            sigma=sigma_m, rho=rho_m, beta=beta_m)
        elif (attractor == 'rossler'):
            xo_dot, yo_dot, zo_dot = rossler(xo_vals[i], yo_vals[i], zo_vals[i])
            xm_dot, ym_dot, zm_dot = rossler(xm_vals[i], ym_vals[i], zm_vals[i])
        elif (attractor == 'chua'):
            xo_dot, yo_dot, zo_dot = chua(xo_vals[i], yo_vals[i], zo_vals[i])
            xm_dot, ym_dot, zm_dot = chua(xm_vals[i], ym_vals[i], zm_vals[i])

        # Apply coupling
        if i < synch_max and i > init_timesteps:
            ex = (xo_vals[i] - xm_vals[i])
            ey = (yo_vals[i] - ym_vals[i])
            ez = (zo_vals[i] - zm_vals[i])
            if (obs_error == 0.0):
                ux = kx * ex
                uy = ky * ey
                uz = kz * ez
            else:
                ux = kx * (ex + obs_error*random.uniform(-xo_vals[i],xo_vals[i]))
                uy = ky * (ey + obs_error*random.uniform(-yo_vals[i],yo_vals[i]))
                uz = kz * (ez + obs_error*random.uniform(-zo_vals[i],zo_vals[i]))
        else:
            ux = 0
            uy = 0
            uz = 0

        # Propagate the observed system
        xo_vals[i + 1] = xo_vals[i] + (xo_dot * dt)
        yo_vals[i + 1] = yo_vals[i] + (yo_dot * dt)
        zo_vals[i + 1] = zo_vals[i] + (zo_dot * dt)
        # Propagate the modeled system
        xm_vals[i + 1] = xm_vals[i] + (xm_dot * dt) + ux
        ym_vals[i + 1] = ym_vals[i] + (ym_dot * dt) + uy
        zm_vals[i + 1] = zm_vals[i] + (zm_dot * dt) + uz

    #### 3D plots!
#    fig_m = plt.figure()
#    fig_m.canvas.set_window_title('3D View (model)')
#    ax_m = fig_m.gca(projection='3d')
#    ax_m.plot(xm_vals, ym_vals, zm_vals)
#    ax_m.set_xlabel("X Axis")
#    ax_m.set_ylabel("Y Axis")
#    ax_m.set_zlabel("Z Axis")
#    ax_m.set_title("Lorenz Attractor Modeled")
#
#    fig_o = plt.figure()
#    fig_o.canvas.set_window_title('3D View (obs)')
#    ax_o = fig_o.gca(projection='3d')
#    ax_o.plot(xo_vals, yo_vals, zo_vals)
#    ax_o.set_xlabel("X Axis")
#    ax_o.set_ylabel("Y Axis")
#    ax_o.set_zlabel("Z Axis")
#    ax_o.set_title("Lorenz Attractor Observed")

    fig3d = plt.figure()
    fig3d.canvas.set_window_title('3D View (model)')
    ax3d = fig3d.gca(projection='3d')
    ax3d.plot(xo_vals, yo_vals, zo_vals, color='g')
    ax3d.plot(xm_vals, ym_vals, zm_vals, color='r')
    ax3d.set_xlabel("X Axis")
    ax3d.set_ylabel("Y Axis")
    ax3d.set_zlabel("Z Axis")
    if (attractor == 'lorenz'):
        ax3d.set_title("Lorenz Attractors Compare")
    elif (attractor == 'rossler'):
        ax3d.set_title("Rössler Attractors Compare")
    elif (attractor == 'chua'):
        ax3d.set_title("Chua's circuits Compare")

#    ### 2D plots of the components of the systems
#    fig_obs = plt.figure()
#    fig_obs = plt.gcf()
#    fig_obs.canvas.set_window_title('Observed System')
#    plt.plot(axis,xo_vals,'r') # plotting t,a separately
#    plt.plot(axis,yo_vals,'g') # plotting t,b separately
#    plt.plot(axis,zo_vals,'b') # plotting t,c separately
#    fig_mod = plt.figure()
#    fig_mod = plt.gcf()
#    fig_mod.canvas.set_window_title('Modeled System')
#    plt.plot(axis,xm_vals,'r') # plotting t,a separately
#    plt.plot(axis,ym_vals,'g') # plotting t,b separately
#    plt.plot(axis,zm_vals,'b') # plotting t,c separately

    #### Plot the difference in the coordinates
    x_diff = xm_vals - xo_vals
    y_diff = ym_vals - yo_vals
    z_diff = zm_vals - zo_vals
    fig_xdiff = plt.figure()
    fig_xdiff = plt.gcf()
    fig_xdiff.canvas.set_window_title('Error in coordinates')
    plt.plot(axis,x_diff,'r') # plotting t,a separately
    plt.plot(axis,y_diff,'g') # plotting t,a separately
    plt.plot(axis,z_diff,'b') # plotting t,a separately
    plt.axvline(init_timesteps, color='k')
    plt.axvline(synch_max, color='k')

    plt.show()



def measureSynchSpace(num_tests=200, kxi=0.011, kyi=0.011, kzi=0.000,
                      kx_dt=0.000038, ky_dt=0.000038, kz_dt=0.0000,
                      dt=0.005, synch_threshold=0.001, max_steps=120000,
                      num_to_average=100):
    """
    This explores coupling space, starts all coupling at zero and
    increases the coupling in each dimension be k*_dt, measures the amount
    of timesteps until coupling is achieved, defined when error in dimensions
    is less than the synch threshold

    k<x,y,z>i  - initial values
    k<x,y,z>_dt  - coupling stregth step increase size
    dt - the timestep size
    max_steps - number of timesteps to give up if no synch has been found
    synch_threshold - error to call it synched at
    num_to_average - average this many tests at each strength
    """
    # Need one more for the initial values
    xo_vals = np.empty(max_steps)
    yo_vals = np.empty(max_steps)
    zo_vals = np.empty(max_steps)
    xm_vals = np.empty(max_steps)
    ym_vals = np.empty(max_steps)
    zm_vals = np.empty(max_steps)
    xo_vals[0], yo_vals[0], zo_vals[0] = initLorenzRandom()
    xm_vals[0], ym_vals[0], zm_vals[0] = initLorenzRandom()

    # Stepping through "time".
    kx = kxi
    ky = kyi
    kz = kzi
    synch_times = []
    couple_strength = []
    for couple_step in range(num_tests):
        kx = kx + kx_dt
        ky = ky + ky_dt
        kz = kz + kz_dt
        couple_strength.append(kx + ky + kz)
        these_solutions = []
        for _ in range(num_to_average):
            for i in range(0,max_steps-1):
                # Calculate the derivatives of the X, Y, Z state
                xo_dot, yo_dot, zo_dot = lorenz(xo_vals[i], yo_vals[i], zo_vals[i])
                xm_dot, ym_dot, zm_dot = lorenz(xm_vals[i], ym_vals[i], zm_vals[i])
                # Apply coupling
                ex = (xo_vals[i] - xm_vals[i])
                ey = (yo_vals[i] - ym_vals[i])
                ez = (zo_vals[i] - zm_vals[i])
                # Check for synchronization
                if ((abs(ex) < synch_threshold) and (abs(ey) < synch_threshold) and \
                   (abs(ez) < synch_threshold)):
                    break
                ux = kx * ex
                uy = ky * ey
                uz = kz * ez
                # Propagate the observed system
                xo_vals[i + 1] = xo_vals[i] + (xo_dot * dt)
                yo_vals[i + 1] = yo_vals[i] + (yo_dot * dt)
                zo_vals[i + 1] = zo_vals[i] + (zo_dot * dt)
                # Propagate the modeled system
                xm_vals[i + 1] = xm_vals[i] + (xm_dot * dt) + ux
                ym_vals[i + 1] = ym_vals[i] + (ym_dot * dt) + uy
                zm_vals[i + 1] = zm_vals[i] + (zm_dot * dt) + uz
            if (i == max_steps-2):
                log.out.debug("No synch achieved.")
            else:
                log.out.debug("Synch at time: " + str(i*dt) + " k<xyz> at : " + \
                              str(kx) + ", " + str(ky) + ", " + str(kz))
            these_solutions.append(i)
        average_convergence_time = np.average(these_solutions)
        log.out.info("Step " + str(couple_step+1) + "/" + str(num_tests) + \
                     " convergence time= " + str(average_convergence_time*dt))
        synch_times.append(average_convergence_time * dt)

    #### Plot the convergence
    axis = couple_strength
    fig_conv = plt.figure()
    fig_conv = plt.gcf()
    fig_conv.canvas.set_window_title('Convergence time vs. Coupling Strength')
    plt.plot(axis,synch_times,'r') # plotting t,a separately
    plt.show()


#########################
#         DRIVER        #
#########################
if __name__ == '__main__':
    log.setLevel('INFO') # Set the logging level
    # Report basic system info
    log.out.info("Python version: " + sys.version)
    CPUCOUNT = mp.cpu_count()
    log.out.info("Number of CPUs: " + str(CPUCOUNT))

    # Choose the demo

    # Couple weakly in two dimensions
    plotSynch(steps=16000, dt=0.001, kx=0.01, ky=0.01, kz=0.00,
              init_timesteps=2000, synch_max=4000)

#    # Couple strongly in one dimensions
#    plotSynch(steps=16000, dt=0.001, kx=0.1, ky=0.00, kz=0.00,
#              init_timesteps=2000, synch_max=6000)

#    # Couple systems with different parameters
#    plotSynch(steps=18000, dt=0.001, kx=0.02, ky=0.02, kz=0.02,
#              init_timesteps=2000, synch_max=12000,
#              sigma_o=10.0, rho_o=30.0, beta_o=3.0,
#              sigma_m=11.0, rho_m=33.0, beta_m=3.3)

#    # Add observational error
#    plotSynch(steps=20000, dt=0.001, kx=0.01, ky=0.01, kz=0.00,
#              init_timesteps=2000, synch_max=8000, obs_error=0.10)

#    # Try the Rossler attractor
#    plotSynch(steps=200000, dt=0.001, kx=0.01, ky=0.01, kz=0.00,
#              init_timesteps=100000, synch_max=102000, attractor='rossler')

#    # Try Chua's circuit
#    plotSynch(steps=400000, dt=0.001, kx=0.002, ky=0.002, kz=0.00,
#              init_timesteps=200000, synch_max=210000, attractor='chua')

#    measureSynchSpace()

    # Stop Logging
    #log.stopLog()

