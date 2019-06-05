import numpy as np

#-------------------------------------------------------------------------------
# ------------------ Numerical Integration & differentiation -------------------
#-------------------------------------------------------------------------------

def diff_1st_order(f, x, h) : 
    '''Takes the first order derivative of function f at given x
       f: Function to be differentiated
       x: values at which derivative is evaluated
       h: step size for derivation
    '''
    return  (f(x+h) - f(x)) / h
    

def composite_trapezoidal(f, r, N) : 
    '''Integration by composite trapezoidal rule.
       f: Function being integrated
       r: Range to be evaluated over; [first, last]
       N: number of steps
    '''
    a, b = r
    x = np.linspace(a,b,N+1) # points to evaluate
    h = float((b-a)) / N     # step size
    
    I = 0                    # result of integration
    I =+ f(a) * h/2          # first point
    I =+ f(b) * h/2          # last point
      
    for i in range(N) :      # middle points
        I += f(x[i]) * h
    
    return I


def composite_simpson(f, r, N) : 
    '''Integration by composite Simpson's rule.
       f: Function being integrated
       r: Range to be evaluated over; [first, last]
       N: number of steps
    '''
    a, b = r
    x = np.linspace(a,b,N+1) # points to evaluate   
    h = float((b-a)) / N     # step size             
    
    I = 0                    # result of integration
    I =+ f(a) * h/6          # first point
    I =+ f(b) * h/6          # last point
    
    for i in range(1,N) :    # middle points: full steps
        I += 2*h/6 * f(x[i])
    
    for i in range(1,N+1) :  # middle points: half steps
        I += 4*h/6 * f(x[i] - h/2)
        
    return I
     
           
#-------------------------------------------------------------------------------
# -------------------------------- ODE Solvers ---------------------------------
#-------------------------------------------------------------------------------
                        
def forward_euler(f, r, h, u0) : 
    '''Solves IVP u'=f(u,x), u(0) = u0 with forward Euler method
       f: Function describing derivative of u
       r: Range to be evaluated over; [first, last]
       h: step size in x
      u0: list of initial values of system u
       '''
    a, b = r  
    N = int(float((b-a)) / h)     # number of steps 
    x = np.linspace(a,b,N+1)      # x points to evaluate
    
    u = np.zeros((N+1, len(u0)))  # solution array
    u[0] = u0
    
    for i in range(1, N+1) : 
        u[i] = u[i-1] + h*f(u[i-1], x[i-1])
        
    return x, u


def backward_euler(f, r, h, u0) :   
    '''Solves IVP u'=f(u,x), u(0) = u0 with backward Euler method
       f: Function describing derivative of u
       r: Range to be evaluated over; [first, last]
       h: step size in x
      u0: initial value of u ## add vector input
    '''
    a, b = r
    N = int(float((b-a)) / h)   # number of steps 
    x = np.linspace(a,b,N+1)    # x points to evaluate
      
    u = np.zeros(N+1)           # solution array
    u[0] = u0
     
    for i in range(1, N+1) :    # solving for u in implicit eq. for each step
        u[i] = newton_root(lambda y: u[i-1] + h*f(y, x[i]) - y, u[i-1], 1e-5)
   
    return x, u  


def trapezoidal_ode(f, r, h, u0) : 
    '''Solves IVP u'=f(u,x), u(0) = u0 with trapezoidal method
       f: Function describing derivative of u
       r: Range to be evaluated over; [first, last]
       h: step size in x
      u0: initial value of u ## add vector input
    '''
    a, b = r
    N = int(float((b-a)) / h)   # number of steps 
    x = np.linspace(a,b,N+1)    # x points to evaluate
      
    u = np.zeros(N+1)           # solution array
    u[0] = u0
    
    for i in range(1, N+1) :    # solving for u in implicit eq. for each step
        u[i] = newton_root(lambda y: u[i-1] + h/2*(f(y, x[i])+f(u[i-1], x[i-1]))
                                                             - y, u[i-1], 1e-5)
        
    return x, u  
 
 
def midpoint_2step(f, r, h, u0) : 
    '''Solves IVP u'=f(u,x), u(0) = u0 with 2-step midpoint method
       f: Function describing derivative of u
       r: Range to be evaluated over; [first, last]
       h: step size in x
      u0: First two values of u; [u0,u1]
    '''
    a, b = r
    N = int(float((b-a)) / h)   # number of steps 
    x = np.linspace(a,b,N+1)    # x points to evaluate
      
    u = np.zeros(N+1)           # solution array
    u[0:2] = u0
    for i in range(2, N+1) :
        u[i] = u[i-2] + 2*h*f(u[i-1], x[i-1])
     
    return x, u
     
 

def get_rk_matrices(method, alpha=0) : 
    '''Returns A, b, and c matrices for specified Runge Kutta method'''
    matrices = {
                
                'RK2':       [np.array([[0, 0],
                                    [0.5, 0]]), 
                              np.array([0, 1]), 
                              np.array([0, 0.5])],
                
                'RK4':       [np.array([[0, 0, 0, 0],
                                  [0.5, 0, 0, 0],
                                  [0, 0.5, 0, 0],
                                  [0, 0, 1, 0]]), 
                              np.array([1/6., 1/3., 1/3., 1/6.]), 
                              np.array([0, 0.5, 0.5, 1])],

            '2S-DIRK':       [np.array([[alpha, 0],
                                [1-alpha, alpha]]), 
                              np.array([1-alpha, alpha]), 
                              np.array([alpha, 1.])],               
                                                
           'Fehlberg':   [np.array([[0, 0, 0, 0, 0, 0], 
                                [0.25, 0, 0, 0, 0, 0], 
                                [3/32., 9/32., 0, 0, 0, 0], 
                                [1932/2197., -7200/2197., 7296/2197., 0, 0, 0], 
                                [ 439/216., -8, 3680/513., -845/4104., 0, 0], 
                                [-8/27., 2,-3544/2565., 1859/4104.,-11/40.,0]]),
                np.array([[16/135., 0, 6656/12825.,28561/56430., -9/50., 2/55.], 
                           [25/216., 0, 1408/2565., 2197/4104., -.2, 0]]),
                np.array([0, .25, 3/8., 12/13., 1, 0.5])]
            

    
                }
    return matrices[method]
 

def runge_kutta(f, r, h, u0, method) :  
    '''Uses specified Runge-Kutta method to solve IVP u' = f(u,x), u(0)=u0
         f: Function describing derivative of u
         r: Range to be evaluated over; [first, last]
         h: step size in x
        u0: list of initial values of system u
    method: 'RK2', 'RK4', ...
    '''
    x_0, x_f = r
    N = int(float((x_f-x_0)) / h)    # number of steps 
    x = np.linspace(x_0,x_f,N+1)     # x points to evaluate
    
    u = np.zeros((N+1, len(u0)))     # solution array
    u[0] = u0
    
    A,b,c = get_rk_matrices(method)  # getting coefficients for method used
    k = np.zeros((len(A), len(u0)))
 
    for i in range(1, N+1) : 
        for j in range(len(k)) : 
            k[j] = h*f(u[i-1] + np.dot(A[j], k), x[i-1] + c[j]*h)
        u[i] = u[i-1] + np.dot(b, k)
        
    return x, u
    

def runge_kutta_implicit(f, r, h, u0, method, alpha) :  
    '''Uses an implicit Runge-Kutta method to solve IVP u' = f(u,x), u(0)=u0
         f: Function describing derivative of u
         r: Range to be evaluated over; [first, last]
         h: step size in x
        u0: list of initial values of system u
    method: '2S-DIRK', ...
     alpha: value of alpha for method
    '''
    x_0, x_f = r
    N = int(float((x_f-x_0)) / h)    # number of steps 
    x = np.linspace(x_0,x_f,N+1)     # x points to evaluate
    
    u = np.zeros((N+1, len(u0)))     # solution array
    u[0] = u0
    
    A,b,c = get_rk_matrices(method, alpha=alpha) 
    k = np.zeros((len(A), len(u0)))
 
    for i in range(1, N+1) : 
        for j in range(1,len(k)) : 
            #defining k1 - h*f(...) = 0
            k_j = lambda y: y-h*f(u[i-1] + np.dot(A[j], k) # sum of row
                                - A[j,j]*k[j] + y*A[j,j],  # solve for diagonal
                                  x[i-1] + c[j]*h)
            k[j] = newton_root(k_j, 0, 1e-5)     # solving for jth element of k
            
        u[i] = u[i-1] + np.dot(b, k)
        
    return x, u
    
    
def runge_kutta_adaptive(f, r, h, u0, method, eps) : 
    '''Uses specified Runge-Kutta method to solve IVP u' = f(u,x), u(0)=u0
         f: Function describing derivative of u
         r: Range to be evaluated over; [first, last]
         h: step size in x
        u0: list of initial values of system u
    method: 'Felhberg', ...
       eps: error threshold for adaptive step
    '''
    x_0, x_f = r
    x = [x_0]                           # points to be evaluated over 
    u = [u0]                            # solution array
    A,b,c = get_rk_matrices(method)     # getting coefficients for method used
    k = np.zeros((len(A), len(u0)))
    while x[-1] <= x_f :                # evaluating until final x is reached
        error = eps + 1                 # initalizing error for each step
        while error >= eps :            # evaluating step until error satisfied
            for j in range(len(k)) : 
                k[j] = h*f(u[-1] + np.dot(A[j], k), x[-1] + c[j]*h)
            
            u_5 = u[-1] + np.dot(b[0], k)        # using higher order b
            u_4 = u[-1] + np.dot(b[1], k)        # using lower order b 
            error = np.abs((u_5 - u_4)[0]) / h   # numerical error at step
             
            d = (0.5*eps/error)**(0.25)  # updating timestep
            if d > 4 : 
                h = 4*h
            elif d < 0.1 : 
                h = 0.1*h
            else : 
                h = d*h
        x.append(x[-1] + h)  # adding x and u to array
        u.append(u_5)     
        
    x = np.array(x)
    u = np.array(u)

    return x, u
                            
  
def BDF2(f, r, h, u0) :   
    '''Uses backwards difference formula to solve IVP u' = f(u,x), u(0)=u0
         f: Function describing derivative of u
         r: Range to be evaluated over; [first, last]
         h: step size in x
        u0: list of initial values of system u
    '''  
    a, b = r
    N = int(float((b-a)) / h)   # number of steps 
    x = np.linspace(a,b,N+1)    # x points to evaluate
      
    u = np.zeros(N+1)           # solution array
    u[0:2] = u0
    for i in range(2, N+1) :
        u[i] = newton_root(lambda y:1.5*y - 2*u[i-1] + .5*u[i-2] - h*f(y, x[i]),
                           u[i-1], 1e-8 )
    return x, u  
     

def FDM_BVP(p, q, g, r, h, ua, ub) : 
    '''Uses finite difference method to solve boundary value problem: 
       u'' + p(x)u' + q(x)u = g(x)
       u(a) = ua, u(b) = ub
   p, q, g: coefficients of u terms in ODE
         r: Range to be evaluated over; [a, b]
         h: step size in x
        ua: value at u(a) 
        ub: value at u(b) 
    ''' 
    a, b = r
    N = int(float((b-a)) / h)  
    x = np.linspace(a,b,N+2)                   
    
    # Au = B -> solve linear system for u
    A = np.zeros((N+2,N+2))
    B = np.zeros(N+2)
    
    # setting boundary points: Au_{0} = ua, Au_{N+1} = ub
    A[0,0] = 1
    B[0] = ua
    A[-1,-1] = 1
    B[-1] = ub
    
    # constructing A & B for the rest of the points (Lec. 8 p. 3-4)
    for i in range(1, len(A)-1) :  
        A[i,i-1] = 1/h**2 - p(x[i])/(2*h)
        A[i,i  ] = -2/h**2 + q(x[i])
        A[i,i+1] = 1/h**2 + p(x[i])/(2*h)
        B[i]     = g(x[i])  
    
    # solve linear system
    u = np.linalg.solve(A, B)
    return x, u  
    

#-------------------------------------------------------------------------------
# -------------------------------- PDE Solvers ---------------------------------
#-------------------------------------------------------------------------------

def FTCS_heat(f, g1, g2, rx, rt, dx, dt) : 
    '''solves IBVP u_t = u_xx, u(x,0)=f(x), u(a,t)=g1(t), u(b,t)=g2(t)
       for a<=x<=b, t>0
       using forward-time central-space method
   f,g1,g2: functions describing initial and boundary conditions
        rx: range of x; [a, b]
        rt: range of t; [t0, tf]
        dx: step in x
        dt: step in t
    '''   
    a, b   = rx
    t0, tf = rt
    x = np.arange(a,  b+dx,  dx)
    t = np.arange(t0, tf+dt, dt)
    u = np.zeros((len(x),len(t)))
    
    r = dt/dx**2
    nx = len(x)
    nt = len(t)
    # setting initial and boundary conditions
    u[ :,0] = f(x)
    u[ 0,:] = g1(t)
    u[-1,:] = g2(t)
    
    for i in range(nt-1) :
        u[1:nx-1, i+1] = u[1:nx-1, i] + r*(u[2:nx,i] - 
                                         2*u[1:nx-1,i] + 
                                           u[0:nx-2,i])
    
    return t, x, u
    

def BTCS_heat(f, g1, g2, rx, rt, dx, dt) : 
    '''solves IBVP u_t = u_xx, u(x,0)=f(x), u(a,t)=g1(t), u(b,t)=g2(t)
       for a<=x<=b, t>0 
       using backward-time, central-space method
   f,g1,g2: functions describing initial and boundary conditions
        rx: range of x; [a, b]
        rt: range of t; [t0, tf]
        dx: step in x
        dt: step in t
    '''   
    a, b   = rx
    t0, tf = rt
    x = np.arange(a,  b+dx,  dx)
    t = np.arange(t0, tf+dt, dt)
    u = np.zeros((len(x),len(t)))
    
    r = dt/dx**2
    nx = len(x)
    nt = len(t)
    
    # setting initial conditions
    u[ :,0] = f(x)
    u[ 0,:] = g1(t)
    u[-1,:] = g2(t)
    
    # solve Au_{n+1} = B
    #constructing A
    A = np.zeros((nx,nx))
    A[0,0] = 1
    A[-1,-1] = 1
    for j in range(1,nx-1) : 
        A[j, j-1] = r
        A[j, j  ] = -(1+2*r)
        A[j, j+1] = r
    
    B = np.zeros(nx)
    for i in range(nt-1) :
        # constructing B for timestep
        B[0]  = g1(t[i+1])
        B[-1] = g2(t[i+1]) 
        B[1:nx-1] = -u[1:nx-1,i]
        
        # solve linear system 
        u[:,i+1] = np.linalg.solve(A, B)
    
    return t, x, u
    
       
def crank_nicolson_heat(f, g1, g2, rx, rt, dx, dt) :
    '''solves IBVP u_t = u_xx, u(x,0)=f(x), u(a,t)=g1(t), u(b,t)=g2(t)
       for a<=x<=b, t>0
       using crank-nicolsen method
   f,g1,g2: functions describing initial and boundary conditions
        rx: range of x; [a, b]
        rt: range of t; [t0, tf]
        dx: step in x
        dt: step in t
    ''' 
    a, b   = rx
    t0, tf = rt
    x = np.arange(a,  b+dx,  dx)
    t = np.arange(t0, tf+dt, dt)
    u = np.zeros((len(x),len(t)))
    
    r = dt/dx**2
    nx = len(x)
    nt = len(t)
    
    # setting initial conditions
    u[ :,0] = f(x)
    u[ 0,:] = g1(t)
    u[-1,:] = g2(t)
    
    # solve Au_{n+1} = B:
    #constructing A
    A = np.zeros((nx,nx))
    A[0,0] = 1
    A[-1,-1] = 1
    for j in range(1,nx-1) : 
        A[j, j-1] = -r/2
        A[j, j  ] = 1+r
        A[j, j+1] = -r/2
    
    B = np.zeros(nx) 
    for i in range(nt-1) :
        # constructing B for timestep
        B[0]  = g1(t[i+1])
        B[-1] = g2(t[i+1]) 
        B[1:nx-1] = r/2*(u[0:nx-2,i] + u[2:nx,i]) + (1-r)*u[1:nx-1,i]
        
        # solve linear system         
        u[:,i+1] = np.linalg.solve(A, B)
    
    return t, x, u
                                                                   

def upwind_hyperbolic(f, g1, g2, k, rx, rt, dx, dt) : 
    '''Solves IBVP u_t + k*u_x = 0, u(x,0)=f(x), u(a,t)=g1(t), u(b,t)=g2(t) 
       for a<=x<=b, t>0
       using the upwind method                                            
   f,g1,g2: functions describing initial and boundary conditions
         k: coefficient of u_x term
        rx: range of x; [a, b]
        rt: range of t; [t0, tf]
        dx: step in x
        dt: step in t
    '''
    a, b   = rx
    t0, tf = rt
    x = np.arange(a,  b+dx,  dx)
    t = np.arange(t0, tf+dt, dt)
    u = np.zeros((len(x),len(t)))
    
    r = dt/dx
    nx = len(x)
    nt = len(t)
    # setting initial and boundary conditions
    u[ :,0] = f(x)
    u[ 0,:] = g1(t)
    u[-1,:] = g2(t)
    
    for i in range(nt-1) :
        u[1:nx-1, i+1] = u[1:nx-1, i] - k*r*(u[1:nx-1,i] - u[0:nx-2,i])
    
    return t, x, u                                                                                                                                                                     


def lax_friedrichs_hyperbolic(f, g1, g2, k, rx, rt, dx, dt) : 
    '''Solves IBVP u_t + k*u_x = 0, u(x,0)=f(x), u(a,t)=g1(t), u(b,t)=g2(t) 
       for a<=x<=b, t>0
       using the Lax-Friedrichs method                                            
   f,g1,g2: functions describing initial and boundary conditions
         k: coefficient of u_x term
        rx: range of x; [a, b]
        rt: range of t; [t0, tf]
        dx: step in x
        dt: step in t
    '''
    a, b   = rx
    t0, tf = rt
    x = np.arange(a,  b+dx,  dx)
    t = np.arange(t0, tf+dt, dt)
    u = np.zeros((len(x),len(t)))
    
    r = dt/dx
    nx = len(x)
    nt = len(t)
    # setting initial and boundary conditions
    u[ :,0] = f(x)
    u[ 0,:] = g1(t)
    u[-1,:] = g2(t)
    
    for i in range(nt-1) :
        u[1:nx-1, i+1] = ((u[2:nx, i] + u[0:nx-2, i])/2
                         - k*r/2 * (u[2:nx,i] - u[0:nx-2,i]))
        
    return t, x, u 
    

def lax_wendroff_hyperbolic(f, g1, g2, k, rx, rt, dx, dt) : 
    '''Solves IBVP u_t + k*u_x = 0, u(x,0)=f(x), u(a,t)=g1(t), u(b,t)=g2(t) 
       for a<=x<=b, t>0
       using the Lax-Wendroff method                                            
   f,g1,g2: functions describing initial and boundary conditions
         k: coefficient of u_x term
        rx: range of x; [a, b]
        rt: range of t; [t0, tf]
        dx: step in x
        dt: step in t
     '''
    a, b   = rx
    t0, tf = rt
    x = np.arange(a,  b+dx,  dx)
    t = np.arange(t0, tf+dt, dt)
    u = np.zeros((len(x),len(t)))
    
    r = dt/dx
    nx = len(x)
    nt = len(t)
    # setting initial and boundary conditions
    u[ :,0] = f(x)
    u[ 0,:] = g1(t)
    u[-1,:] = g2(t)
    
    for i in range(nt-1) :
        u[1:nx-1, i+1] = (u[1:nx-1, i] 
                          - k*r/2*(u[2:nx, i]-u[0:nx-2, i])
                          + (k*r)**2/2 * (u[2:nx,i] 
                                      - 2*u[1:nx-1, i]
                                        + u[0:nx-2,i]))
    
    return t, x, u 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
#-------------------------------------------------------------------------------
# ---------------------------------- Root Finding ------------------------------
#-------------------------------------------------------------------------------
                    
      
def newton_root(f, x0, tol) : 
    '''returns a root of the given function using the Newton method
        f: Function to find root of 
       x0: initial guess
      tol: tolerance for the zero
    '''
    h = 1e-5       # step size for differentiation
    error = 1      # error to check for converence
    num_iters = 0  # number of iterations
    x = x0         # initializing x 

    while error > tol :  
        # check for non-convergence
        if num_iters > 40 :  
            break
            
        num_iters += 1
        # numerical differentiation
        df = (f(x+h)  - f(x-h)) / (2*h)
        dx = -f(x) / df
        
        # update error and x
        error = np.abs(dx)
        x += dx

    return x
    