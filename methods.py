import numpy as np

#-------------------------------------------------------------------------------
# --------------------------- Numerical Integration ----------------------------
#-------------------------------------------------------------------------------

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
        u[i] = newton_root(lambda y: u[i-1] + h/2*(f(y, x[i])+ f(u[i-1], x[i-1]))
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
        x.append(x[-1] + h)             # adding x and u to array
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
     steps: number of steps to use (2,3, or 4) ## to do
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
    
    # setting first and last points Au_{0} = ua, Au_{N+2} = ub
    A[0,0] = 1
    B[0] = ua
    A[-1,-1] = 1
    B[-1] = ub
    
    # constructing edges of A and B (Lec. 8 p. 3-4)
    A[1,1]   = -2/h**2 + q(x[0])
    A[1,2]   = 1/h**2 + p(x[0])/(2*h)
    B[1]     = g(x[0]) - (1/h**2 - p(x[0])/(2*h))*ua
    A[N,N-1] = 1/h**2 - p(x[N+1])/(2*h)
    A[N,N]   = -2/h**2 + q(x[N+1])
    B[N]     = g(x[N+1]) - (1/h**2 + p(x[N+1])/(2*h))*ub
    
    # constructing A & B for the rest of the points
    for i in range(2, len(A)-2) :  
        A[i,i-1] = 1/h**2 - p(x[i])/(2*h)
        A[i,i  ] = -2/h**2 + q(x[i])
        A[i,i+1] = 1/h**2 + p(x[i])/(2*h)
        B[i]     = g(x[i])  
    
    # solve linear system
    u = np.linalg.solve(A, B)
    return x, u  
    

    
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