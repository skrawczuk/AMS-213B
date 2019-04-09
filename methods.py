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
    
    I = 0                  # result of integration
    I =+ f(a) * h/2        # first point
    I =+ f(b) * h/2        # last point
    
    for i in range(N) :  # middle points
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
    
    I = 0                  # result of integration
    I =+ f(a) * h/6        # first point
    I =+ f(b) * h/6        # last point
    
    for i in range(1,N) :  # middle points: full steps
        I += 2*h/6 * f(x[i])
        
    for i in range(1,N+1) :    # middle points: half steps
        I += 4*h/6 * f(x[i] - h/2)
        
    return I
        
#-------------------------------------------------------------------------------
# --------------------------- Differential Equations ---------------------------
#-------------------------------------------------------------------------------
                        
def forward_euler(f, r, h, u0) : 
    '''Solves IVP u'=f(u,x), u(0) = u0
       f: Function describing derivative of u
       r: Range to be evaluated over; [first, last]
       h: step size in x
      u0: initial value of u
       '''
    a, b = r  
    N = int(float((b-a)) / h)
    x = np.linspace(a,b,N+1)
    
    u = np.zeros(N+1) 
    u[0] = u0
    
    for i in range(1, N+1) : 
        u[i] = u[i-1] + h*f(u[i-1], x[i-1])
        
    return x, u
    

def backward_euler(f, r, h, u0) :  
    '''Solves IVP u'=f(u,x), u(0) = u0
       f: Function describing derivative of u
       r: Range to be evaluated over; [first, last]
       h: step size in x
      u0: initial value of u
       '''
    a, b = r
    N = int(float((b-a)) / h)
    x = np.linspace(a,b,N+1)
    
    u = np.zeros(N+1) 
    u[0] = u0
     
    for i in range(1, N+1) : 
        u[i] = newton_solver(lambda y: u[i-1] + h*f(y, x[i]) - y, u[i-1], 1e-5)
   
    return x, u  
 
    
def rk4(f, r, h, u0) : 
    '''Uses 4th order Runge-Kutta to solve IVP u' = f(u,x), u(0)=u0
       f: Function describing derivative of u
       r: Range to be evaluated over; [first, last]
       h: step size in x
      u0: list of initial values of system u
    '''
    a, b = r
    N = int(float((b-a)) / h)
    x = np.linspace(a,b,N+1)
    
    u = np.zeros((N+1, len(u0)))
    u[0] = u0
    
    for i in range(1, N+1) : 
        

#-------------------------------------------------------------------------------
# ------------------------ Optimization & Root Finding -------------------------
#-------------------------------------------------------------------------------
                    
      
def newton_solver(f, x0, tol) : 
    '''finds a root of the given function
        f: Function to find root of 
       x0: initial guess
      tol: tolerance for zero
    '''
    h = 1e-5    # step size for differentiation
    error = 1   
    num_iters = 0
    x = x0
    
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
    
    
    
    