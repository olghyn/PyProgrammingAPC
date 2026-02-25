import sys
import matplotlib.pyplot as plt
import random

def plot(N):
    
    x_grid,y_grid = pi_comp(1, N)[1]
    xc_grid,yc_grid = pi_comp(1, N)[2]
    
    plt.scatter(x_grid, y_grid, color='blue', s=10)
    plt.scatter(xc_grid, yc_grid, color='orange', s=5)
    plt.show()
    
    return

def pi_comp(r, N):
    
    counts_tot = []
    counts_pi = []
    x = []
    y = []
    x_c = []
    y_c = []
    
    for i in range(0, N):
        
        counts_tot.append(i)
        a = random.random()
        b = random.random()
        x.append(a)
        y.append(b)
        circ = a*a + b*b
        
        if circ <= r*r: 
            
            counts_pi.append(circ)
            x_c.append(a)
            y_c.append(b)
            
    pi_estimate = 4*len(counts_pi)/len(counts_tot)
        
    return pi_estimate, [x,y], [x_c, y_c]

def mean_err(N, n):
    
    pi_est = []
    
    for i in range(0,n):
        
        pi_est.append(pi_comp(1, N)[0])
        
    mean_pi = sum(pi_est)/len(pi_est)
    var = sum((i - mean_pi) ** 2 for i in pi_est) / len(pi_est)
        
    return mean_pi, var**(1/2)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Nothing to be done here, write num of iterations')
        sys.exit(0)  # Use sys.exit(), not exit()

    n = int(sys.argv[1])
    l = 1
    r = 1
    pi, err = mean_err(10000, n)
    print("pi: ", pi, "\nerr: ", err)   
            
