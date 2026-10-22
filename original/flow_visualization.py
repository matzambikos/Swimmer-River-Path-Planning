import numpy as np
import matplotlib.pyplot as plt
D=10
x1=-1
x2=1
y1=1
y2=-1
Q=1
V=1

N=100
X, Y = np.meshgrid(np.linspace(-D/2, D/2, N), np.linspace(-5, 5, N))

input_parameters=(D,x1,x2,y1,y2,Q,V)

def River_Flow(X,Y,input_parameters):
    
    #### value gathering from a list/tuple
    (D,x1,x2,y1,y2,Q,V)=input_parameters
    #### 2D-variables (arrays NxN)
    R1=((X - x1)**2 + (Y - y1)**2)
    R2=((X - x2)**2 + (Y - y2)**2)
    Vx=Q/(2*np.pi)*((X - x1) / R1**2-(X - x2) / R2**2)
    Vy=-V+Q/(2*np.pi)*((Y - y1) / R1**2-(Y - y2) / R2**2)
    return(Vx,Vy)

Vx,Vy=River_Flow(X,Y,input_parameters)

yt=-1
plt.figure()
plt.quiver(X, Y, Vx, Vy, scale=10)
plt.scatter([-D/2, D/2], [0, yt], c='r')  # Start/target
plt.figure()
plt.streamplot(X, Y, Vx, Vy, density=2, color='b')
plt.scatter([-D/2, D/2], [0, yt], c='r')  # Start/target
plt.xlabel('x (meters)')
plt.ylabel('y (meters)')
plt.title('River flow')



