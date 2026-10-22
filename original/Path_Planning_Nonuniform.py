import numpy as np
import matplotlib.pyplot as plt


#### set input parameters
D=10
x1=-1
x2=1
y1=1
y2=-1
Q=2
V=0.5
input_parameters=(D,x1,x2,y1,y2,Q,V)
ytarget=0 ###target y-value
vm=1 ### swimmer's constant relative speed absolute value (m/s)

#### algorithm parameters
Np=100 ### total timesteps
total_time=12.7 #seconds
Dt=total_time/Np  ##timestep
e_dist=1e-02  ### maximum absolute error to stop
alpha=0.01 ###learning rate
e_theta=1e-03 ### for finite difference method step
#thetas=np.zeros(Np)  ### inital guess of angles, swim straight
thetas=0.87*np.ones(Np)

def River_Flow(X,Y,input_parameters):
    
    #### value gathering from a list/tuple
    (D,x1,x2,y1,y2,Q,V)=input_parameters
    #### 2D-variables (arrays NxN)
    R1=((X - x1)**2 + (Y - y1)**2)
    R2=((X - x2)**2 + (Y - y2)**2)
    Vx=Q/(2*np.pi)*((X - x1) / R1**2-(X - x2) / R2**2)
    Vy=-V+Q/(2*np.pi)*((Y - y1) / R1**2-(Y - y2) / R2**2)
    return(Vx,Vy)


def compute_cost(thetas,Dt,vm,ytarget,input_parameters):
    #### thetas: a Np-length vector of angles 
    #### each one of them corresponds to a particular time point 
    Np=len(thetas)
    D=input_parameters[0]
    ##### initial position is always considered as rA=(-D/2,0)
    x0,y0=-D/2,0
    xn,yn=x0,y0  ### initialize swimmer's position rA
    x_h=[x0]
    y_h=[y0]
    for n in range(0,Np):
        theta_n=thetas[n]
        vtotalx=vm*np.cos(theta_n)+River_Flow(xn,yn,input_parameters)[0]
        vtotaly=vm*np.sin(theta_n)+River_Flow(xn,yn,input_parameters)[1]
        xn+=Dt*vtotalx  ### update variable values for the next iteration
        yn+=Dt*vtotaly  ### update variable values for the next iteration
        x_h.append(xn)
        y_h.append(yn)
    #### when all steps are finished, we reclaim the final position 
    #### with the input thetas
    xNp,yNp=xn,yn
    J=(xNp-D/2)**2+(yNp-ytarget)**2
    return(J,xNp,yNp,x_h,y_h)
        

##### MAIN ITERATIVE PROCESS
max_iterations=400
J_history=[]
for iteration in range(max_iterations):
    ##### gradient computation
    gradJ=np.zeros(Np)
    for n in range(Np):
        theta_plus=thetas.copy() ### make a clone to avoid mutation
        theta_plus[n]+=e_theta
        theta_minus=thetas.copy() ### make a clone to avoid mutation
        theta_minus[n]-=e_theta
        ### compute gradients via the finite difference method
        gradJ[n]=(compute_cost(theta_plus,Dt,vm,ytarget,input_parameters)[0]-\
            compute_cost(theta_minus,Dt,vm,ytarget,input_parameters)[0])/(2*e_theta)
    ### after the computation of all the Np gradients, update the angle
    #### update rule
    #thetas=thetas-alpha*gradJ
    thetas-=alpha*gradJ
    J=compute_cost(thetas,Dt,vm,ytarget,input_parameters)[0]
    xNp=compute_cost(thetas,Dt,vm,ytarget,input_parameters)[1]
    yNp=compute_cost(thetas,Dt,vm,ytarget,input_parameters)[2]
    print('Iteration '+str(iteration+1)+':')
    print('Final Position: xNp='+str(np.round(xNp,2))+', yNp='+str(np.round(yNp,2)))
    print('Distance Deviation: '+str(np.round(np.sqrt(J),2))+ 'm')
    J_history.append(J)
    if J<e_dist:
        break
x_history=compute_cost(thetas,Dt,vm,ytarget,input_parameters)[3]
y_history=compute_cost(thetas,Dt,vm,ytarget,input_parameters)[4]



plt.figure()
plt.plot(np.arange(0,iteration+1),np.sqrt(J_history))
plt.xlim([0,iteration])
plt.grid()
plt.title('Deviation from target vs GDA iteration, D=10m, $v_m$='+str(vm)+'m/s, 100 steps')
plt.xlabel('# of GDA iteration')
plt.ylabel('Deviation from target (m)')



#### show swimmer's final path 
X, Y = np.meshgrid(np.linspace(-D/2, D/2, Np), np.linspace(-5, 5, Np))
Vx,Vy=River_Flow(X,Y,input_parameters)
dirx=np.cos(thetas)
diry=np.sin(thetas)

plt.figure()
plt.streamplot(X, Y, Vx, Vy, density=2, color='b')
plt.scatter([-D/2, D/2], [0, ytarget], c='r')  # Start/target
plt.scatter(x_history,y_history,c='r')
plt.xlabel('x (meters)')
plt.ylabel('y (meters)')
plt.title('Swimmer path on River flow')



arrowscale=2
plt.figure()
plt.streamplot(X, Y, Vx, Vy, density=2, color='b')
plt.scatter([-D/2, D/2], [0, ytarget], c='r')  # Start/target
plt.quiver(x_history[:-1],y_history[:-1], dirx,diry,scale_units='inches',scale=arrowscale, color='r')
plt.quiver(x_history[:-1],y_history[:-1],np.diff(x_history),np.diff(y_history),scale_units='inches',scale=arrowscale/10,color='g')
plt.xlabel('x (meters)')
plt.ylabel('y (meters)')
plt.title('Swimmer path on River flow; Red: selected angle. Green: Actual direction')
    


# plt.figure()
# plt.semilogy(np.arange(0,iteration+1),np.sqrt(J_history))
# plt.xlim([0,iteration])
# plt.grid()
# plt.title('Deviation from target vs GDA iteration, D=10m, v_m='+str(vm)+'m/s, 100 steps')
# plt.xlabel('# of GDA iteration')
# plt.ylabel('Deviation from target (m)')

# plt.figure()
# plt.loglog(np.arange(0,iteration+1),np.sqrt(J_history))
# plt.xlim([0,iteration])
# plt.grid()
# plt.title('Deviation from target vs GDA iteration, D=10m, v_m='+str(vm)+'m/s, 100 steps')
# plt.xlabel('# of GDA iteration')
# plt.ylabel('Deviation from target (m)')


plt.legend(('$u_m$ angle','$u_{tot}$ angle'))