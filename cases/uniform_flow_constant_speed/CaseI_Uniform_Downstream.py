import numpy as np
import matplotlib.pyplot as plt

#### parameters for question A

rA=np.array([[0],[0]]) #swimmer's initial position
D=10 #river's width (meters)
theta=0 #swimmer's swimming angle (rad) 
vs=np.array([[0],[-1]]) #river's downstream speed(m/s)
vm_norm=1.3 #absolute swimmer's speed (m/s)

### estimation of total speed, time and final position 
vm=vm_norm*np.array([[np.cos(theta)],[np.sin(theta)]]) #vectorized swimmer's speed
vtotal=vs+vm ## total speed (add the relatives)

### two [0][0] because vtotal has two dimensions 2x1
td=D/vtotal[0][0] #total time needed for the swimmer to cross the river

### final position
rB=rA+vtotal*td

vertical_displacement=rB[1][0]-rA[1][0]
print('For theta='+str(np.rad2deg(theta))+' degrees, the swimmer')
print('will reach the opposite bank with a vertical displacement of')
print(str(np.round(vertical_displacement,2))+' meters\n')

### Question C
"""
Estimate the theta that will lead the swimmer to the desired
position B (from A)
"""
rA=np.array([[0],[0]]) #swimmer's initial position
D=100 #river's width (meters)

vs=np.array([[0],[-1]]) #river's downstream speed(m/s)
vm_norm=1.3 #absolute swimmer's speed (m/s)
yBtargets=np.arange(-200,101,25)### list of target final y positions
rB=[]
for yb in yBtargets:
    rB.append(np.array([[D],[yb]])) #list of swimmer's final xB,yB positions

Nm=10000
thetas=np.linspace(np.deg2rad(-75),np.deg2rad(75),Nm) ### search list of angles
yBs=rA[1][0]+D*(np.tan(thetas)+vs[1][0]/(vm_norm*np.cos(thetas))) ### corresponding final y-coordinates for these angles

def find_func_min(x,f):
    """
    Returns the x* which minimizes a 1-D function f(x) 
    as well as the minimum f value and the corresponding index
    """
    fmin=np.min(f)
    index_min=np.where(f==fmin)[0][0]
    optimal_x=x[index_min]
    return(optimal_x,fmin,index_min)
    
### find the optimal angle for each y-target case and fill the list optimal_thetas
optimal_thetas=[]
indices=[]
for yb in yBtargets:
    f_theta=np.abs(yBs-yb)
    optimal_theta,e,imin=find_func_min(thetas,f_theta)
    optimal_thetas.append(optimal_theta)
    indices.append(imin)
    print('The swimmer wants to swim from A to B.\n')
    print('Under the previous parameters,\nthe optimal angle is '+str(np.round(np.rad2deg(optimal_theta),2))+' degrees')
    print('The absolute error of this solution is '+str(np.round(e,6)))

plt.figure()
plt.plot(np.rad2deg(thetas),yBs)
for optimal_theta,imin in zip(optimal_thetas,indices):
    plt.plot([np.rad2deg(optimal_theta), np.rad2deg(optimal_theta)],[-750,250],color='red',linestyle='--')
    plt.plot(np.rad2deg(optimal_theta),yBs[imin],'*',markersize=15)
plt.grid()
plt.xlim([-75,75])
plt.ylim([-750,250])
plt.xlabel('$\Theta (^o)$')
plt.ylabel('Vertical displacement y (m)')
plt.title('Downstream shift across y axis')


#==========================OTHER parametrisations============================
####################### Parametrize swimmer's speed
um_speeds=np.arange(1.2,3.1,0.2)
#### D, us, same as before
yBs_ums=[]  ### a list which includes the resulting yBs for all the swimmer's speed and angle cases
plt.figure()
um_index=0
for um in um_speeds:
    yBs_ums.append(rA[1][0]+D*(np.tan(thetas)+vs[1][0]/(um*np.cos(thetas)))) 
    ### corresponding final y-coordinates for these angles
    plt.plot(np.rad2deg(thetas),yBs_ums[um_index])
    um_index+=1
plt.grid()
plt.xlim([-75,75])
plt.ylim([-750,250])
plt.xlabel('$\Theta (^o)$')
plt.ylabel('Vertical displacement y (m)')
plt.title('Downstream shift across y axis')
#plt.legend(['$u_m$=1.2m/s','$u_m$=1.4m/s','$u_m$=1.6m/s','$u_m$=1.8m/s',\
#            '$u_m$=2m/s','$u_m$=2.2m/s','$u_m$=2.4m/s','$u_m$=2.6m/s',\
#                '$u_m$=2.8m/s','$u_m$=3m/s'])


####################### Parametrize river width D
Ds=np.arange(10,101,10)
vs=np.array([[0],[-1]]) #river's downstream speed(m/s)
vm_norm=1.3 #absolute swimmer's speed (m/s)
yBs_Ds=[]  ### a list which includes the resulting yBs for all D and angle cases
plt.figure()
Ds_index=0
for Di in Ds:
    yBs_Ds.append(rA[1][0]+Di*(np.tan(thetas)+vs[1][0]/(um*np.cos(thetas)))) 
    ### corresponding final y-coordinates for these angles
    plt.plot(np.rad2deg(thetas),yBs_Ds[Ds_index])
    Ds_index+=1
plt.grid()
plt.xlim([-75,75])
plt.ylim([-750,250])
plt.xlabel('$\Theta (^o)$')
plt.ylabel('Vertical displacement y (m)')
plt.title('Downstream shift across y axis')
#plt.legend(['$D$=10m','$D$=20m','$D$=30m','$D$=40m','$D$=50m',\
#            '$D$=60m','$D$=70m','$D$=80m','$D$=90m','$D$=100m'])














