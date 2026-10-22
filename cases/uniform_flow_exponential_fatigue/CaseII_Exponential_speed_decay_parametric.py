import numpy as np
import matplotlib.pyplot as plt


def find_func_min(x,f):
    """
    Returns the x* which minimizes a 1-D function f(x) 
    as well as the minimum f value and the corresponding index
    """
    ### remove nans from f
    fnan=np.isnan(f)
    offset=np.where(fnan==False)[0][0]
    flimited=f[np.logical_not(fnan)]
    #xlimited=x[np.logical_not(fnan)]
    fmin=5000 ### mod to get the first minima
    for fi in flimited:
        if fi<fmin:
            fmin=fi
        else:
            break
    #min=np.min(flimited)
    index_min=offset+np.where(flimited==fmin)[0][0]
    optimal_x=x[index_min]
    return(optimal_x,fmin,index_min)



#================== um as parameter================================

rA=np.array([[0],[0]]) #swimmer's initial position
D=10 #river's width (meters)
vs=np.array([[0],[-1]]) #river's downstream speed(m/s)
vm_initial_list=[1.3, 1.48, 1.5, 1.7, 2] #initial absolute swimmer's speed (m/s)
adecay=0.03 #swimmer's speed decay factor
ytarget=0

Nm=10000
thetas=np.linspace(np.deg2rad(-75),np.deg2rad(75),Nm) 
yactlist=[]
for vmi in vm_initial_list:
    if adecay>0:
        tds=1/adecay*np.log((vmi*np.cos(thetas))/(vmi*np.cos(thetas)-D*adecay))
    else: # case constant speed, no decay
        tds=D/(vmi*np.cos(thetas))
    ### estimation of total speed, time and final position 
    vm=np.squeeze(vmi*np.exp(-adecay*tds)*np.array([[np.cos(thetas)],[np.sin(thetas)]])) #vectorized swimmer's speed
    vtotal=vm+vs

    ###### equation that gives the final point as a function of theta\
    if adecay!=0:
        yact_i=rA[1]+vmi*np.sin(thetas)*\
            1/adecay*(1-(vmi*np.cos(thetas)-D*adecay)/(vmi*np.cos(thetas)))+vs[1]*tds
    else:
        yact_i=rA[1]+vtotal[1]*tds
    H=np.abs(yact_i-ytarget)
    yactlist.append(yact_i)
    thetaopt,mindist,index_min=find_func_min(thetas,H)
    print('For a='+str(adecay)+', D='+str(D)+'m, and swimmer''s initial relative speed um='+str(vmi)+'m/s, ')
    print('the needed angle is '+str(np.round(np.rad2deg(thetaopt),4))+' degrees (solution with minimum abs.')
    print('The needed time to cross is t='+str(tds[index_min]))
    print('\n')



plt.figure()
for yacti in yactlist:
    plt.plot(np.rad2deg(thetas),yacti)
plt.plot([-75,75],[ytarget,ytarget],'--')
plt.xlim([25,65])
plt.ylim([-10,10])
plt.grid()
plt.xlabel('$\Theta$ (degrees)')
plt.ylabel('y (m)')
#plt.legend(('u_m=1.3m/s','u_m=1.48m/s','u_m=1.5m/s','u_m=1.7m/s','u_m=2m/s','target y'))
#plt.title('Deviation for D=10m, river downforce 1m/s, a=0.03 and parametric initial swimmer''s speed')


print('For u_m=1.48m/s it is possible to reach the target')


#================== a as parameter================================

vm_initial=1.7
##### different parameters a; for constant speed
adecaylist=[0, 0.01, 0.03, 0.04, 0.05, 0.06]
yactlist=[]
for adecay in adecaylist:
    if adecay!=0:
        tds=1/adecay*np.log((vm_initial*np.cos(thetas))/(vm_initial*np.cos(thetas)-D*adecay))
    else: # case constant speed, no decay
        tds=D/(vm_initial*np.cos(thetas))
    ### estimation of total speed, time and final position 
    if adecay!=0:
        vm=np.squeeze(vm_initial*np.exp(-adecay*tds)*np.array([[np.cos(thetas)],[np.sin(thetas)]])) #vectorized swimmer's speed
    else:
        vm=np.squeeze(vm_initial*np.array([[np.cos(thetas)],[np.sin(thetas)]]))
    vtotal=vm+vs

    ###### equation that gives the final point as a function of theta\
    if adecay!=0:
        yact_i=rA[1]+vm_initial*np.sin(thetas)*\
            1/adecay*(1-(vm_initial*np.cos(thetas)-D*adecay)/(vm_initial*np.cos(thetas)))+vs[1]*tds
    else:
        yact_i=rA[1]+vtotal[1]*tds
    H=np.abs(yact_i-ytarget)
    yactlist.append(yact_i)
    thetaopt,mindist,index_min=find_func_min(thetas,H)
    print('For a='+str(adecay)+', D='+str(D)+'m, and swimmer''s initial relative speed um='+str(vmi)+'m/s, ')
    print('the needed angle is '+str(np.round(np.rad2deg(thetaopt),4))+' degrees (solution with minimum abs.')
    print('The needed time to cross is t='+str(tds[index_min]))
    print('\n')

plt.figure()
for yacti in yactlist:
    plt.plot(np.rad2deg(thetas),yacti)
plt.plot([-75,75],[ytarget,ytarget],'--')
plt.xlim([30,65])
plt.ylim([-10,10])
plt.grid()
plt.xlabel('$\Theta$ (degrees)')
plt.ylabel('y (m)')
#plt.legend(('a=0','a=0.01','a=0.03','a=0.04','a=0.05','a=0.06','target y'))
#plt.title('Deviation for D=10m, river downforce 1m/s, initial $u_m$=1.7m/s and parametric a')


print('For a<0.05 it is possible to reach the target')


#================== D as parameter================================
vm_initial=1.7
adecay=0.03
Dlist=[5,10,15,16.5]

yactlist=[]
for D in Dlist:
    if adecay!=0:
        tds=1/adecay*np.log((vm_initial*np.cos(thetas))/(vm_initial*np.cos(thetas)-D*adecay))
    else: # case constant speed, no decay
        tds=D/(vm_initial*np.cos(thetas))
    ### estimation of total speed, time and final position 
    if adecay!=0:
        vm=np.squeeze(vm_initial*np.exp(-adecay*tds)*np.array([[np.cos(thetas)],[np.sin(thetas)]])) #vectorized swimmer's speed
    else:
        vm=np.squeeze(vm_initial*np.array([[np.cos(thetas)],[np.sin(thetas)]]))
    vtotal=vm+vs

    ###### equation that gives the final point as a function of theta\
    if adecay!=0:
        yact_i=rA[1]+vm_initial*np.sin(thetas)*\
            1/adecay*(1-(vm_initial*np.cos(thetas)-D*adecay)/(vm_initial*np.cos(thetas)))+vs[1]*tds
    else:
        yact_i=rA[1]+vtotal[1]*tds
    H=np.abs(yact_i-ytarget)
    yactlist.append(yact_i)
    thetaopt,mindist,index_min=find_func_min(thetas,H)
    print('For a='+str(adecay)+', D='+str(D)+'m, and swimmer''s initial relative speed um='+str(vmi)+'m/s, ')
    print('the needed angle is '+str(np.round(np.rad2deg(thetaopt),4))+' degrees (solution with minimum abs.')
    print('The needed time to cross is t='+str(tds[index_min]))
    print('\n')
    
plt.figure()
for yacti in yactlist:
    plt.plot(np.rad2deg(thetas),yacti)
plt.plot([-75,75],[ytarget,ytarget],'--')
plt.xlim([30,65])
plt.ylim([-10,10])
plt.grid()
plt.xlabel('$\Theta$ (degrees)')
plt.ylabel('y (m)')
#plt.legend(('D=5m','D=10m','D=15m','D=16.5m','target y'))
#plt.title('Deviation for river downforce 1m/s, initial $u_m$=1.7m/s, a=0.03 and parametric bank distance D')


print('For D<16.5m it is possible to reach the target')
