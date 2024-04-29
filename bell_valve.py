import math
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


def main():
    r_vec=(0.118,0.148,0.165)  #0.12,0.145,0.165
    R_vec=(0.32924,0.40885,0.45943)
   
    N = 100
    N2 = 50
    Z = []
    crank_vec=np.linspace(800,4000,N2)
    
    for m in range(0,len(crank_vec)):
        s_total_max_v=0
        s_total=0
        z_result = [0]*(N+1)
        #print("LENLEN=", len(r_vec))
        for i in range(0,len(r_vec)):
            crank_rpm=crank_vec[m]
            valve_rpm=3000
            r=r_vec[i]
            R=R_vec[i]
            valve_opening=0
            valve_closing=207
            v_open_duration=valve_closing-valve_opening
            v_spd = valve_rpm
            c_spd = crank_rpm
            v_v=v_spd*2*math.pi*R/60 #valve linear speed
            motion_t=2*r/v_v   #valve motion duration
            a_c=c_spd*360/60  #crank angle speed
            theta_0=a_c*motion_t  #crank angle duration for full open valve
            #print("theta0= ", theta_0)
            s_max_v=math.pi*r*r   #max valve opening area
            d_theta=v_open_duration / N
            area_rec=s_max_v*(v_open_duration-2*theta_0)  #rectangular area
            N1 = math.floor(theta_0/d_theta)
            theta = [0]*N1
            for k in range(0,N1):
                theta[k]=k*d_theta
            theta = np.asarray(theta)
            z=s_fan(theta,a_c,v_v,r)
            #print("theta= ",theta)
            #print("z = ",z)
            
            
            s_motion=2*sum(d_theta*z)  #motion area
            for k in range(0,N1):
                z_result[k] = 4*z[k]+z_result[k]


            s_total+= s_motion+area_rec
            s_total_max_v+=s_max_v
        #print("theta_0 = ",theta_0)
        for k in range(0,N1):
                z[k] = z_result[k]
        z_final = list(z) + list([4*s_total_max_v]*(N-2*N1+1)) + list(z[::-1])
        
        #print("z_final = ",z_final)
        #print("111",4*s_total)
        #print("222",4*s_total_max_v)
        Z =  np.concatenate((Z,z_final))
    #print("Z = ",Z)
    Z = Z.reshape((N2,N+1))
    x = [0]*(N+1)
    for j in range(0,N+1):
        x[j] = d_theta*j
    #print("x = ",x)
    #print("y_result = ",z_final)
    #print("d_theta = ",d_theta)
    #print("length = ",N)
    #ax.plot(x, z_final)
    X, Y = np.meshgrid(x, crank_vec)
    #print("X",X)
    #print("Y",Y)
    #print("Z",Z[25,:])
    ax.plot_surface(X, Y, Z)
    plt.show()



    #####VE
    index=25
    Pa=101000 #Pa
    T=295 #k
    R=287 #j/(kg.K)
    rho=Pa/T/R
    gamma=1.35 #or 1.4
    cu=math.sqrt(gamma*R*T) #sound speed
    gamma_coef=(2/(gamma+1))**((gamma+1)/(2*(gamma-1)))
    Af=Z[index,:] #z_final
    #print("AF= ", Af)
    mass_f_r=rho*Af*cu*gamma_coef
    nr=2 #for 4 stroke
    Vd=0.2 #200 cc
    Ne=crank_vec[index]
    VE=mass_f_r*nr/rho/Vd/Ne
    #print("Ne= ", Ne)
    #print("ma= ", mass_f_r)
    
    #print("VE= ", VE)

    plt.figure(1)
    plt.plot(x,VE)
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #plt.figure(2)
    #ax.plot_surface(X, Y, Z)

    plt.show()




    



def s_fan(theta,a_c,v_v,r):
    t=theta/a_c #valve opened time
    d_valve_open=t*v_v/2  #valve opened distance 
   # print("d_valve_open = ",d_valve_open)
    #print("temp acos = ",np.arccos((r-d_valve_open)/r))
    #print("temp_dot = ",(np.dot(np.sqrt(r*r-(r-d_valve_open)**2),(r-d_valve_open))))
    #print("sqrt = ",np.sqrt(r*r-(r-d_valve_open)**2))
    #print("r-d = ",(r-d_valve_open))
    s_fan=np.arccos((r-d_valve_open)/r)*r*r-(np.multiply(np.sqrt(r*r-(r-d_valve_open)**2),(r-d_valve_open)))  #截圆面积
    return 2*s_fan
    


def area(c_spd,v_spd,r,v_open_duration,R):
    v_v=v_spd*2*math.pi*R/60 #valve linear speed
    motion_t=2*r/v_v   #valve motion duration
    a_c=c_spd*360/60  #crank angle speed
    theta_0=a_c*motion_t  #crank angle duration for full open valve
    s_max_v=math.pi*r*r   #max valve opening area
    area_rec=s_max_v*(v_open_duration-2*theta_0)  #rectangular area
    N = 30000
    theta=np.linspace(0,theta_0,N)
    y=s_fan(theta,a_c,v_v,r)
    d_theta=theta_0 / N
    s_motion=2*sum(d_theta*y)  #motion area

    print("v_v = ",v_v)
    print("motion_t = ",motion_t)
    print("a_c = ",a_c)
    print("theta_0 = ",theta_0)
    print("v_v = ",v_v)
    print("s_max_v = ",s_max_v)
    print("area_rec = ",area_rec)
    print("d_theta = ",d_theta)
    for i in range(1,min(N,10)):
        print("y(",i,") = ",y[i])

    s_result =[0,0,0,0]
    s_result[0] = s_motion+area_rec
    s_result[1] = s_max_v
    s_result[2] = d_theta
    s_result[3] = theta_0
    return s_result

main()
