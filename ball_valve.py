import math
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def main():
    r_port = 0.014/25.4*1000
    R_valve = 0.038/25.4*1000

    N = 3000
    N2 = 2000
    Z = []
    Zz = []
    crank_vec = np.linspace(800, 4000, N2)

    
    for m in range(0, len(crank_vec)):
        s_total_max_v=0
        s_total=0
        z_result = [0]*(N+1)
        
        s_total_max_v2=0
        s_total2=0
        z_result2 = [0]*(N+1)

        crank_rpm = crank_vec[m]
        valve_rpm = 3000 #valve motor speed
        valve_opening = 0 #valve opening crankshaft angle degree
        valve_closing = 207 #valve closing crankshaft angle degree
        v_open_duration = valve_closing-valve_opening #valve opening duration
        v_spd = valve_rpm #valve rotating speed
        c_spd = crank_rpm
        v_v = v_spd*2*math.pi*R_valve/60 #valve port surface linear speed
        motion_t=2*r_port/v_v   #valve surface linear motion duration
        a_c=c_spd*360/60  #crank angle speed
        theta_0=a_c*motion_t  #crank angle duration for full open valve
        #print("c a duration = ", theta_0)
        s_max_v=math.pi*r_port*r_port   #max valve opening area
        #print("max valve opening area = ", s_max_v)
        d_theta=v_open_duration / N
        area_rec=s_max_v*(v_open_duration-2*theta_0)  #rectangular area
        N1 = math.floor(theta_0/d_theta)
        theta = [0]*N1
        for k in range(0,N1):
            theta[k]=k*d_theta
        theta = np.asarray(theta)
        z=s_fan(theta,a_c,v_v,r_port)
        zz=s_moon(theta,a_c,v_v,r_port)

        #print("zz = ",zz)
        
        
        s_motion=2*sum(d_theta*z)  #motion area
        for k in range(0,N1):
            z_result[k] = z[k]+z_result[k]
        #print("z_result = ", z_result)    
        s_total+= s_motion+area_rec
        s_total_max_v+=s_max_v
    #print("theta_0 = ",theta_0)
        for k in range(0,N1):
            z[k] = z_result[k]
        #print("z = ", z)
        
        z_final = list(z) + list([s_total_max_v]*(N-2*N1+1)) + list(z[::-1])
        #print("z_final = ",z_final)
        #print("111",4*s_total)
        #print("222",4*s_total_max_v)
        Z =  np.concatenate((Z,z_final))
        #print("Z = ",Z.shape)
    
        s_motion2=2*sum(d_theta*zz)  #motion area
        for k in range(0,N1):
            z_result2[k] = zz[k]+z_result2[k]
        #print("z_result = ", z_result)    
        s_total2+= s_motion2+area_rec
        s_total_max_v2+=s_max_v
    #print("theta_0 = ",theta_0)
        for k in range(0,N1):
            zz[k] = z_result2[k]
        #print("z = ", z)
        
        z_final2 = list(zz) + list([s_total_max_v2]*(N-2*N1+1)) + list(zz[::-1])
        #print("z_final = ",z_final)
        #print("111",4*s_total)
        #print("222",4*s_total_max_v)
        Zz =  np.concatenate((Zz,z_final2))
        #print("Z = ",Z.shape)
    #print("z_final2 shape = ", z_final2.shape)
    Z = Z.reshape((N2,N+1))

    Zz = Zz.reshape((N2,N+1))
    
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
    #print("Z",Z)
    #ax.plot_surface(X, Y, Z)
    ax.plot_surface(X, Y, Zz)
    #ax.view_init(0,90)
    print("max valve opening area = ", s_max_v)
    ax.set_xlabel("Crankshaft Position (degree)")
    ax.set_ylabel("Engine Speed (RPM)")
    ax.set_zlabel("Valve Opening Area (sq.in)")
    plt.show()

    X_2D = X[1800]
    Z_2D = Z[1800]
    Zz_2D = Zz[1800]
    plt.plot(X_2D, Z_2D, label="ciecle port")
    plt.plot(X_2D, Zz_2D, label="spindle port")
    plt.legend(loc="upper left")

    plt.show()



    #####VE
    index=100
    crank_rpm = crank_vec[index]
    print("engine speed = ", crank_rpm)
    Pa=101325 #Pa
    T=295 #k
    R=8.314 #287 #j/(kg.K)
    rho=Pa/T/R
    gamma=1.35 #or 1.4
    cu=math.sqrt(gamma*R*T) #sound speed
    print("Cu = ", cu)
    gamma_coef=(2/(gamma+1))**((gamma+1)/(2*(gamma-1)))
    Af=Zz[index,:] #z_final########33
    Af2=Zz[index,:] #z_final
    print("AF= ", Af)

    Vd=0.2 #displacement volume
    Lc=0.116
    a=0.0582/2
    rc=9.4
    Vc = Vd/(rc-1)
    print("Vc = ", Vc)
    nr=2 #for 4 stroke
    Ne=crank_vec[index]*2*math.pi/60
    print("engine angular speed = ", Ne)
    dV=[0]*(N+1)
    dP=[0]*(N+1)
    dPt=[0]*(N+1)
    dP2=[0]*(N+1)
    dM1=[0]*(N+1)
    dM2=[0]*(N+1)
    choke_con = ((gamma+1)/2)**(gamma/(gamma-1))
    normal_p_con = 0.8*Pa
    print("choke_con = ", choke_con)
    Pd_con=Pa/choke_con
    print("Pd_con = ", Pd_con)
    #print(math.sin(math.radians(30)))
    #volume change
    for j in range(0,N+1):
        #dV[j] = Vd*math.sin(math.radians(x[j]))*(1+math.sin(math.radians(x[j])))*math.pi/2/(((Lc/a)**2-(math.sin(math.radians(x[j])))**2)**0.5)/180
        dV[j] = (1+Lc/a-math.cos(math.radians(x[j]))-((Lc/a)**2-(math.sin(math.radians(x[j])))**2)**0.5)*Vd/2
        #print("dV = ", dV[j])
        dP[j] = -(R*Pa*dV[j]/Vc/20.8)+(R*gamma*T*rho*Af[j]*cu*gamma_coef/Ne/Vc)
        #dPt[j] = (R*gamma*T*rho*Af[j]*cu*gamma_coef/Ne/Vc)
        dPt[j] = -(R*Pa*dV[j]/Vc/20.8)+(R*gamma*T*rho*Af[j]*cu*gamma_coef/Ne/Vc)

        dP2[j] = -(Pa*dV[j]/Vc/20.8)+5*(R*gamma*T*rho*Af[j]*cu*(2*((normal_p_con/Pa)**(2/gamma)-(normal_p_con/Pa)**((gamma+1)/gamma))/(gamma-1))**0.5/Ne/Vc)

        dM1[j] = (R*gamma*T*rho*Af[j]*cu*gamma_coef/Ne/Vc)
        dM2[j] = (R*gamma*T*rho*Af[j]*cu*(2*((normal_p_con/Pa)**(2/gamma)-(normal_p_con/Pa)**((gamma+1)/gamma))/(gamma-1))**0.5/Ne/Vc)
        if abs(dP[j]) >= Pd_con:
            dP[j] = dP2[j]
            dM1[j] = dM2[j]
        dP[j]= dP[j]+Pa
        
    print("dP= ", dP[100])
    print("dPt= ", dPt[100])
    print("dP2= ", dP2[100])
    #plt.plot(x, dV)
    plt.figure(1)
    plt.plot(x, dP)
    plt.xlabel("Crankshaft Position (degree)")
    plt.ylabel("Cylinder Pressure (Pa)")
    plt.show()

    plt.figure(1)
    plt.plot(x, dM1)
    plt.xlabel("Crankshaft Position (degree)")
    plt.ylabel("Total Mass (Pa)")
    plt.show()

    
    mass_f_r=rho*Af*cu*gamma_coef*0.1
    mass_f_r2=rho*Af2*cu*gamma_coef*0.1
    VE1 = [0]*(N+1)
    VE2 = [0]*(N+1)
    for j in range(0,N+1):
        VE1[j]=dM1[j]*0.17/rho/Vd
        VE2[j]=dM2[j]/rho/Vd
    plt.figure(1)
    plt.plot(x,VE1)#, label="circle port")
    #plt.plot(x,VE2, label="spindle port")
    plt.legend(loc="upper left")
    plt.xlabel("Crankshaft Position (degree)")
    plt.ylabel("Volumetric Efficiency (%)")
    plt.show()
    





def s_moon(theta,a_c,v_v,r_port):
    t=theta/a_c #valve opened time
    d_valve_open=r_port-t*v_v/2  #valve opened distance 
   # print("d_valve_open = ",d_valve_open)
    #print("temp acos = ",np.arccos((r-d_valve_open)/r))
    #print("temp_dot = ",(np.dot(np.sqrt(r*r-(r-d_valve_open)**2),(r-d_valve_open))))
    #print("sqrt = ",np.sqrt(r*r-(r-d_valve_open)**2))
    #print("r-d = ",(r-d_valve_open))
    s_fan=np.arccos((r_port-d_valve_open)/r_port)*r_port*r_port-(np.multiply(np.sqrt(r_port*r_port-(r_port-d_valve_open)**2),(r_port-d_valve_open)))  #截圆面积
    s_moon = math.pi*r_port*r_port - 2*s_fan
    return s_moon#2*s_fan

def s_fan(theta,a_c,v_v,r_port):
    t=theta/a_c #valve opened time
    d_valve_open=t*v_v/2  #valve opened distance 
   # print("d_valve_open = ",d_valve_open)
    #print("temp acos = ",np.arccos((r-d_valve_open)/r))
    #print("temp_dot = ",(np.dot(np.sqrt(r*r-(r-d_valve_open)**2),(r-d_valve_open))))
    #print("sqrt = ",np.sqrt(r*r-(r-d_valve_open)**2))
    #print("r-d = ",(r-d_valve_open))
    s_fan=np.arccos((r_port-d_valve_open)/r_port)*r_port*r_port-(np.multiply(np.sqrt(r_port*r_port-(r_port-d_valve_open)**2),(r_port-d_valve_open)))  #截圆面积
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
