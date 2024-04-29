import math
import sympy as sy
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def main():
    r_vec=(0.119,0.148,0.165)  #0.12,0.145,0.165
    R_vec=(0.32924,0.40885,0.45943)
   
    N = int(1000)
    N2 = 50
    Z = []
    crank_vec=np.linspace(800,6000,N2)
    
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
            #print("N1=",N1)
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
    print("rotary valve max opening area=", [4*s_total_max_v])
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



    #####VE
    index=20 #34

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
    print("Ne= ", Ne)
    #print("ma= ", mass_f_r)
    
    #print("VE= ", VE)


    #######original VE
    theta = [0]*(N+1)
    valve_opening=0
    valve_closing=207
    v_open_duration=valve_closing-valve_opening
    d_theta=v_open_duration / N
    
    for l in range(0,int(N/2)):
        theta[l]=l*d_theta
    theta_1 = np.asarray(theta[0:int(N/2)])
    z_o_1=(8.724489572245177*10**(-30))*theta_1**15+(-1.35313642804956*10**(-26))*theta_1**14+(9.45434850116815*10**(-24))*theta_1**13+(-3.93036276539530*10**(-21))*theta_1**12+(1.08112500700193*10**(-18))*theta_1**11+(-2.06876504858949*10**(-16))*theta_1**10+(2.81579223001290*10**(-14))*theta_1**9+(-2.73591172698436*10**(-12))*theta_1**8+(1.87301430848427*10**(-10))*theta_1**7+(-8.74677136652550*10**(-9))*theta_1**6+(2.62475776957364*10**(-7))*theta_1**5+(-4.57848868734980*10**(-6))*theta_1**4+(4.04649013154706*10**(-5))*theta_1**3+(-0.000135047970803517)*theta_1**2+(0.000360469542360491)*theta_1**1+(4.54797548741340*10**(-5))
    #z_o_1=z_o_1*(-0.001+3.2437*z_o_1/1.3-1.794 *(z_o_1/1.3)**2)
    s_o_1=z_o_1*math.pi*1.3
    
    for m in range(int(N/2),N+1):
        theta[m]=m*d_theta
    theta_2 = np.asarray(theta[int(N/2):N+1])
    z_o_2=(1.34057576152852*10**(-29))*theta_2**14+(-1.97553920205812*10**(-26))*theta_2**13+(1.46162855199868*10**(-23))*theta_2**12+(-7.37941269758326*10**(-21))*theta_2**11+(2.74879288704423*10**(-18))*theta_2**10+(-7.49437859202936*10**(-16))*theta_2**9+(1.45815618211744*10**(-13))*theta_2**8+(-1.97909977340450*10**(-11))*theta_2**7+(1.82797615744295*10**(-9))*theta_2**6+(-1.10629478732787*10**(-7))*theta_2**5+(4.10232387967087*10**(-6))*theta_2**4+(-8.27649925543051*10**(-5))*theta_2**3+(0.000792266345094215)*theta_2**2+(-0.00231720481869556)*theta_2**1+(0.000147178187343951)
    #z_o_2=z_o_2*(-0.001+3.2437*z_o_2/1.3-1.794 *(z_o_2/1.3)**2)
    s_o_2=z_o_2*math.pi*1.3
    
    s_o=np.append(s_o_1,s_o_2)
    #C_f=-0.001+3.2437*np.append(z_o_1,z_o_2)/1.3-1.794 *(np.append(z_o_1,z_o_2)/1.3)**2
    #s_o=s_o*C_f
    #print("s_o=", s_o)
    mass_f_r_o=rho*s_o*cu*gamma_coef
    VE_o=mass_f_r_o*nr/rho/Vd/Ne
    s_max_v_o=0.322*1.3*np.pi
    print("poppet valve max opening area=", s_max_v_o)
    #print("LEN=", len(VE_o))
    #print("z_o=", z_o)
    
    #print("s_o=",s_o)



    

    ##total asperated air

    ##original
    a=sy.symbols('a')
    f_1=(8.724489572245177*10**(-30))*a**15+(-1.35313642804956*10**(-26))*a**14+(9.45434850116815*10**(-24))*a**13+(-3.93036276539530*10**(-21))*a**12+(1.08112500700193*10**(-18))*a**11+(-2.06876504858949*10**(-16))*a**10+(2.81579223001290*10**(-14))*a**9+(-2.73591172698436*10**(-12))*a**8+(1.87301430848427*10**(-10))*a**7+(-8.74677136652550*10**(-9))*a**6+(2.62475776957364*10**(-7))*a**5+(-4.57848868734980*10**(-6))*a**4+(4.04649013154706*10**(-5))*a**3+(-0.000135047970803517)*a**2+(0.000360469542360491)*a**1+(4.54797548741340*10**(-5))
    f_2=(1.34057576152852*10**(-29))*a**14+(-1.97553920205812*10**(-26))*a**13+(1.46162855199868*10**(-23))*a**12+(-7.37941269758326*10**(-21))*a**11+(2.74879288704423*10**(-18))*a**10+(-7.49437859202936*10**(-16))*a**9+(1.45815618211744*10**(-13))*a**8+(-1.97909977340450*10**(-11))*a**7+(1.82797615744295*10**(-9))*a**6+(-1.10629478732787*10**(-7))*a**5+(4.10232387967087*10**(-6))*a**4+(-8.27649925543051*10**(-5))*a**3+(0.000792266345094215)*a**2+(-0.00231720481869556)*a**1+(0.000147178187343951)
    f_1_l=0
    f_1_u=(N/2-1)*d_theta
    f_2_l=f_1_u
    f_2_u=N*d_theta
    r_1=sy.integrate(f_1,(a,f_1_l,f_1_u))
    r_2=sy.integrate(f_2,(a,f_2_l,f_2_u))
    r=r_1+r_2
    r=r*math.pi*1.3
    r=rho*r*cu*gamma_coef
    r1=r*nr/rho/Vd/Ne
    print("r=",r1)

    ##new
    r2=np.trapz(VE, x)
    print("r=",r2)

    ##performance ratio
    r_p=r1/r2
    print("r_p=",r_p)


    




    
    ###old valve opening
   # plt.plot(x,s_o,'r', label='rotary valve')
   # plt.ylabel('valve opening area')
   # plt.xlabel('crankshaft angle')
   # plt.legend()
    
    ###new valve opening
   # fig = plt.figure()
   # ax = fig.add_subplot(111, projection='3d')
   # ax.plot_surface(X, Y, Z)

    ###VE
    ax = plt.subplot(111)
    ax.plot(x, VE,'b', label='rotary valve')
    ax.plot(x, VE_o,'r', label='poppet valve')
    plt.ylabel('valve efficiency')
    plt.xlabel('crankshaft angle')
    ax.legend()

    
    plt.show()

    

    
    
    
    
    


    



def s_fan(theta,a_c,v_v,r):
    t=theta/a_c #valve opened time
    d_valve_open=t*v_v/2  #valve opened distance 
    #print("d_valve_open = ",d_valve_open)
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

def original_VE(theta):
    z_o=(8.724489572245177*10**(-30))*theta**15+(-1.35313642804956*10**(-26))*theta**14+(9.45434850116815*10**(-24))*theta**13+(-3.93036276539530*10**(-21))*theta**12+(1.08112500700193*10**(-18))*theta**11+(-2.06876504858949*10**(-16))*theta**10+(2.81579223001290*10**(-14))*theta**9+(-2.73591172698436*10**(-12))*theta**8+(1.87301430848427*10**(-10))*theta**7+(-8.74677136652550*10**(-9))*theta**6+(2.62475776957364*10**(-7))*theta**5+(-4.57848868734980*10**(-6))*theta**4+(4.04649013154706*10**(-5))*theta**3+(-0.000135047970803517)*theta**2+(0.000360469542360491)*theta**1+(4.54797548741340*10**(-5))
    s_o=z_o*math.pi*1.3
    
    return s_o







































    

main()
