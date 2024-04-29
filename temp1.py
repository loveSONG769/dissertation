import math
import numpy as np
from scipy import integrate


def main():
    r=0.14
    angle=90
    crank_rpm=1500
    valve_rpm=3000

    crank_angle_spd=angle_spd(crank_rpm)
    valve_angle_spd=angle_spd(valve_rpm)
    valve_rad_spd=radian_spd(valve_rpm)

    valve_opening_area=2*opening_area(angle)
    opening_duration=valve_opening_angle
    opened_duration=duration-2*valve_opening_angle

    x=valve_angle_spd*crank_angle*v_c_ratio
    dx=valve_angle_spd*v_c_ratio/crank_angle

    volumetric_efficiency
    



def angle_spd(speed):
    angle_spd=speed*360/60
    return angle_spd

def radian_spd(speed):
    radian_spd=speed*2*math.pi/60
    return radian_spd

def half_circle(x):
    return (1 - x ** 2) ** 0.5
    
N = 10000
x = np.linspace(-1, 1, N)
dx = 2. / N;
y = half_circle(x)
area =sum( dx * y) #rectangle area 
print (np.trapz(y, x) * 2) #numerical intergrate 
#pi_half, err = integrate.quad(half_circle, -1,1) #intergrate
#print (pi_half * 2)

def opening_area(n):
    R=0.2
    r=0.14
    integrate(math.sqrt(r**2-(r-(n*math.pi*R/180))**2),(n,0,10))
    return opening_area
#need to link the variable to crank





def valve_crank_angle_relationship(crank_rpm,valve_rpm):
    v_c_ratio=valve_rpm/crank_rpm
    
    
    


main()
    
