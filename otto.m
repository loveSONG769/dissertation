clc
clear all
close all

%parameters 
V=0.2;                    %displacement 
N=1;                      %cylinder number
n=4000;                   %speed
CR=9.4;                  %CR
N_carbon=582/660;                %S/B
MCH_e=0.89;               %ME
N_carbon=8;              %HYDRO-CARBON C 
N_hydrogen=18;             %HYDRO CARBON H
AFR=14.7;                  %AFR
burn_e=1;               %burn efficiency
K=1.35;                   %specific heat ratio
R=0.287;                  %gas constant         
CO_c_v=0.821;                %specific heat constant v    
CO_c_p=R+CO_c_v;                %specific heat constant p 
P_0=103;                  %atm p                  
T_0=55;                   %initial temp
rho_a=P_0/(R*(293.15));   %air density
fuel_heat=((14600*(N_carbon*12)/((N_carbon*12)+(N_hydrogen)))+(62000*(N_hydrogen*0.68)/((N_carbon*12)+(N_hydrogen))))*2.326;    %Fuel Heat
SAFR=(34.56*(N_carbon/N_hydrogen+4))/(12.011+(1.008*N_carbon/N_hydrogen));             %Stoichiometric AFRR
lambda=AFR/SAFR;
eta_c=1; 

V_d=(V/N)/1000;                 %DV
V_c=V_d/(CR-1);                %CV
B=((4*V_d/pi*N_carbon)^(1/3))*100;     %B 
S=B*N_carbon;                          %S 
A_p=(pi*(B^2))/4 ;              %Piston top area

%1-2compression
T_1=(T_0)+273;          %temp         
P_1=P_0;                %pressure       
V_1=V_d+V_c;                    %volume
M_m=(P_1*V_1)/(R*T_1);   %total
M_a=(AFR/(AFR+1))*M_m; %air 
M_f=(1/(AFR+1))*M_m;   %fuel
%2-3combustion
T_2=(T_1)*((CR)^(K-1));    
P_2=(P_1)*((CR)^K);     
V_2=V_1/CR; 
Q_12=0;
W_12=M_m*CO_c_v*(T_1-T_2);
Q_23=M_f*fuel_heat*eta_c;     
W_23=0;
%3-4expansion
T_3=((fuel_heat*eta_c)/((AFR+1)*CO_c_v))+T_2;   
P_3=(P_2/T_2)*T_3;             
V_3=V_2;                   
T_4=T_3*((1/CR)^(K-1));              
P_4=P_3*((1/CR)^K);             
V_4=(M_m*R*T_4)/P_4; 
Q_34=0;
W_34=M_m*CO_c_v*(T_3-T_4);
%4-5exhaust
T_5=T_1;                          
P_5=P_1;                       
V_5=V_1;                     
T_6=T_0;                        
P_6=P_0;                          
V_6=V_2;                           
Q_45=M_m*CO_c_v*(T_5-T_4);  
W_45=0;





%result
eta_t=W_12+W_34/M_f*fuel_heat*eta_c;                             %thermal efficiencN_hydrogen
W_idat=W_12+W_34*N*(n/(60*2));                    %indicated power
W_bdat=MCH_e*W_12+W_34*N*(n/(60*2));                    %brake power

a1=['thermal efficiencN_hydrogen: ',num2str(eta_t)];
    disp(a1)   
a5 = ['indicated power is ',num2str(W_idat), ' KW under ',num2str(n), ' rpm '];
    disp(a5)
a6 = ['brake power is ',num2str(W_bdat), ' KW under ',num2str(n), ' rpm '];
    disp(a6)


%P-V
title('P-V Diagram','FontSize', 20 , 'FontName', 'Helvetica','FontWeight','bold');xlabel('Volume (m^3)','FontSize', 14 , 'FontName', 'Helvetica','FontWeight','bold');ylabel('Pressure (kPa)','FontSize', 14 , 'FontName', 'Helvetica','FontWeight','bold');
hold on
V_12=linspace(V_1,V_2,100);
P_12=((V_1./V_12).^K)*P_1;
plot(V_12,P_12);
V_23=[V_2 V_3];
P_23=[P_2 P_3];
plot(V_23,P_23);
V_34=linspace(V_3,V_4,100);
P_34=((V_3./V_34).^K)*P_3;
plot(V_34,P_34);
V_41=[V_4 V_1];
P_41=[P_4 P_1];
plot(V_41,P_41);
V_56=[V_5 V_6];
P_56=[P_5 P_6];
plot(V_56,P_56);
hold off
text(V_1,P_1, '1,5' ,  'FontSize' ,12)
text(V_2,P_2, '2' ,  'FontSize' ,12)
text(V_3,P_3, '3' ,  'FontSize' ,12)
text(V_4,P_4, '4' ,  'FontSize' ,12)
text(V_6,P_6, '0,6' ,  'FontSize' ,12)


