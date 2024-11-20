% function [Vsav,Wsav]=AlievPanfilov2D_RK_Istim(BCL,ncyc,extra,ncells,iscyclic,flagmovie)
% (Ventricular) Aliev-Panfilov model in single-cell with the parameters
% from Goektepe et al, 2010
% Marta, 23/04/2021

% BCL in AU: basic cycle length: time between repeated stimuli (e.g. 30)
% ncyc: number of cycles, number of times the cell is stimulated (e.g. 10)
% extra in AU: time after BCL*ncyc during which the simulation runs (e.g.
% 0)
% ncells is number of cells in 1D cable (e.g. 200)
% iscyclic, = 0 for a cable, = 1 for a ring (connecting the ends of the
% cable - the boundary conditions are not set for the ring yet!)
% flagmovie, = 0 to show a movie of the potential propagating, = 0
% otherwise

% Aliev-Panfilov model parameters 
% V is the electrical potential difference across the cell membrane in 
% arbitrary units (AU)
% t is the time in AU - to scale do tms = t *12.9


ncells=32;

% one of the biggest determinants of the propagation speed
% (D should lead to realistic conduction velocities, i.e.
% between 0.6 and 0.9 m/s)
X = ncells + 2; % to allow boundary conditions implementation
Y = ncells + 2;

% Model parameters
% time step below 10x larger than for forward Euler
dt=0.001; % 10^(-3) ~ 10^(-5) AU, time step for finite differences solver

data = zeros(1000, 2, 32, 32, 2);



% for loop for explicit RK4 finite differences simulation
for i = 1:1000 % for 1000 data points
    V(1:X,1:Y) = 0; % initial V
    W(1:X,1:Y) = 0.01; % initial W
    V(2: X - 1, 2: X - 1) = rand(32, 32);
    data(i, 1, :, :, 1) = V(2: end - 1, 2: end - 1);
    data(i, 1, :, :, 2) = W(2: end - 1, 2: end - 1);

    y=zeros(2,size(V,1),size(V,2));
    y(1,:,:)=V;
    y(2,:,:)=W;
    k1=AlPan(y);
    k2=AlPan(y+dt/2.*k1);
    k3=AlPan(y+dt/2.*k2);
    k4=AlPan(y+dt.*k3);
    y=y+dt/6.*(k1+2*k2+2*k3+k4);
    V=squeeze(y(1,:,:));
    W=squeeze(y(2,:,:));
                  
    data(i, 2, :, :, 1) = V(2: end - 1, 2: end - 1);
    data(i, 2, :, :, 2) = W(2: end - 1, 2: end - 1);
end
save("2D_data_3.mat","data");

function dydt = AlPan(y)
    a = 0.01;
    k = 8.0;
    mu1 = 0.2;
    mu2 = 0.3;
    epsi = 0.002;
    b  = 0.15;
    h = 0.1; % mm cell length
    D = 0.05; % mm^2/UA, diffusion coefficient (for monodomain equation)
    
    V=squeeze(y(1,:,:));
    W=squeeze(y(2,:,:));
    dV=4*D.*del2(V,h);
    dWdt=(epsi + mu1.*W./(mu2+V)).*(-W-k.*V.*(V-b-1));
    dVdt=(-k.*V.*(V-a).*(V-1)-W.*V)+dV;
    dydt(1,:,:)=dVdt;
    dydt(2,:,:)=dWdt;
end
% end
