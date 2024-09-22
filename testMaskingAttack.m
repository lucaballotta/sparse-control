%{
    Check to see if masking attacks exist.

    FROM: "Dual rate control for security in cyber-physical systems", Naghnaeian et al. (2015)
%}

%% Clear the workspace

close all;
clearvars;
clc;

% ------ PARAMETERS ------ %

% Set the system dynamics
A = [0, 1; -2, 3];
B = [0, 0; 2, 1];
C = [1, 0];
% A = [1, 1; 0, 1];
% B = [1, 0; 0, 1];
% C = [3, 4];

% Set the input
syms t
u_1 = exp(t);

% Set the parameters which determine performance
t_end = 10;
n_inter = 1000;

% ------ SCRIPT ------ %

% Create the time-array
t = linspace(0, t_end, n_inter);

% Create the systems
sys = ss(A, B, C, []);
sys_1 = ss(A, B(:, 2), C, []);
sys_2 = ss(A, B(:, 2), C, []);

% Compute the transfer functions
tfunc = tf(sys);
tfunc_1 = tfunc(1);
tfunc_2 = tfunc(2);

% Compute the transfer function from U_1 to U_2
P_zero = -inv(tfunc(2)) * tfunc(1);

% Set the tansfer function varaible
s_tf = tf('s');
syms s_sym

% Compute the LaPlace transform of the input u_1
U_1_sym = laplace(u_1);
% [num, den] = tfdata(U_1_sym);
% U_1 = poly2sym(cell2mat(num), s) / poly2sym(cell2mat(den), s);
% FIXME
U_1 = 1 / (s_tf - 1);

% Compute the output U_2
U_2 = P_zero * U_1;

% Convert to the time-domain
[num, den] = tfdata(U_2);
sys_syms = poly2sym(cell2mat(num), s_sym) / poly2sym(cell2mat(den), s_sym);
u_2 = ilaplace(sys_syms);

% Convert to functions
u_1 = matlabFunction(u_1);
u_2 = matlabFunction(u_2);

% Perform the simulation
y = lsim(sys, [u_1(t); u_2(t)], t);

% Do something

% Do something else

% ------ PRINTING ------ %

% Print something about the system
fprintf('n_x = %d, n_u = %d, n_y = %d\n', size(A, 1), size(B, 2), size(C, 1));

% FIXME: There seems to be a bug with fprintf
disp(0)