close all
clear all

experiment_id='expEllipse';

load(['Logging_' experiment_id '_ref.mat'])

Pc_1 = Pc_iter;
Pcf_hat_1 = Pcf_hat_iter;
Qc_1 = Qc_iter;
Sigma_1 = Sigma_m;
t_1 = tr;

Pc_1(:, 1) = [];
Pcf_hat_1(:, 1) = [];
Qc_1(:, 1) = [];
Sigma_1(:, 1:3) = [];
t_1(end) = [];


for i=1:length(t_1)
  Rc =  quat2rot(Qc_1(:, i));
  z_1(:,i) = 0.1*Rc(:,3);
  pf_hat_1(:, i) = Pc_1(:,i) +  Rc *  Pcf_hat_1(: , i);
end

clear Pc_iter Pcf_hat_iter Qc_iter Sigma_m tr

load(['Logging_' experiment_id '_1.mat'])

Pc_2 = Pc_iter;
Pcf_hat_2 = Pcf_hat_iter;
Qc_2 = Qc_iter;
Sigma_2 = Sigma_m;
t_2 = t;

Pc_2(:, 1) = [];
Pcf_hat_2(:, 1) = [];
Qc_2(:, 1) = [];
Sigma_2(:, 1:3) = [];
t_2(end) = [];

for i=1:length(t_2)
  Rc =  quat2rot(Qc_2(:, i));
  z_2(:,i) = 0.1*Rc(:,3);
  pf_hat_2(:, i) = Pc_2(:,i) +  Rc *  Pcf_hat_2(: , i);
end

clear Pc_iter Pcf_hat_iter Qc_iter Sigma_m t


figure()
plot3(pf_hat_1(1,:), pf_hat_1(2,:), pf_hat_1(3,:),'b');
hold on 
plot3(pf_hat_2(1,:), pf_hat_2(2,:), pf_hat_2(3,:),'r');


for i = 1:100:length(t_1)
  plot3([pf_hat_1(1,i) pf_hat_1(1,i)+z_1(1,i)],[pf_hat_1(2,i) pf_hat_1(2,i)+z_1(2,i)],[pf_hat_1(3,i) pf_hat_1(3,i)+z_1(3,i)],'k')
end

for i = 1:100:length(t_2)
  plot3([pf_hat_2(1,i) pf_hat_2(1,i)+z_2(1,i)],[pf_hat_2(2,i) pf_hat_2(2,i)+z_2(2,i)],[pf_hat_2(3,i) pf_hat_2(3,i)+z_2(3,i)],'m')
end

axis equal


