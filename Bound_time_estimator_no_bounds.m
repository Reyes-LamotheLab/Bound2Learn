function [Tbound, Tbound_ci, Tbound_err] = Bound_time_estimator_no_bounds(data, Ttrack, Tbleach,var_bleach, truncpt)
Tbound = (Ttrack * Tbleach)/(Tbleach - Ttrack);
x=0:0.1:max(data);
pdf_truncexp = @(data_in,mu_trunc) exppdf(data_in,mu_trunc) ./(1-expcdf(truncpt,mu_trunc));
pdf_exp_two_kinetics = @(data_in,Tbound_in,Tbleach_in) ((Tbleach_in + Tbound_in)/(Tbound_in*Tbleach_in))*exp(-(((Tbleach_in + Tbound_in)/(Tbound_in*Tbleach_in))*(data_in-truncpt)));
%neg_like_two_exp =  @(data_in,Tbound_in,Tbleach_in, truncpt, cens, freq)-sum(log(((Tbleach_in + Tbound_in)/(Tbound_in*Tbleach_in))*exp(-(((Tbleach_in + Tbound_in)/(Tbound_in*Tbleach_in))*(data_in-truncpt))))); 

boot_options=statset('UseParallel',true); 
options = statset('MaxIter',10000,'MaxFunEvals',10000,'TolBnd',10^-6);

start =[Tbound, Tbleach]; 

lb = [0.1, start(2)*(1-var_bleach)];
ub = [10000, (1 + var_bleach)*start(2)];
%param_fit_mle_norm=@(Track_duration) mle(Track_duration, 'nloglf',neg_like_two_exp, 'start',start,'LowerBound',lb,'UpperBound',ub,'options',options,'Optimfun','fmincon');

param_fit_mle_norm=@(Track_duration) mle(Track_duration, 'pdf',pdf_exp_two_kinetics, 'start',start,'LowerBound',lb,'UpperBound',ub,'options',options,'Optimfun','fmincon'); 
[boot_CI, bootstat_norm]=bootci(1000,{param_fit_mle_norm,data},'type','bca','options',boot_options);
 Tbound_ci = boot_CI(:,1) 
 Tbound_err = std(bootstat_norm(:,1))
 %histogram(bootstat_norm(:,1))
 y1 = exppdf(x,Tbound);
 y2 = pdf_truncexp (x, Tbleach);
 y3 = pdf_truncexp (x, Ttrack);
 figure, histogram (data,'BinMethod','sqrt','Normalization','pdf');
 hold on
plot(x, y1);
 plot(x, y2);
 plot(x, y3);
 legend('Data','Bound Time','Bleach Time', 'Track Duration')
  xlabel('Duration (Seconds)')
  ylabel('Probability Density Function')
end
%  
%   Tbound_norm_mean = mean(bootstat_norm(:,1));
%   std_Tbleach_norm = std(bootstat_norm(:,2));  
%   Tbleach_norm_mean = mean(bootstat_norm(:,2));
 