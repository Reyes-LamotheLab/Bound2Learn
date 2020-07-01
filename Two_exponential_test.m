function [hratio, pValue_ratio, bic,uMLE]   = Two_exponential_test (data, time_int, pstart, Tboundalphastart,Tboundbetastart, Tbleachstart, Tboundalpha_ub,Tboundbeta_ub,percent_var_bl, alpha_sig, truncpt)

%truncp=min(data);
 pdf_exp_two_kinetics = @(data,p,Tboundalpha,Tboundbeta,Tbleach) p*(((Tbleach + Tboundalpha)/(Tboundalpha*Tbleach))*exp(-(((Tbleach + Tboundalpha)/(Tboundalpha*Tbleach))*(data-truncpt))))+...
     (1-p)*(((Tbleach + Tboundbeta)/(Tboundbeta*Tbleach))*exp(-(((Tbleach + Tboundbeta)/(Tboundbeta*Tbleach))*(data-truncpt))));
pdf_truncexp = @(data_in,mu_trunc) exppdf(data_in,mu_trunc) ./(1-expcdf(truncpt,mu_trunc));
%time_int = 1
t1=[min(data):time_int:max(data)];
%pstart=0.5;
% muStart = quantile(On_time_final,[.25 .75])
%Tboundalphastart=27.5391;
 %Tboundbetastart=Tboundalphastart*0.5;
%Tbleachstart=19.557;

Tboundalpha_lb=0.0001;%Tboundbetastart;
%Tboundalpha_ub = 5400;%Tboundalphastart*1.7;
Tboundbeta_lb = 0.0001;%Tboundbetastart*0.3;
%Tboundbeta_ub = 5400%Tboundalphastart;
%percent_var_bl = 0.20;
per_lb = 1-percent_var_bl;
per_ub = 1 + percent_var_bl;
Tbleach_lb=Tbleachstart*per_lb;
Tbleach_ub=Tbleachstart*per_ub;

start=[pstart Tboundalphastart Tboundbetastart Tbleachstart];
lb = [0 Tboundalpha_lb Tboundbeta_lb Tbleach_lb] ;
ub = [1 Tboundalpha_ub Tboundbeta_ub Tbleach_ub];

options = optimoptions('fmincon','Algorithm','interior-point','MaxFunEvals',10000,...
    'Display','off',...
    'GradObj','on','MaxIter',10000);
x=[0:0.1:max(data)];
 %num_of_bins=ceil(sqrt(numel(data)));
options2 = statset('MaxIter',10000,'MaxFunEvals',10000);

%uMLE2=mle(data, 'pdf',pdf_exp_two_kinetics, 'start',start,'LowerBound',lb,'UpperBound',ub,'options',options2,'OptimFun','fmincon')
 [uMLE,unLogL] = fmincon(@(on) double_nloglike_exp_withgrad ([on(1) on(2) on(3) on(4)],data, truncpt),start,[],[],[],[],lb,ub,[],options);
%[unLogL, ungrad]=double_nloglike_exp_withgrad(uMLE,On_time_final);
uLogL=-unLogL;
uMLE
%uMLE2
%theta2u=uMLE(4)*uMLE(3)/(uMLE(4)+ uMLE(3));
%theta1u=uMLE(2)*uMLE(3)/(uMLE(2)+ uMLE(3));

%F=@(x) uMLE(1)*exp((truncp - x)/theta2u) - uMLE(1)*exp((truncp - x)/theta1u) - exp((truncp - x)/theta2u) + 1;
%[uH,uP,uSTATS] = chi2gof(On_time_final,'cdf',F,'alpha',0.05,'Nparams',3,'Nbins',length(t1),'Emin',5)

 %uacov = mlecov(uMLE,data,'pdf',pdf_exp_two_kinetics);
 %use = sqrt(diag(uacov))
 boot_options=statset('UseParallel',true); 
  param_fit_mle_norm_two=@(Track_duration) mle(Track_duration, 'pdf',pdf_exp_two_kinetics, 'start',start,'Lowerbound',lb, 'Upperbound',ub,'options',options2);
  [boot_CI, bootstat_norm]=bootci(1000,{param_fit_mle_norm_two,data},'type','bca','options',boot_options);
  Tbound_ci = boot_CI; 
   %Tbound_err = std(bootstat_norm);
  disp(Tbound_ci)
   %disp(Tbound_err)
figure,histogram(data,'BinMethod','fd','Normalization','pdf')
%[f_ks, xi] = ksdensity (data,'Support','positive');
%plot(xi,f_ks)
hold on
plot(x,pdf_exp_two_kinetics(x,uMLE(1),uMLE(2),uMLE(3),uMLE(4)))
 %plot(x,pdf_truncexp(x,uMLE(2)));
 %plot(x,pdf_truncexp(x,uMLE(3)));
 %plot(x,exppdf(x,uMLE(4)));
  %plot(x, pdf_truncexp(x,Tbleachstart))
 legend('Track Duration Data','Two-exponential fit')
hold off




lb2 = [1 Tboundalpha_lb Tboundbeta_lb Tbleach_lb] ;
ub2 = [1 Tboundalpha_ub Tboundbeta_ub Tbleach_ub];
start2=[1 Tboundalphastart Tboundbetastart Tbleachstart];

%cMLE=mle(On_time_final, 'pdf',pdf_exp_two_kinetics, 'start',start2,'LowerBound',lb2,'UpperBound',ub2,'options',options2,'OptimFun','fmincon')
 [cMLE,cnLogL] = fmincon(@(on) double_nloglike_exp_withgrad ([on(1) on(2) on(3) on(4)],data,truncpt),start2,[],[],[],[],lb2,ub2,[],options);
%[cnLogL, cngrad]=double_nloglike_exp_withgrad(cMLE,On_time_final);
cMLE
cLogL=-cnLogL;
% [cMLE,cnLogL]=fmincon(@(on) double_nloglike_exp_withgrad([on(1) on(2) on(3) on(4)],On_time_final),start2,[],[],[],[],lb2,ub2,[],options);
%cLogL=-cnLogL;

%theta2c=cMLE(4)*cMLE(3)/(cMLE(4)+ cMLE(3));
%theta1c=cMLE(2)*cMLE(3)/(cMLE(2)+ cMLE(3));

%F2=@(x) cMLE(1)*exp((truncp - x)/theta2c) - cMLE(1)*exp((truncp - x)/theta1c) - exp((truncp - x)/theta2c) + 1;
%[cH,cP,cSTATS] = chi2gof(On_time_final,'cdf',F2,'alpha',0.05,'Nparams',1,'Nbins',length(t1),'Emin',5)
% cacov = mlecov(cMLE,On_time_final,'pdf',pdf_exp_two_kinetics);
% cse = sqrt(diag(cacov))
figure,histogram(data,'BinMethod','fd','Normalization','pdf')
 hold on
plot(x,pdf_exp_two_kinetics(x,cMLE(1),cMLE(2),cMLE(3),cMLE(4)))
 plot(x,pdf_truncexp(x,cMLE(2)));
 plot(x,pdf_truncexp(x,cMLE(3)));
 %plot(x,exppdf(x,cMLE(4)));
 plot(x, pdf_truncexp(x,Tbleachstart))
 legend('Track Duration Data','Track Duration','Tboundalpha','Tboundbeta','Tbleach')
 hold off

[hratio,pValue_ratio,stat_ratio,cValue_ratio] = lratiotest(uLogL,cLogL,1,alpha_sig);

[aic bic]=aicbic([cLogL,uLogL], ...
    [2,4],length(data))

num_of_bins=ceil(sqrt(numel(data)));
F=@(x) 1-exp(-(x-truncpt)*(cMLE(2)+ cMLE(4))/(cMLE(2)*cMLE(4)))
[H,P,STATS] = chi2gof(data,'cdf',F,'alpha',0.005,'Nparams',2,'Nbins',num_of_bins,'Emin',5)

end
%% 
% mu=mean(On_time_final);
% pdf_truncexp = @(On_time_final,mu) exppdf(On_time_final,mu) ./(1-expcdf(truncp,mu));
% start = [mean(On_time_final)]
% [est ci] = mle(On_time_final, 'pdf',pdf_truncexp, 'start',start,'lowerbound',0)
% F=@(x) 1-exp(-(x-truncp)/est);
% [N edges]=histcounts(On_time_final,'BinMethod','fd');
% num_bins=length(N);
% [H,P,STATS] = chi2gof(On_time_final,'cdf',F,'alpha',0.005,'Nparams',1,'Nbins',length(t1),'Emin',5)
