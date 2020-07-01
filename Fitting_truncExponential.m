function [est, ci, se] = Fitting_truncExponential (data, interval_time, truncpt) 
%interval_time = 1.0;
mu=mean(data);
%truncpt=min(data);
pdf_truncexp = @(data_in,mu_trunc) exppdf(data_in,mu_trunc) ./(1-expcdf(truncpt,mu_trunc));
start = mu;
[est, ci] = mle(data, 'pdf',pdf_truncexp, 'start',start,'lowerbound',0);
% disp(est)
% disp(ci)
x=0:0.1:max(data);
y=pdf_truncexp(x,est);
y2=exppdf(x,mu);

t1=[min(data):interval_time:max(data)];

nlogL_exp = explike(mu,data);
nlogL_exp_trunc = -sum(log(pdf_truncexp(data,est)));
acov = mlecov(est,data,'pdf',pdf_truncexp);
se = sqrt(diag(acov));


boot_input_dlg = inputdlg('Do Boostrap Sampling?','Error Calculation?',[1 50],{'Y'});
if boot_input_dlg{1} == 'Y' | boot_input_dlg{1} == 'y'    
%OPTION: Bootstrapping
     boot_options=statset('UseParallel',true); 
     param_fit_mle_norm=@(Track_duration) mle(Track_duration, 'pdf',pdf_truncexp, 'start',start,'lowerbound',0);
     [boot_CI, bootstat_norm]=bootci(1000,{param_fit_mle_norm,data},'type','bca','options',boot_options);
     Tbound_ci = boot_CI; 
     Tbound_err = std(bootstat_norm);
%      disp(Tbound_ci)
%      disp(Tbound_err)
     ci = Tbound_ci;
     se = Tbound_err;
end
% [fi,xi]=ksdensity(On_time_final,'Support',[0; 200]);
 %figure,histogram(data,'BinMethod','sqrt','Normalization','pdf');
  %figure,histogram(On_time_final,'BinMethod','fd','Normalization','pdf');
%hold on
% plot(xi,fi)
%plot(x,y,'--r')

%plot(x,y2,'--b')
%legend('Data','MLE Truncated Exponential Fit')
%xlabel('Seconds')
%ylabel('Probability Density Function')

 

 %% 

 [f, g]=ecdf(data);
 x2=min(data):interval_time:max(data);
 [f_ks, x_ks] = ksdensity (data, x2, 'Support','positive', 'Function', 'cdf');
 y3=zeros(length(x2),1);
 y4=zeros(length(x2),1);
  for j=1:length(x2)
      y3(j)=1-exp(-(x2(j)-truncpt)/est);
      y4(j)=1-exp(-(x2(j)/mu));
  end
  figure,
  %subplot(1,2,1);
  %plot(x_ks,f_ks)
  %hold on 
  %plot(x2,y3);
  %plot(x2,y4);
  %legend('ECDF','MLE Truncated Exponential Fit')
  
  %subplot(1,2,2);
  histogram(data,'BinMethod','fd','Normalization','pdf');
  %[f_ks, xi] = ksdensity (data,'Support','Positive');
  %plot(xi,f_ks)
  hold on
% 
  plot(x,y,'--r')
%plot(x,y2,'--b')
  legend('Data','MLE Truncated Exponential Fit')
  xlabel('Track Duration (Seconds)')
  ylabel('Probability Density Function')
%% 
fun=@(x5)(1/est)*exp(-(x5-truncpt)/est);
F=@(x) 1-exp(-(x-truncpt)/est);
[N, edges]=histcounts(data,'BinMethod','fd');
num_bins=length(N);
[H,P,STATS] = chi2gof(data,'cdf',F,'alpha',0.05,'Nparams',1,'Nbins',length(t1),'Emin',5)
%% 
sample_size = length(data);
se_string = strcat('Standard Error:', num2str(se));
est_string = strcat('Estimate of Lifetime:' , num2str(est));
ci_string = strcat('Confidence Interval:','[', num2str(ci(1)),',',num2str(ci(2)),']');
sample_string = strcat('Sample Size:',num2str(sample_size));

waitfor(msgbox({est_string,ci_string,se_string,sample_string},'Analysis Complete'))
end
