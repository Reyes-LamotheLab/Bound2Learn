function [nll2 ngrad] = double_nloglike_exp_withgrad(params,data,truncp,cens,freq)
%truncp=min(data);
theta1=params(2)*params(4)/(params(2)+params(4));
theta2=params(3)*params(4)/(params(3)+params(4));
ll=zeros(length(data),1);
loll=zeros(length(data),1);
grad=zeros(length(data),4);

p=params(1);
Tbd1=params(2);
Tbd2=params(3);
Tch=params(4);

for i=1:length(data)
r=params(1)*(1/theta1)*exp(-(((data(i,1)-truncp)/theta1)));
s=(1-params(1))*(1/theta2)*exp(-((data(i,1)-truncp)/theta2));
ll(i,1)=(r+s);
loll(i,1) =log(r+s);
   if nargout > 1
       z=data(i,1);
      px(i)=((exp(((Tbd1 + Tch)*(truncp - z))/(Tbd1*Tch))*(Tbd1 + Tch))/(Tbd1*Tch) - (exp(((Tbd2 + Tch)*(truncp - z))/(Tbd2*Tch))*(Tbd2 + Tch))/(Tbd2*Tch))/((p*exp(((Tbd1 + Tch)*(truncp - z))/(Tbd1*Tch))*(Tbd1 + Tch))/(Tbd1*Tch) - (exp(((Tbd2 + Tch)*(truncp - z))/(Tbd2*Tch))*(Tbd2 + Tch)*(p - 1))/(Tbd2*Tch));
      Tbd1x(i)=((p*exp(((Tbd1 + Tch)*(truncp - z))/(Tbd1*Tch)))/(Tbd1*Tch) - (p*exp(((Tbd1 + Tch)*(truncp - z))/(Tbd1*Tch))*(Tbd1 + Tch))/(Tbd1^2*Tch) + (p*exp(((Tbd1 + Tch)*(truncp - z))/(Tbd1*Tch))*(Tbd1 + Tch)*((truncp - z)/(Tbd1*Tch) - ((Tbd1 + Tch)*(truncp - z))/(Tbd1^2*Tch)))/(Tbd1*Tch))/((p*exp(((Tbd1 + Tch)*(truncp - z))/(Tbd1*Tch))*(Tbd1 + Tch))/(Tbd1*Tch) - (exp(((Tbd2 + Tch)*(truncp - z))/(Tbd2*Tch))*(Tbd2 + Tch)*(p - 1))/(Tbd2*Tch));
      Tbd2x(i)=-((exp(((Tbd2 + Tch)*(truncp - z))/(Tbd2*Tch))*(p - 1))/(Tbd2*Tch) - (exp(((Tbd2 + Tch)*(truncp - z))/(Tbd2*Tch))*(Tbd2 + Tch)*(p - 1))/(Tbd2^2*Tch) + (exp(((Tbd2 + Tch)*(truncp - z))/(Tbd2*Tch))*(Tbd2 + Tch)*((truncp - z)/(Tbd2*Tch) - ((Tbd2 + Tch)*(truncp - z))/(Tbd2^2*Tch))*(p - 1))/(Tbd2*Tch))/((p*exp(((Tbd1 + Tch)*(truncp - z))/(Tbd1*Tch))*(Tbd1 + Tch))/(Tbd1*Tch) - (exp(((Tbd2 + Tch)*(truncp - z))/(Tbd2*Tch))*(Tbd2 + Tch)*(p - 1))/(Tbd2*Tch));
      Tchx(i)=((p*exp(((Tbd1 + Tch)*(truncp - z))/(Tbd1*Tch)))/(Tbd1*Tch) - (exp(((Tbd2 + Tch)*(truncp - z))/(Tbd2*Tch))*(p - 1))/(Tbd2*Tch) - (p*exp(((Tbd1 + Tch)*(truncp - z))/(Tbd1*Tch))*(Tbd1 + Tch))/(Tbd1*Tch^2) + (exp(((Tbd2 + Tch)*(truncp - z))/(Tbd2*Tch))*(Tbd2 + Tch)*(p - 1))/(Tbd2*Tch^2) + (p*exp(((Tbd1 + Tch)*(truncp - z))/(Tbd1*Tch))*(Tbd1 + Tch)*((truncp - z)/(Tbd1*Tch) - ((Tbd1 + Tch)*(truncp - z))/(Tbd1*Tch^2)))/(Tbd1*Tch) - (exp(((Tbd2 + Tch)*(truncp - z))/(Tbd2*Tch))*(Tbd2 + Tch)*((truncp - z)/(Tbd2*Tch) - ((Tbd2 + Tch)*(truncp - z))/(Tbd2*Tch^2))*(p - 1))/(Tbd2*Tch))/((p*exp(((Tbd1 + Tch)*(truncp - z))/(Tbd1*Tch))*(Tbd1 + Tch))/(Tbd1*Tch) - (exp(((Tbd2 + Tch)*(truncp - z))/(Tbd2*Tch))*(Tbd2 + Tch)*(p - 1))/(Tbd2*Tch));
  
      
      grad(i,:)=[px(i), Tbd1x(i), Tbd2x(i), Tchx(i)];
    end
end
 
% sumll=sum(ll);
% negsum11=-sum(11);
ngrad=-(sum(grad,1));
nll2=-(sum(loll));
% nll2=real(nll2);
% ngrad2=sum(ngrad) 
end
% p=params(1);
% Tbd1=params(2);
% Tbd2=params(3)
% Tch=params(4);