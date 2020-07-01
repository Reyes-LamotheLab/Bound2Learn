
function [BestModel, numComponents_BIC] = GMM_BIC_ML_log(data, GMM_models_tested, shared_cov)
options = statset('Display','off','MaxIter',10000,'UseParallel',true);
data= log(data);
% gm = fitgmdist(intensities_final,2,'Options',options);
%GMM_models_tested = 5;
BIC = zeros(1,GMM_models_tested);
GMModels = cell(1,GMM_models_tested);
for k = 1:GMM_models_tested
    GMModels{k} = fitgmdist(data,k,'Options',options,'Replicates',200,'SharedCovariance', shared_cov);
    BIC(k)= GMModels{k}.BIC;
end

[minBIC,numComponents_BIC] = min(BIC);


BestModel = GMModels{numComponents_BIC};
figure, 
%[fi,xi]=ksdensity(data,'Support','positive');
histogram(data, 'BinMethod','fd','Normalization','pdf')
hold on
x= [min(data):0.01:max(data)];
x = x(:);
y = pdf(BestModel, x);

plot(x, y)
%plot(xi,fi)
hold off;
end
 
