
function [BestModel, numComponents_BIC] = GMM_BIC_2D (data, GMM_models_tested)
options = statset('Display','off','MaxIter',10000,'UseParallel',true);
Shar_Cov_condition = false;
if length(data(:,1))< 50
    Shar_Cov_condition = true;
end
% gm = fitgmdist(intensities_final,2,'Options',options);
%GMM_models_tested = 5;
BIC = zeros(1,GMM_models_tested);
GMModels = cell(1,GMM_models_tested);
for k = 1:GMM_models_tested
    GMModels{k} = fitgmdist(data,k,'Options',options,'Replicates',15,'SharedCovariance',Shar_Cov_condition);
    BIC(k)= GMModels{k}.BIC;
end

[minBIC,numComponents_BIC] = min(BIC);


BestModel = GMModels{numComponents_BIC};
 cluster_psf = cluster(BestModel, data);
% 
 figure,
% 
 gscatter(data(:,1),data(:,2),cluster_psf);
% hold on 

%f1=@(x,y)pdf(BestModel,[x y]);
%fcontour(f1);
% plot(BestModel.mu(:,1),BestModel.mu(:,2),'kx','LineWidth',2,'MarkerSize',10)
%    
%    title('{\bf Mixture PSFs}')
end
 
