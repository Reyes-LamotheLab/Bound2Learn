
function [BestModel, numComponents_BIC] = GMM_BIC (data, GMM_models_tested)
options = statset('Display','off','MaxIter',10000,'UseParallel',true);
% gm = fitgmdist(intensities_final,2,'Options',options);
%GMM_models_tested = 5;
BIC = zeros(1,GMM_models_tested);
GMModels = cell(1,GMM_models_tested);
for k = 1:GMM_models_tested
    GMModels{k} = fitgmdist(data,k,'Options',options,'Replicates',30,'SharedCovariance', false);
    BIC(k)= GMModels{k}.BIC;
end

[minBIC,numComponents_BIC] = min(BIC);


BestModel = GMModels{numComponents_BIC};

idx = cluster(BestModel, data);
cluster_array = zeros(length( data),numComponents_BIC);
clust = zeros(length( data),numComponents_BIC);
data_values = cell(numComponents_BIC,1);

    for j = 1:numComponents_BIC
        cluster_array(:,j) = (idx==j);
        clust(:,j) = cluster_array(:,j).* data;
        data_values{j} = nonzeros(clust(:,j));
    end
num_of_bins2 = ceil(sqrt(numel( data))); 
bin_width = (max( data)-min( data))/num_of_bins2;
mean_data = zeros(numComponents_BIC,1);
    figure,
    for i=1:numComponents_BIC
        histogram(data_values{i},'BinWidth',bin_width,'Normalization','count') %might want to incorporate bin width instead
        hold on
        mean_data(i) = mean(data_values{i});
    end
    xlabel('Data (A.U)')
    ylabel('Counts')
    hold off

figure, 
histogram (data, 'BinMethod','sqrt','Normalization','pdf')
hold on;
x= [min(data):0.1:max(data)];
x = x(:);
y = pdf(BestModel, x);

plot(x, y)
hold off;
end
 
