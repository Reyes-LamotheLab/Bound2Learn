
[BestModel_D, numComponents_D] = GMM_BIC_ML(Diffusion_norm,2);

idx_D = cluster(BestModel_D,Diffusion_norm);
bound_index = find(idx_D ==1);
tracks_Bd = data_tracks.Segmented_Tracks;
tracks_Bd = tracks_Bd(bound_index,:);
 cluster_array_int = zeros(length( Diffusion_norm), numComponents_D);
    Int_clust = zeros(length(Diffusion_norm), numComponents_D);
    Int_values = cell( numComponents_D,1);

    for j = 1: numComponents_D
        cluster_array_int(:,j) = (idx_D==j);
        Int_clust(:,j) = cluster_array_int(:,j).* Diffusion_norm;
        Int_values{j} = nonzeros(Int_clust(:,j));
    end

    num_of_bins2 = ceil(sqrt(numel( Diffusion_norm))); 
    bin_width = (max(Diffusion_norm)-min(Diffusion_norm))/num_of_bins2;
    mean_intensities = zeros(numComponents_D,1);
    figure,
    for i=1: numComponents_D
        histogram(Int_values{i},'BinWidth',bin_width,'Normalization','count') %might want to incorporate bin width instead
        hold on
        mean_intensities(i) = mean(Int_values{i});
    end
    xlabel('Apparent Diffusion Coefficient (um^2/s)')
    ylabel('Counts')
    hold off   