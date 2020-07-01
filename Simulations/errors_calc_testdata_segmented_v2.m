tracks = data_tracks_pred.Tracks_pred;
%bound_molecules_elements = find(Training_classification);
bound_molecules = tracks;%;(bound_molecules_elements,:);
num_bound = length(bound_molecules(:,1));
track_err_arr = zeros(num_bound,3);
coords_sim = cell2mat(track_err_arr_tr_new);
%coords_sim = coords_sim *0.10;
spatial_thresh = 1.0;
for i = 1:num_bound
    find_bound = find(remove_elements == bound_molecules(i,1));
    if isempty(find_bound) == 0
        continue
    end
    element_spatial = find(((bound_molecules(i,17)- coords_sim(:,2)).^2 + (bound_molecules(i,18) - coords_sim(:,3)).^2) < spatial_thresh);
    if isempty(element_spatial) == 0
        track_err_arr(i,:) = [bound_molecules(i,17), bound_molecules(i,18), 1];
        %track_err_arr{i,2} = 1;
    else 
        track_err_arr(i,:) = [bound_molecules(i,17), bound_molecules(i,18),2];
        %track_err_arr{i,2} = 0;
    end
end
%% 
%track_err_arr_2 = track_err_arr(~cellfun('isempty',track_err_arr));
err_values = find(track_err_arr(:,3)= 1);
err_calculator = length(err_values)/num_bound;