tracks = data_tracks_pred.Tracks_pred;
%bound_molecules_elements = find(Training_classification);
bound_molecules = tracks;%;(bound_molecules_elements,:);
num_bound = length(bound_molecules(:,1));
track_err_arr = cell(num_bound,1);
coords_sim = cell2mat(track_err_arr_tr_new);
%coords_sim = coords_sim *0.10;
spatial_thresh = 1.0;
remove_count = [];
for i = 1:num_bound
%      find_bound = find(remove_elements == bound_molecules(i,1));
%      if isempty(find_bound) == 0
%          remove_count = [remove_count;1];
%          continue
%      end
    element_spatial = find(((bound_molecules(i,17)- coords_sim(:,2)).^2 + (bound_molecules(i,18) - coords_sim(:,3)).^2) < spatial_thresh);
    if isempty(element_spatial) == 0
        track_err_arr{i,1} = [bound_molecules(i,17), bound_molecules(i,18),1];
        %track_err_arr{i,2} = 1;
    else 
        track_err_arr{i,1} = [bound_molecules(i,17), bound_molecules(i,18),0];
        %track_err_arr{i,2} = 0;
    end
end
%% 
track_err_arr_2 = track_err_arr(~cellfun('isempty',track_err_arr));
tracks_final = cell2mat(track_err_arr_2);
err_values = find(tracks_final(:,3));
err_calculator_accuracy = length(err_values)/num_bound;
num_bound_avail = length(coords_sim(:,1));

missed_bound =  (num_bound_avail - length(err_values))/num_bound_avail;
%length(coords_sim(:,1)); %(num_bound - length(remove_count));