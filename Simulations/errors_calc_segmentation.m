tracks = data_tracks.Segmented_Tracks;
%bound_molecules_elements = find(Training_classification);
all_molecules = tracks;%(bound_molecules_elements,:);
num_tracks = length(all_molecules(:,1));
track_err_arr_tr = cell(num_tracks,1);
coords_sim = cell2mat(Total_tracks_cell(:,1));
coords_sim = coords_sim *0.10;
spatial_thresh = 2.0;
remove_elements = [];
for i = 1:num_tracks
    element_spatial = find(((all_molecules(i,17)- coords_sim(:,1)).^2 + (all_molecules(i,18) - coords_sim(:,2)).^2) < spatial_thresh);
    if isempty(element_spatial) == 0
        track_err_arr_tr{i,1} = [all_molecules(i,1), all_molecules(i,17), all_molecules(i,18)];
        %track_err_arr_tr{i,2} = 1;
    else 
        remove_elements = [remove_elements;all_molecules(i,1)];
        %continue
         %track_err_arr_tr{i,1} = [all_molecules(i,17), all_molecules(i,18)];
         %track_err_arr_tr{i,1} = all_molecules(i,1);
    end
end
track_err_arr_tr_new= track_err_arr_tr(~cellfun('isempty',track_err_arr_tr));
%% 
% err_values = find(cell2mat(track_err_arr_tr(:,2)));
% err_calculator = length(err_values)/num_tracks;