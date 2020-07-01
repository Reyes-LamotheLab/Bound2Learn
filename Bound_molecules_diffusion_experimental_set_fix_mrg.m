filenames_tracks = uigetfile ('*_Q.mat', 'Picks tracks files', 'Multiselect', 'on');
filenames_tracks = filenames_tracks';
% filenames_classification = uigetfile ('*classification_ALL.mat', 'Picks classification files', 'Multiselect', 'on');
% filenames_classification = filenames_classification';
[filenames_spots, dir_spots] = uigetfile('*tif_spots.csv', 'Pick spots files', 'Multiselect', 'on');
filenames_spots = filenames_spots';
num_files = length(filenames_tracks);
%%fix for tracks with multiple spots
for i = 1:num_files
    tracks_seg = importdata(filenames_tracks{i});
    tracks_seg2 = tracks_seg.Tracks_pred;
%     class_file = importdata(filenames_classification{i});
%     class_file = class_file';
    %elements_bound = find(class_file);
    num_bound = length(tracks_seg2(:,1));
    tracks_bound_ID = tracks_seg2(:,1);
    spots = csvread((strcat(dir_spots,filenames_spots{i})));
    tracks_bound = cell(num_bound,2);
        for j = 1:num_bound
            ID = tracks_bound_ID(j);
            spots_find = find(spots(:,1) == ID);
            ID_spots = spots(spots_find,:);
            [~, idx] = sort(ID_spots(:,2),1);
            rev_spots = ID_spots(idx,:);
            %Pick spot closest to initial location 
            track_coords = rev_spots;
            %find initial location
            %x_init = track_coords(1,3);
            %y_init = track_coords(1,4);

            [uni_vals,ele_unique] = unique(track_coords(:,2));
            uni_vals(end + 1) = uni_vals(end) + 1;
            counts_vals = histcounts(track_coords(:,2),uni_vals);
            num_vals = length(counts_vals);
            revised_coords = zeros(num_vals,5);
            for ii = 1:num_vals
                if counts_vals(ii)>1
                    elem_trks = find(track_coords(:,2)==uni_vals(ii));
                    if elem_trks(1) == 1
                        x_mean = mean(track_coords(:,3));
                        y_mean = mean(track_coords(:,4));
                        num_elem_trks = length(elem_trks);
                        rad_dist_corr = zeros(num_elem_trks,1);
                        tracks_dup_corr = track_coords(elem_trks,3:4);
                        for kk = 1:num_elem_trks
                            rad_dist_corr(kk) = sqrt((tracks_dup_corr(kk,1) - x_mean)^2 + (tracks_dup_corr(kk,2) - y_mean)^2);
                        end 
                        min_rad_corr = min(rad_dist_corr);
                        min_corr_elem = find(rad_dist_corr == min_rad_corr);
                        elem_trks_corr = elem_trks(min_rad_elem);
                        revised_coords(ii,:) = track_coords(elem_trks_corr,:);
                    else
                    tracks_dup = track_coords(elem_trks,3:4);
                    num_counts = counts_vals(ii);
                    rad_dist = zeros(num_counts,1);
                    for jj = 1:num_counts
                        rad_dist(jj) = sqrt((tracks_dup(jj,1) - revised_coords(ii-1,3))^2 + (tracks_dup(jj,2) - revised_coords(ii-1,4))^2);
                    end
                    min_rad = min(rad_dist);
                    min_rad_elem = find(rad_dist==min_rad);
                    if length(min_rad_elem)>1
                       min_rad_elem = min_rad_elem(1);
                    end
                    elem_trks_fin = elem_trks(min_rad_elem);
                    %elem_uni = find(track_coords(:,2) == uni_vals(elem_trks_fin));

                    revised_coords(ii,:) = track_coords(elem_trks_fin,:);
                    end
                else
                    elem_trc_inc = find(track_coords(:,2)==uni_vals(ii));
                    revised_coords(ii,:) = track_coords(elem_trc_inc,:); 
                end
                
            end
        

            tracks_bound{j,1} = revised_coords;
            tracks_bound{j,2} = ID;
        end
        %filename_save = strrep(filenames_tracks{i}, '6000trees_LeafSz50_Sam2_BagFr0.50_Preds4_Q', 'Tracks_D_exp_fx_mrg');
        filename_save = strrep(filenames_tracks{i}, 'Q', 'Tracks_D_exp_fx_mrg');
        save(filename_save, 'tracks_bound')
end
        
        

