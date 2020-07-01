%% 
coordinates_filename = uigetfile('*results.mat', 'Pick coordinates file');
filename_Diffusion = uigetfile('*results_D.mat', 'Pick Diffusion file');
tracks_cell = 2; %tracks per cell
data_loc_thresh = 2;
tracks_info = importdata(coordinates_filename);
tracks_info_D = importdata(filename_Diffusion);
num_cells = length(tracks_info);
cell_tr_arr = cell(num_cells*tracks_cell,2);
numb_tracks = tracks_cell*num_cells;
cell_tr_1 = cell(num_cells,2);
coords_cell = {};
D_cell = {};
for i = 1:num_cells
    for j = 1:tracks_cell
        coords = tracks_info(:,:,j,i);
        D_state = tracks_info_D(:,:,j,i);
        bound_states = find(D_state == 1);
        bound_time = length(bound_states);
%         if bound_time ==0
%             continue
%         end
        
%         diff_states = find(D_state == 0);
%         diff_time = length(diff_time);
%         length_bleach = length(bleach_states);
        if bound_time >= data_loc_thresh
               x_mean = mean(nonzeros(coords(bound_states,1)));
               y_mean = mean(nonzeros(coords(bound_states,2)));
                coords_cell = [coords_cell;[x_mean, y_mean]];
                D_cell = [D_cell;D_state];
        end
    end
end

% for i = 1:num_cells
%     for j = 1:tracks_cell
%         
%         %x_mean = mean(nonzeros(coords(:,1)));
%         %y_mean = mean(nonzeros(coords(:,2)));
%         
%     end
% end
%% 

Total_tracks_cell = [coords_cell, D_cell];
       
    
    