filenames_tracks = uigetfile ('*tracksdata.mat', 'Picks tracks files');
%filenames_tracks = filenames_tracks';
filenames_classification = uigetfile ('*classification*.mat', 'Picks classification files');
%filenames_classification = filenames_classification';
[filenames_spots, dir_spots] = uigetfile('*tif_spots.csv', 'Pick spots files');
%filenames_spots = filenames_spots';
num_files = length(filenames_tracks(1));
%%fix for tracks with multiple spots
for i = 1:num_files
    tracks_seg = importdata(filenames_tracks);
    tracks_seg2 = tracks_seg.Segmented_Tracks;
    class_file = importdata(filenames_classification);
    class_file = class_file';
    elements_bound = find(class_file);
    num_bound = length(elements_bound);
    tracks_bound_ID = tracks_seg2(elements_bound,1);
    spots = csvread((strcat(dir_spots,filenames_spots)));
    tracks_bound = cell(num_bound,2);
        for j = 1:num_bound
            ID = tracks_bound_ID(j);
            spots_find = find(spots(:,1) == ID);
            ID_spots = spots(spots_find,:);
            [~, idx] = sort(ID_spots(:,2),1);
            rev_spots = ID_spots(idx,:);
            tracks_bound{j,1} = rev_spots;
            tracks_bound{j,2} = ID;
        end
        filename_save = strrep(filenames_tracks, 'tracksdata', 'Tracks_D');
        save(filename_save, 'tracks_bound')
end
        
        

