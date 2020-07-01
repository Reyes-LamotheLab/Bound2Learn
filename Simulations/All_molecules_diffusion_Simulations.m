filenames_tracks = uigetfile ('*tracksdata.mat', 'Picks tracks files');
%filenames_tracks = filenames_tracks';
%filenames_classification = uigetfile ('*classification_ALL.mat', 'Picks classification files');
%filenames_classification = filenames_classification';
[filenames_spots, dir_spots] = uigetfile('*tif_spots.csv', 'Pick spots files');
%filenames_spots = filenames_spots';
num_files = length(filenames_tracks(1));
%%fix for tracks with multiple spots
for i = 1:num_files
    tracks_seg = importdata(filenames_tracks);
    tracks_seg2 = tracks_seg.Segmented_Tracks;
    %class_file = importdata(filenames_classification);
    %class_file = class_file';
    %elements_bound = find(class_file);
    
    tracks_bound_ID = tracks_seg2(:,1);
    num_tracks = length(tracks_bound_ID);
    spots = csvread((strcat(dir_spots,filenames_spots)));
    tracks_all = cell(num_tracks,2);
        for j = 1:num_tracks
            ID = tracks_bound_ID(j);
            spots_find = find(spots(:,1) == ID);
            ID_spots = spots(spots_find,:);
            [~, idx] = sort(ID_spots(:,2),1);
            rev_spots = ID_spots(idx,:);
            tracks_all{j,1} = rev_spots;
            tracks_all{j,2} = ID;
        end
        filename_save = strrep(filenames_tracks, 'tracksdata', 'Tracks_D_All');
        save(filename_save, 'tracks_all')
end
        
        

