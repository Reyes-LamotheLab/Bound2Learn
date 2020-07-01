%%%No speed factor compensation

filenames_images = uigetfile('*binary_auto.tif', 'Pick Binary Image Files', 'Multiselect', 'on');
filenames_spots = uigetfile('*spots.csv', 'Pick spots.csv', 'Multiselect', 'on');
filenames_tracks = uigetfile('*tracks.csv', 'Pick tracks.csv', 'Multiselect', 'on');
num_files_images = length(filenames_images);
user_input_tracks = inputdlg({'What was the time interval','Min # of localizations for track',' # of localizations for Intensity','Intensity Thresh', 'Track Window' , 'Spot Thresh'},'Tracks Information',...
    [1 50; 1 50; 1 50; 1 50; 1 50; 1 50;],{'1','4','4', '500', '3', '1.5'});
folder_save = strcat('AnalysisMLnew_S','_','Time_Int',num2str(user_input_tracks{1}),'_',num2str(user_input_tracks{4}),'_','Trkwd_',num2str(user_input_tracks{5}),'_','datpt_',num2str(user_input_tracks{2}),'_','Spt_Thr_', num2str(user_input_tracks{6}));
mkdir(folder_save);
time_scale = str2num(user_input_tracks{1});
data_point = str2num(user_input_tracks{2});

data_point_intensity = str2num(user_input_tracks{3});
spt_thresh = str2num(user_input_tracks{6});
for j = 1:num_files_images
bin_image = imread(filenames_images{1,j});
[row, column, v] = find (bin_image ==1);
%% 

filename_spot = filenames_spots{1,j};
filename_track = filenames_tracks{1,j};
Table_Track = csvread(filename_track);
Table_Spot = csvread(filename_spot);


%% 

Quality_Tracks = Table_Track;%.data;
Quality_Tracks_seg = zeros(length(Quality_Tracks(:,1)),18);
for i = 1:length(Quality_Tracks(:,1))
    x_coord = round(Quality_Tracks(i,17));
    y_coord = round(Quality_Tracks(i,18));
    row_find = find (row(:,1) == y_coord);
    if isempty(row_find) ==1
        continue
    end
    if ismember (x_coord, column(row_find,1)) == 1
         Quality_Tracks_seg (i,:) = Quality_Tracks(i,:);
    else 
        continue
    end
  
    
end
save_name_tracks_seg = strrep(filename_track, 'tracks.csv', 'seg.mat');   

%% 
Quality_Tracks_seg = Quality_Tracks_seg(any(Quality_Tracks_seg,2),:);


thresh_find = find (Quality_Tracks_seg(:,2)<data_point);
Quality_Tracks_seg(thresh_find,:) = [];


%% 
intensities_track = zeros(length(Quality_Tracks_seg(:,1)),1);
spot_widths_track = zeros(length(Quality_Tracks_seg(:,1)),1);
%SNR_track = zeros(length(Quality_Tracks_seg(:,1)),1);
Track_mate_training = zeros(length(Quality_Tracks_seg(:,1)),13);
spot_dat = Table_Spot;%.data;
Quality_Tracks_Seg2 = zeros(length(Quality_Tracks_seg(:,1)),18);
for i = 1:length(Quality_Tracks_seg(:,1))
    ID = Quality_Tracks_seg(i,1);
    
    ID_find = find(spot_dat(:,1)==ID);
    %spot_dat = Table_Spot.data;
    ID_spots = spot_dat(ID_find,:);
    [~, idx] = sort(ID_spots(:,2),1);
    rev_spots = ID_spots(idx,:);
    intensities_track_spot = rev_spots(:,5);

    mean_track_intensity_all = mean(intensities_track_spot(1:end,1));

    max_track_intensity = max(intensities_track_spot(1:end,1));

    if Quality_Tracks_seg(i,2)/(Quality_Tracks_seg(i,14)+1)>spt_thresh
        continue
    else
    Track_mate_training(i,1) = Quality_Tracks_seg(i,3);
    Track_mate_training(i,13) = max_track_intensity;

    
    Track_mate_training(i,2:6) = Quality_Tracks_seg(i,4:8);
    Track_mate_training(i,7:11) = Quality_Tracks_seg(i,9:13);
    Track_mate_training(i,12) = mean_track_intensity_all;
    Quality_Tracks_Seg2(i,:) = Quality_Tracks_seg(i,:);

    end
end
Quality_Tracks_Seg2 = Quality_Tracks_Seg2(any(Quality_Tracks_Seg2,2),:);
Track_mate_training = Track_mate_training(any(Track_mate_training,2),:);

save_name_tracks = strrep(filename_track, '.csv', 'data.mat');

data_tracks = struct ('Segmented_Tracks',Quality_Tracks_Seg2, 'Training', Track_mate_training);
save(strcat(folder_save,'/',save_name_tracks), 'data_tracks')
end

    
    
%% 


