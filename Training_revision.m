filename_data_tracks = uigetfile('*tracksdata.mat', 'Pick tracks file');
filename_classification = uigetfile('*classification.mat', 'Pick classification file');
    [filename_img,  path_img] = uigetfile('*.tif*','Pick Image');
    num_images_st = length(imfinfo(strcat(path_img,filename_img)));
    tracks_data_fil = importdata(filename_data_tracks);
    
    Tracks = tracks_data_fil.Segmented_Tracks;
    Trainer = tracks_data_fil.Training;
    class_file = importdata(filename_classification);
    bound_ele = find(class_file);
    num_tracks = length(bound_ele);
    %num_tracks = length(Trainer);
    Training_classification = []; %zeros(num_tracks,1);
    %% 

for i = 1:num_tracks
            track_num = bound_ele(i);
            track_iter = num2str(track_num);
            img_stack = [];
            x_coord_num = Tracks(track_num,17);
            y_coord_num = Tracks(track_num,18);
        x_coord = num2str(x_coord_num);
        y_coord = num2str(y_coord_num);
        rad_img = 10;
        frame_start = num2str(Tracks(track_num,15) + 1);
        frame_end = num2str(Tracks(track_num,16) + 1);
         vec_read = [Tracks(track_num,15) + 1:(Tracks(track_num,16) + 1) + 3];
         if (Tracks(track_num,16) + 1)  + 3 > num_images_st
             vec_read = [Tracks(track_num,15)+1: Tracks(track_num,16) + 1];
         end
        for k = vec_read(1):vec_read(end)
            img_tmp = imread(strcat(path_img,filename_img), k);
            img_tmp_shp = insertShape(img_tmp,'circle',[x_coord_num y_coord_num rad_img], 'Opacity', 0.5,'Color','white');
            img_gray = rgb2gray(img_tmp_shp); 
            img_stack(:,:,k - (vec_read(1)-1)) = img_gray;
         %img_stack(:,:,k - (vec_read(1)-1)) = imread(strcat(path_img,filename_img), k);
         %img_stack(:,:,k - (vec_read(1)-1)) = mat2gray(img_stack(:,:,k - (vec_read(1)-1)));
         %img_stack(:,:,k - (vec_read(1)-1)) = insertShape(img_stack(:,:,k - (vec_read(1)-1)),'circle',[x_coord_num y_coord_num rad_img]);
        end
        
            %waitfor(user_input)
        
         %img_stack=mat2gray(img_stack);
         img_stack = uint16(img_stack);
         h1= implay(img_stack,5);
         
         
         h1.Visual.setPropertyValue('UseDataRange',true);
         h1.Visual.setPropertyValue('DataRangeMin',120);
         h1.Visual.setPropertyValue('DataRangeMax',220);
         h1.Visual.ColorMap.MapExpression = 'gray';
         h1.Parent.Position = [300 100 700 700];
        
         quest_tr = MFquestdlg ( [ 0.6 , 0.1 ] , 'Classification of Molecule', num2str(i),'Noise', 'Bound', 'Noise');
        switch quest_tr
            case 'Noise'
                class_input = 0;
            case 'Bound'
                class_input = 1;
        end
        if isempty(quest_tr) == 1
            disp('Stopped')
            break
        end
        
          
        
        class_file(track_num) = class_input;%str2num(user_input{1});
        close(h1)
end
    %% 

    filename_save = strrep(filename_data_tracks, 'tracksdata','classification_revised');
    input_save = inputdlg('Save?','Saving Classification',[1 50], {'Y'});
    if input_save{1} == 'Y' | input_save{1} =='y'
        save(filename_save,'class_file');
    end