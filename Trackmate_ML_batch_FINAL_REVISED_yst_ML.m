%% **
function Trackmate_ML_batch_FINAL_REVISED_yst_ML
list_ML = {'Learning','Training','Training Combiner','Classifier 1','Analysis','Concatenate','Classifier 2'};  
ML_input = listdlg('PromptString','What do you want to do?','ListString',list_ML,'SelectionMode','single');

if ML_input == 1
    Learner_model
elseif ML_input == 2
    Training_data  
elseif ML_input == 3
    Training_combiner
elseif ML_input == 4
    Classifier_1
elseif ML_input == 5
    Trackmate_Analysis_ML_FINAL
elseif ML_input == 6
    Concatenate_tracks
elseif ML_input == 7
    Classifier_2
end
end
%% 
function Classifier_2
 
    [filename_classifier,  path_classifier] = uigetfile('*mod*.mat','Pick Classifier file');
    pred_used_class = inputdlg({'Mean Speed', 'Max Speed', 'Min Speed','Median Speed','Max Quality',}, 'Predictors', ...
        [1 100; 1 100; 1 100; 1 100; 1 100], {'1','1','1','1','1'});
    pred_used_2_class = str2num(cell2mat(pred_used_class));
    pred_var_class = find(pred_used_2_class);   
    classifier = importdata(strcat(path_classifier,filename_classifier));
    filename_new_data = uigetfile('*NOQ*.mat','Pick new data files', 'Multiselect', 'on');
    filename_new_data = filename_new_data';
    filename_tracks_combined = uigetfile('*tracks_training_combined_2*', 'Pick tracks combined file');
    Fraction_factors = inputdlg('Quality Factor', 'Fraction Factors', [1 100],{'3000'});
    tracks_combined = importdata(filename_tracks_combined);
    max_quality_values = (tracks_combined (:, 5));
    [Best_q, comp_q] = GMM_BIC_ML_log(max_quality_values,2, true);
    mu_q = Best_q.mu;
     if length (mu_q) > 1
      mu_q = unique(mu_q);
      mean_q = mu_q(2);
     else 
         mean_q = mu_q;
     end
      disp(mean_q);
      GM_fit_q = inputdlg('Log(MeanQ)', 'Fit Correction', [1 100], {num2str(mean_q)}); 
      mean_q = str2num(GM_fit_q{1});
      mean_q = exp(mean_q);
      
    for i = 1:length(filename_new_data)

        new_data = importdata(filename_new_data{i});
        new_data_2 = new_data.TrainingQ;      
        frac_quality = str2num(Fraction_factors{1})/mean_q;
        new_data_2(:,5) = new_data_2(:,5)*frac_quality;
        new_data_3 = new_data_2;
        new_data_2 = new_data_2(:,pred_var_class);
        if isempty(new_data_2) == 1
            continue
        end
        tracks = importdata(filename_new_data{i});
        tracks_2 = tracks.Tracks_pred; 
        new_data_var = new_data_2;%(:,1:5);
        prediction_class = predict(classifier, new_data_var);
        if iscell(prediction_class) == 1
            prediction_class = cell2mat(prediction_class);
            prediction_class = str2num(prediction_class);
        end
        pred_isolate  = find(prediction_class (:,1) == 1);
        tracks_prediction = tracks_2(pred_isolate, :);
        new_data_4 = new_data_3(pred_isolate,:);
        filename_tracks_save = strrep(filename_new_data{i},'NOQ', 'Q');
        data_tracks_pred = struct('Tracks_pred',tracks_prediction, 'Prediction_class', prediction_class,'Training_Scaled',new_data_2,'TrainingFinal',new_data_4);
        save (filename_tracks_save, 'data_tracks_pred');
    end 
end
%% 

function Concatenate_tracks 
    conc_input = inputdlg('Combine Initial Tracks?', 'Track Combiner', [1 100], {'Y'});
    if conc_input{1} == 'Y'
        filenames_TR = uigetfile('*tracksdata.mat', 'Training','Multiselect', 'on');
        filenames_TR = filenames_TR'; 
        num_files_TR = length(filenames_TR);
        Tracks_cell_TR = cell(num_files_TR,1);
        Tracks_cell_seg = cell(num_files_TR,1);
        for i = 1:num_files_TR

                 ld_TR = importdata(filenames_TR{i});
                 training_TR = ld_TR.Training;%Track_mate_training;
                 seg_TR = ld_TR.Segmented_Tracks;
                 Tracks_cell_TR{i} = training_TR ;
                 Tracks_cell_seg{i} = seg_TR;
        end
        tracks_training_final = vertcat(Tracks_cell_TR{:});
        tracks_seg_final = vertcat(Tracks_cell_seg{:});
        save('tracks_training_combined.mat', 'tracks_training_final');
        save('tracks_seg_combined.mat', 'tracks_seg_final');
    else
        filenames_TR = uigetfile('*NOQ*.mat', 'Training','Multiselect', 'on');
        filenames_TR = filenames_TR'; 
        num_files_TR = length(filenames_TR);
        Tracks_cell_TR = cell(num_files_TR,1);
        Tracks_cell_seg = cell(num_files_TR,1);
        for i = 1:num_files_TR

                 ld_TR = importdata(filenames_TR{i});
                 training_TR = ld_TR.TrainingQ;%Track_mate_training;
                 seg_TR = ld_TR.Tracks_pred;
                 Tracks_cell_TR{i} = training_TR ;
                 Tracks_cell_seg{i} = seg_TR;
        end
        tracks_training_final = vertcat(Tracks_cell_TR{:});
        tracks_seg_final = vertcat(Tracks_cell_seg{:});
        save('tracks_training_combined_2.mat', 'tracks_training_final');
        save('tracks_seg_combined_2.mat', 'tracks_seg_final');
    end
end
        
%% 
 
function Classifier_1
    [filename_classifier,  path_classifier] = uigetfile('*mod*.mat','Pick Classifier file');
    pred_used_class = inputdlg({'Spot Width','Mean Speed', 'Max Speed', 'Min Speed', 'Median Speed','Sigma Speed','Mean Quality', 'Max Quality', 'Min Quality', 'Median Quality', 'Sigma Quality','Mean Total Intensity', 'Max Intensity'}, 'Predictors', ...
        [1 100; 1 100; 1 100; 1 100; 1 100; 1 100; 1 100; 1 100; 1 100; 1 100; 1 100; 1 100; 1 100], {'0','1','1','1','1','0','0','0','0','0','0','0','0'});
    pred_used_2_class = str2num(cell2mat(pred_used_class));
    pred_var_class = find(pred_used_2_class);
    
    classifier = importdata(strcat(path_classifier,filename_classifier));
    filename_new_data = uigetfile('*tracksdata.mat','Pick new data files', 'Multiselect', 'on');
    filename_new_data = filename_new_data';

    filename_tracks_combined = uigetfile('*tracks_training_combined*', 'Pick tracks combined file');
    Fraction_factors = inputdlg('Speed Factor', 'Fraction Factors', [1 100],{'1.31'});
    tracks_combined = importdata(filename_tracks_combined);
    mean_speed_values = tracks_combined(:, 2);
     [Best_sp, comp_sp] = GMM_BIC_ML_log(mean_speed_values,2, false);
     mu_sp = Best_sp.mu;
     mu_sp = unique(mu_sp);
     mean_speed = mu_sp(1);
     
     if Best_sp.ComponentProportion(1) > 0.85 | Best_sp.ComponentProportion(2) > 0.85
         mean_speed = mean(log(mean_speed_values));
     end
     disp(mean_speed)
     GM_fit_sp = inputdlg('Log(MeanSP)', 'Fit Correction', [1 100], {num2str(mean_speed)}); 
      mean_sp = str2num(GM_fit_sp{1});
      
     mean_speed = exp(mean_sp);

    for i = 1:length(filename_new_data)
        
        new_data = importdata(filename_new_data{i});
        
        new_data_2 = new_data.Training;
        if isempty(new_data_2) == 1
            continue
        end
         frac_speed = str2num(Fraction_factors{1})/mean_speed;
         
         new_data_2(:,2:6) = new_data_2(:,2:6)*frac_speed;
        
        new_data_2 = new_data_2(:,pred_var_class);
        new_data_3 = new_data_2;
        new_data_3(:,5:7) = new_data.Training(:, [8,12,13]);
%         if isempty(new_data) == 1
%             continue
%         end
        tracks = importdata(filename_new_data{i});
        tracks_2 = tracks.Segmented_Tracks;
        new_data_var = new_data_2;%(:,1:5);

        prediction_class = predict(classifier, new_data_var);
        if iscell(prediction_class) == 1
            prediction_class = cell2mat(prediction_class);
            prediction_class = str2num(prediction_class);
        end
        pred_isolate  = find(prediction_class (:,1) == 1);
        tracks_prediction = tracks_2(pred_isolate, :);
        new_data_4 = new_data_3(pred_isolate,:);

         filename_tracks_save = strrep(filename_new_data{i},'tracksdata.mat', 'NOQ.mat');

        data_tracks_pred = struct('Tracks_pred',tracks_prediction, 'Prediction_class', prediction_class,'Training_Scaled',new_data_2,'TrainingQ',new_data_4);
        save (filename_tracks_save, 'data_tracks_pred');
    end 
end


%% 
function Training_combiner
    filenames_CL = uigetfile('*classification*.mat', 'Pick the classification.mat files','Multiselect', 'on');
    filenames_CL = filenames_CL';
    num_files_CL = length(filenames_CL);
    Tracks_cell_CL = cell(num_files_CL,1);
    filenames_TR = uigetfile('*tracksdata.mat', 'Training','Multiselect', 'on');
    filenames_TR = filenames_TR';  
    Tracks_cell_TR = cell(num_files_CL,1);
    for i = 1:num_files_CL
             ld = importdata(filenames_CL{i});
             ld_TR = importdata(filenames_TR{i});
             classify_CL = ld;%.Training_classification;
             training_TR = ld_TR.Training;%Track_mate_training;
             class_len = length(classify_CL);
             training_TR_rev = training_TR(1:class_len,:);
             Tracks_cell_CL{i}=classify_CL';
             Tracks_cell_TR{i} = training_TR_rev;
             
    end
    
    tracks_classification_final = vertcat(Tracks_cell_CL{:});
    tracks_training_final = vertcat(Tracks_cell_TR{:});
    save('classification_combined.mat', 'tracks_classification_final')
    save('training_combined.mat', 'tracks_training_final');
end

%%
function Training_data
    filename_data_tracks = uigetfile('*tracksdata.mat', 'Pick tracks file');
    [filename_img,  path_img] = uigetfile('*.tif*','Pick Image');
    num_images_st = length(imfinfo(strcat(path_img,filename_img)));
    tracks_data_fil = importdata(filename_data_tracks);
    
    Tracks = tracks_data_fil.Segmented_Tracks;
    Trainer = tracks_data_fil.Training;
    num_tracks = length(Trainer);
    Training_classification = []; %zeros(num_tracks,1);
    %% 
    disp(['Number of tracks:', num2str(num_tracks)])
    for i = 1:num_tracks
            track_iter = num2str(i);
            img_stack = [];
            x_coord_num = Tracks(i,17);
            y_coord_num = Tracks(i,18);
        x_coord = num2str(x_coord_num);
        y_coord = num2str(y_coord_num);
        rad_img = 10;
        frame_start = num2str(Tracks(i,15) + 1);
        frame_end = num2str(Tracks(i,16) + 1);
         vec_read = [Tracks(i,15) + 1:(Tracks(i,16) + 1) + 3];
         if (Tracks(i,16) + 1)  + 3 > num_images_st
             vec_read = [Tracks(i,15)+1: Tracks(i,16) + 1];
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
         h1.Visual.setPropertyValue('DataRangeMin',100);
         h1.Visual.setPropertyValue('DataRangeMax',300);
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
        
        %          opts.WindowStyle = 'Normal';
%          user_input = inputdlg('Noise/Bound?', strcat(track_iter,',', x_coord,',', y_coord,',',frame_start,',',frame_end), [1 100], {'0'},opts);
%          if user_input{1} == 'Stop' | user_input{1} == 'STOP'
%             break
%          end
          
        
        Training_classification(i) = class_input;%str2num(user_input{1});
        close(h1)
    end
    %% 

    filename_save = strrep(filename_data_tracks, 'tracksdata','classification');
    input_save = inputdlg('Save?','Saving Classification',[1 50], {'Y'});
    if input_save{1} == 'Y' | input_save{1} =='y'
        save(filename_save,'Training_classification');
    end
end


%%
function Learner_model
    learn_input = inputdlg({'Use Combined training data set?'}, 'Training',[1 100], {'Combined'});    
    pred_used = inputdlg({'Spot Width','Mean Speed', 'Max Speed', 'Min Speed', 'Median Speed','Sigma Speed','Mean Quality', 'Max Quality', 'Min Quality', 'Median Quality', 'Sigma Quality','Mean Total Intensity', 'Max Intensity'}, 'Predictors', ...
        [1 100; 1 100; 1 100; 1 100; 1 100; 1 100; 1 100; 1 100; 1 100; 1 100; 1 100; 1 100; 1 100], {'0','1','1','1','1','0','0','1','0','0','0','0','0'});
    pred_used_2 = str2num(cell2mat(pred_used));
    pred_var = find(pred_used_2);
    
    if strcmpi('Combined',learn_input{1})
        filename_trainer = uigetfile('*training_combined*.mat','Pick training file');
        Trainer = importdata(filename_trainer);
        Trainer_variables = Trainer;
        Trainer_variables = Trainer_variables(:,pred_var); %(:,1:6);%(:,1:5);
    %Trainer_variables(:,5) = Trainer(:,7);
        filename_classification = uigetfile('*classification_combined*.mat','Pick Classification file');
        Classification = importdata(filename_classification);
    else
%          pred_used = inputdlg({'Spot Width','Mean Speed', 'Max Speed', 'Min Speed', 'Median Speed','Sigma Speed','Mean Quality', 'Max Quality', 'Min Quality', 'Median Quality', 'Sigma Quality','Mean Total Intensity', 'Max Intensity'}, 'Predictors', ...
%          [1 100; 1 100; 1 100; 1 100; 1 100; 1 100; 1 100; 1 100; 1 100; 1 100; 1 100; 1 100; 1 100], {'1','1','1','1','1','0','0','1','0','0','1','0','0'});
%          pred_used_2 = str2num(cell2mat(pred_used));
%          pred_var = find(pred_used_2);
%     
    %Classification = Classifica  tion'; 
    
        filename_trainer = uigetfile('*tracksdata.mat','Pick training file');
        Trainer = importdata(filename_trainer);
        Trainer_variables = Trainer.Training; %(:,1:6);%(:,1:5);
        Trainer_variables = Trainer_variables(:,pred_var);
        filename_classification = uigetfile('*classification*.mat','Pick Classification file');
        Classification = importdata(filename_classification);
        Classification = Classification';
    end
    %tree = fitctree (Trainer_variables, Classification,'OptimizeHyperparameters','auto');
    %% 
    learn_classifier = inputdlg({'Algorithm?'}, 'Learner', [1 100], {'SVM'});
    pred_used_len = length(pred_var);
    if strcmpi('Linear',learn_classifier{1})  
        
        Mdl =fitcdiscr(Trainer_variables, Classification,'DiscrimType','linear', 'OptimizeHyperparameters','auto', 'HyperparameterOptimizationOptions',...
              struct('MaxObjectiveEvaluations',100, 'Repartition', 1));
        filename_learned_model = strrep(filename_classification, 'classification_combined','mod_lin');
        filename_learned_model = strcat(filename_learned_model,'Totalpred',num2str(pred_used_len),'.mat');
        save(filename_learned_model, 'Mdl')
    elseif strcmpi('SVM',learn_classifier{1})
    
        Mdl = fitcsvm(Trainer_variables,Classification,'OptimizeHyperparameters','auto','Standardize',true,...
         'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
          'expected-improvement-plus','MaxObjectiveEvaluations',100));
        filename_learned_model = strrep(filename_classification, 'classification_combined','mod_SVM');
        filename_learned_model = strcat(filename_learned_model,'Totalpred',num2str(pred_used_len),'.mat');
        save(filename_learned_model, 'Mdl')
    
    elseif strcmpi('Tree',learn_classifier{1})
         Mdl = fitcensemble(Trainer_variables,Classification,'OptimizeHyperparameters','auto',...
        'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
        'expected-improvement-plus','MaxObjectiveEvaluations',100));
        filename_learned_model = strrep(filename_classification, 'classification_combined','mod_Tree');
        filename_learned_model = strcat(filename_learned_model,'Totalpred',num2str(pred_used_len),'.mat');
         save(filename_learned_model, 'Mdl')   
    
    elseif strcmpi('Bag',learn_classifier{1})
        bag_input = inputdlg({'Trees?','Leaf Size', 'Predictor Samples','InFraction'}, 'Number of Trees', [1 100; 1 100; 1 100; 1 100], {'150', '15', '2', '0.25'});
        Mdl = TreeBagger(str2num(bag_input{1}),Trainer_variables,Classification,'InBagFraction',str2num(bag_input{4}),'MinLeafSize',str2num(bag_input{2}),'NumPredictorsToSample',str2num(bag_input{3}), 'OOBPred','on','OOBPredictorImportance','on');
        filename_learned_model = strrep(filename_classification, 'classification','mod_Bag');
        filename_learned_model = strcat(filename_learned_model,'_',bag_input{1},'tr','_','LfSz',bag_input{2},'_','Sam',bag_input{3},'_','BagFr',bag_input{4},'_','Preds',num2str(pred_used_len),'.mat');
        figure, 
        plot(oobError(Mdl)) 
        
        save(filename_learned_model, 'Mdl')   
    
    end
end


function Trackmate_Analysis_ML_FINAL

skip_filter_input = inputdlg('Would you like to skip filtering tests?','Existing Data?',[1 50],{'N'});
if skip_filter_input{1} == 'N' | skip_filter_input{1} == 'n'
    input_GMM_clustering = inputdlg({'Intensities components','Time Interval','Truncation Point'}, 'Options',...
        [1 50; 1 50; 1 50], {'4', '1', '3'});
    time_int = str2num(input_GMM_clustering{2});
    truncation_pt = str2num(input_GMM_clustering{3});
    %% 
    
    filenames = uigetfile('*_Q*.mat', 'Pick the segmented tracks .mat files','Multiselect', 'on');
     filenames = filenames';
     num_files = length(filenames);
     Tracks_cell = cell(num_files,1);
     training_cell = cell(num_files,1);
     for i = 1:num_files
             ld=importdata(filenames{i});
             Tracks_cell{i}=ld.Tracks_pred;
             training_cell{i}=ld.TrainingFinal;
     end
    
    tracks_conc = vertcat(Tracks_cell{:});
    num_tracks = length(tracks_conc(:,1));
    disp(num2str(num_tracks));
 
    
    On_time_final = (tracks_conc(:,14))*time_int;
    cat_training = vertcat(training_cell{:});
    inten_use = inputdlg({'Use Mean Intensity?'}, 'GMM Intensity',...
        [1 50], {'Y'});
    if inten_use{1} == 'Y'
        intensities_final =  cat_training(:,6);
    else
        intensities_final = cat_training(:,7);
    end
    
 
    intensities_final_bound = intensities_final;
     tracks_final_bound = tracks_conc;
     On_time_final_bound = On_time_final;
%     disp('classification tracks')
    disp(length(intensities_final_bound(:,1)))
    



    %% 
    intensities_models_tested = str2num(input_GMM_clustering{1});
    [BestModel_intensities, numComponents_intensities] = GMM_BIC ( intensities_final_bound,intensities_models_tested, true)


    idx_int = cluster(BestModel_intensities, intensities_final_bound);
    cluster_array_int = zeros(length( intensities_final_bound),numComponents_intensities);
    Int_clust = zeros(length( intensities_final_bound),numComponents_intensities);
    Int_values = cell(numComponents_intensities,1);

    for j = 1:numComponents_intensities
        cluster_array_int(:,j) = (idx_int==j);
        Int_clust(:,j) = cluster_array_int(:,j).* intensities_final_bound;
        Int_values{j} = nonzeros(Int_clust(:,j));
    end

    num_of_bins2 = ceil(sqrt(numel( intensities_final_bound))); 
    bin_width = (max( intensities_final_bound)-min( intensities_final_bound))/num_of_bins2;
    mean_intensities = zeros(numComponents_intensities,1);
    figure,
    for i=1:numComponents_intensities
        histogram(Int_values{i},'BinWidth',bin_width,'Normalization','count') %might want to incorporate bin width instead
        hold on
        mean_intensities(i) = mean(Int_values{i});
    end
    xlabel('Intensity (A.U)')
    ylabel('Counts')
    hold off
    single_intensities_ID = min(mean_intensities);
    if numComponents_intensities > 2
        unique_intensities = unique(mean_intensities);
        single_intensities_ID = unique_intensities(1);
    end

    Int_col = find(mean_intensities == single_intensities_ID);

    Single_molecules = Int_clust(:,Int_col);
    find_single_molecules = find(Single_molecules);
    %Quality_Tracks_seg_bound_single = Quality_Tracks_seg_bound_total(find_single_molecules,:);
    On_time_bound_single = On_time_final_bound (find_single_molecules,:);
    intensities_bound_single = intensities_final_bound (find_single_molecules, :);
    disp(mean(intensities_bound_single))
    disp(std(intensities_bound_single))
    tracks_bound_single = tracks_final_bound(find_single_molecules,:);
    if length (intensities_final_bound) < 50
        %Quality_Tracks_seg_bound_single = Quality_Tracks_seg_bound_total;
        On_time_bound_single = On_time_final_bound;
        intensities_bound_single = intensities_final_bound;
        tracks_bound_single =  tracks_final_bound;
    end

    %save( 'Quality_Tracks_seg_bound_single.mat', 'Quality_Tracks_seg_bound_single')
else 
    filenames_tracks = uigetfile('*TrackMate_tracks_bound_single.mat', 'Pick the segmented tracks .mat files','Multiselect', 'on');
    tracks_load = load(filenames_tracks);
    tracks_bound_single = tracks_load.tracks_bound_single;
   
     filename_On_time_final = uigetfile('*TrackMate_On_time_bound_single.mat', 'Pick On time bound single file');
        On_time_load = load(filename_On_time_final);
         On_time_bound_single = On_time_load.On_time_bound_single;
     filename_intensities_final = uigetfile('*TrackMate_intensities_bound_single.mat', 'Pick intensities bound single file');
         intensities_load = load(filename_intensities_final);
         intensities_bound_single = intensities_load.intensities_bound_single;
     time_input = inputdlg({'Time Interval', 'Truncation Point'}, 'Time Settings', [1 50;1 50], {'1', '3'});
     time_int = str2num(time_input{1});
     truncation_pt = str2num(time_input{2});
end

%% 
figure,
histogram(On_time_bound_single,'BinMethod','sqrt','Normalization','pdf');
input_test = inputdlg('Would you like to test for two exponentials?','Two Exponentials',[1 50],{'N'});
if input_test{1} == 'Y' | input_test{1}== 'y' 
    
        Trackmate_Analysis_Step2 (On_time_bound_single)
    
    
else
    histogram(On_time_bound_single,'BinMethod','sqrt','Normalization','pdf');
    input_outlier = inputdlg('Eliminate Outliers?', 'Outlier Removals', [1 50], {'Y'});
    if input_outlier{1} =='Y'| input_outlier{1}=='y'
    TF = isoutlier (On_time_bound_single, 'quartiles','ThresholdFactor',4.0);

    On_time_bound_single_filtered = On_time_bound_single(TF~=1,1);
  
    intensities_bound_single_filtered = intensities_bound_single (TF~=1,1);
    tracks_bound_single_filtered = tracks_bound_single(TF~=1,:);
    [est_filtered, ci_filtered, se_filtered] = Fitting_truncExponential (On_time_bound_single_filtered, time_int,truncation_pt);
    
    input_err = inputdlg({'Do you want to calculate bound time?'}, 'Error Calculator', [1 50], {'Y'});
    if input_err{1} == 'Y'| input_err {1} == 'y'
        input_bleach = inputdlg({'Bleach Time', 'Variation in bleach' },'Errors', [1 50; 1 50], {'20', '0.10'});
        [Tbound_filt, Tbound_ci_filt, Tbound_err_filt] = Bound_time_estimator_no_bounds(On_time_bound_single_filtered, est_filtered, str2num(input_bleach{1}),str2num(input_bleach{2}), truncation_pt);
        waitfor(msgbox({num2str(Tbound_filt),strcat(num2str(Tbound_ci_filt(1)), ':', num2str(Tbound_ci_filt(2))), num2str(Tbound_err_filt)}, 'Bound Time'));
        
        input_save_filter = inputdlg('Save Results?', 'Save', [1 50], {'Y'});
    
        if input_save_filter{1} == 'Y' | input_save_filter{1} == 'y'
        Results = struct('BoundTimeFiltered', Tbound_filt, 'BoundTimeCIFiltered', Tbound_ci_filt, 'BoundTimeSTDErrorFiltered', Tbound_err_filt,...
            'TrackDurationFiltered', est_filtered, 'TrackDurationCIFiltered',ci_filtered, 'TrackDurationSEFiltered', se_filtered);
        

        save_dir_input = inputdlg('Pick a name for the folder','Save Folder', [1 50], {'Analysis Files'});
        mkdir(save_dir_input{1}) 
        save(strcat(save_dir_input{1},'/','Results.mat'),'Results');
        save(strcat(save_dir_input{1},'/','Trackmate_On_time_bound_single_filtered.mat'),'On_time_bound_single_filtered')
        save(strcat(save_dir_input{1},'/','Trackmate_intensities_bound_single_filtered.mat'),'intensities_bound_single_filtered')
        save(strcat(save_dir_input{1},'/','Trackmate_tracks_bound_single_filtered.mat'),'tracks_bound_single_filtered')
  
        save(strcat(save_dir_input{1},'/','Trackmate_On_time_bound_single.mat'),'On_time_bound_single')
        save(strcat(save_dir_input{1},'/','Trackmate_intensities_bound_single.mat'),'intensities_bound_single')
        save(strcat(save_dir_input{1},'/','Trackmate_tracks_bound_single.mat'),'tracks_bound_single')
        save(strcat(save_dir_input{1},'/','Trackmate_tracks_TOTAL.mat'),'tracks_conc')
       
        end
    else
        input_save_filter = inputdlg('Save Results?', 'Save', [1 50], {'Y'});
    
        if input_save_filter{1} == 'Y' | input_save_filter{1} == 'y'
        save_dir_input = inputdlg('Pick a name for the folder','Save Folder', [1 50], {'Analysis Files'});
        mkdir(save_dir_input{1}) 
        Results = struct('TrackDurationFiltered', est_filtered, 'TrackDurationCIFiltered',ci_filtered, 'TrackDurationSEFiltered', se_filtered);
        save(strcat(save_dir_input{1},'/','Results.mat'),'Results');
        save(strcat(save_dir_input{1},'/','Trackmate_On_time_bound_single_filtered.mat'),'On_time_bound_single_filtered')
        save(strcat(save_dir_input{1},'/','Trackmate_intensities_bound_single_filtered.mat'),'intensities_bound_single_filtered')
        save(strcat(save_dir_input{1},'/','Trackmate_tracks_bound_single_filtered.mat'),'tracks_bound_single_filtered')
  
        save(strcat(save_dir_input{1},'/','Trackmate_On_time_bound_single.mat'),'On_time_bound_single')
        save(strcat(save_dir_input{1},'/','Trackmate_intensities_bound_single.mat'),'intensities_bound_single')
        save(strcat(save_dir_input{1},'/','Trackmate_tracks_bound_single.mat'),'tracks_bound_single')
        save(strcat(save_dir_input{1},'/','Trackmate_tracks_TOTAL.mat'),'tracks_conc')
        %save(strcat(save_dir_input{1},'/','PMtracker_PSFS_TOTAL.mat'),'psfs_final')
        end
    end

    
    else
        [est_original, ci_original, se_original] = Fitting_truncExponential (On_time_bound_single,time_int,truncation_pt);
        input_err = inputdlg({'Do you want to calculate bound time?'}, 'Error Calculator', [1 50], {'Y'});
    if input_err{1} == 'Y'| input_err {1} == 'y'
        input_bleach = inputdlg({'Bleach Time', 'Variation in bleach' },'Errors', [1 50; 1 50], {'20','0.10'});
        [Tbound_original, Tbound_ci_original, Tbound_err_original] = Bound_time_estimator_no_bounds(On_time_bound_single, est_original, str2num(input_bleach{1}),str2num(input_bleach{2}), truncation_pt);
        waitfor(msgbox({num2str(Tbound_original),strcat(num2str(Tbound_ci_original(1)), ':', num2str(Tbound_ci_original(2))), num2str(Tbound_err_original)}, 'Bound Time'));
        input_save = inputdlg('Save Results?', 'Save', [1 50], {'Y'});
    
        if input_save{1} == 'Y' | input_save{1} == 'y'
        save_dir_input = inputdlg('Pick a name for the folder','Save Folder', [1 50], {'Analysis Files'});
        mkdir(save_dir_input{1}) 
        Results = struct('BoundTime', Tbound_original, 'BoundTimeCI', Tbound_ci_original, 'BoundTimeSTDError',  Tbound_err_original,...
            'TrackDuration', est_original, 'TrackDurationCI',ci_original, 'TrackDurationSE', se_original);
        save(strcat(save_dir_input{1},'/','Results.mat'),'Results');
        save(strcat(save_dir_input{1},'/','Trackmate_On_time_bound_final.mat'),'On_time_final_bound')
        save(strcat(save_dir_input{1},'/','Trackmate_intensities_bound_final.mat'),'intensities_final_bound')
        save(strcat(save_dir_input{1},'/','Trackmate_On_time_bound_single.mat'),'On_time_bound_single')
        save(strcat(save_dir_input{1},'/','Trackmate_intensities_bound_single.mat'),'intensities_bound_single')
        save(strcat(save_dir_input{1},'/','Trackmate_tracks_bound_single.mat'),'tracks_bound_single')
        save(strcat(save_dir_input{1},'/','Trackmate_tracks_TOTAL.mat'),'tracks_conc')
        
        end
    else
        input_save = inputdlg('Save Results?', 'Save', [1 50], {'Y'});
    
        if input_save{1} == 'Y' | input_save{1} == 'y'
        save_dir_input = inputdlg('Pick a name for the folder','Save Folder', [1 50], {'Analysis Files'});
        mkdir(save_dir_input{1}) 
        Results = struct('TrackDuration', est_original, 'TrackDurationCI',ci_original, 'TrackDurationSE', se_original);
        save(strcat(save_dir_input{1},'/','Results.mat'),'Results');
        save(strcat(save_dir_input{1},'/','Trackmate_On_time_bound_final.mat'),'On_time_final_bound')
        save(strcat(save_dir_input{1},'/','Trackmate_intensities_bound_final.mat'),'intensities_final_bound')
        save(strcat(save_dir_input{1},'/','Trackmate_On_time_bound_single.mat'),'On_time_bound_single')
        save(strcat(save_dir_input{1},'/','Trackmate_intensities_bound_single.mat'),'intensities_bound_single')
        save(strcat(save_dir_input{1},'/','Trackmate_tracks_bound_single.mat'),'tracks_bound_single')
        save(strcat(save_dir_input{1},'/','Trackmate_tracks_TOTAL.mat'),'tracks_conc')
        
        end
    end
    end

end



end          
%% 
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
     