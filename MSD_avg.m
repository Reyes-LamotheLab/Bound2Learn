
files_MSD = uigetfile('*MSDs*fx_mrg*.mat','Multiselect','on');
files_MSD = files_MSD';
num_files = length(files_MSD(:,1));
pixel_um = 1;%0.100;
frame_int_seconds = 1.0; %time interval
texp = 0.5; %exposure time

%%
MSD_arr2 = [];
MSD_dat = [];
for j = 1:num_files

    track_MSD = importdata(files_MSD{j});
    
    %MSD_all = [];
    num_tracks = length(track_MSD(:,1));
    trk_length_max = 300;
    MSD_arr = zeros(num_tracks,trk_length_max);
    MSD_dat_points = zeros(num_tracks,trk_length_max);
    for i = 1:num_tracks
        track_vals = track_MSD{i,1};
        if isempty(track_vals) == 1
            continue
        end
        MSD_vals = track_vals(:,2);
        dat_vals = track_vals(:,3);
        dat_vals = dat_vals';
        MSD_vals = MSD_vals';
        trk_len = length(MSD_vals);
        MSD_arr(i,1:trk_len) = MSD_vals;
        MSD_dat_points(i,1:trk_len) = dat_vals;
    end
    MSD_arr2 = [MSD_arr2;MSD_arr];
    MSD_dat = [MSD_dat;MSD_dat_points];
end
    MSD_vals_tot = zeros(trk_length_max,1);
    MSD_dat_tot = zeros(trk_length_max,1);
    for i = 1:trk_length_max
        num_dat = sum(nonzeros(MSD_dat(:,i)));
        w_dat = nonzeros(MSD_dat(:,i))./num_dat;
        w_mean = sum(nonzeros(MSD_arr2(:,i)).*w_dat);
        MSD_vals_tot(i) = w_mean;%mean(nonzeros(MSD_arr2(:,i)));
        MSD_dat_tot(i) = num_dat;%sum(nonzeros(MSD_dat(:,i)));
    end

    %% 

 %MSD_vals_tot_fin = MSD_vals_tot(~isnan(MSD_vals_tot));
 MSD_vals_tot_fin = MSD_vals_tot(MSD_vals_tot~=0);
 MSD_dat_tot_fin = MSD_dat_tot(MSD_dat_tot~=0); 
 %% 
 num_time_points = length(MSD_vals_tot_fin);
 time_s = [1:num_time_points]'*frame_int_seconds;
 Weights = MSD_dat_tot_fin(:,1)./(sum(MSD_dat_tot_fin(:,1)));
 MSD_fit = fittype('4*D*t^a + (4*b^2 - 8*(1/6)*(Texp/Tint)*D*Tint)', 'problem', {'Tint', 'Texp'},'dependent',{'y'},'independent',{'t'},'coefficients',{'D','a','b'});
%MSD_fit = fittype('4*D*t^a + (4*b^2)', 'dependent',{'y'},'independent',{'t'},'coefficients',{'D','a','b'});
 MSD_t1 =  MSD_vals_tot_fin(1);
       % D_t1 = MSD_t1/(4*frame_int_seconds) - 0.04^2/frame_int_seconds;
        start_init = [0.002, 0.5, 0.04];
        lower_bd = [0, 0.0, 0.001];
        upper_bd = [10, 2, 0.700];
        exclude_dat = time_s>50;
       %[MSD_params,gof,output] = fit(time_s, MSD_vals_tot_fin, MSD_fit,'problem',{frame_int_seconds, texp},'Weights',Weights,'Start',start_init,'Lower',lower_bd,'Upper',upper_bd,'MaxIter',10000000,'TolFun', 10^-6, 'Robust', 'Bisquare');
        [MSD_params,gof,output] = fit(time_s, MSD_vals_tot_fin, MSD_fit,'problem',{frame_int_seconds, texp},'Exclude',exclude_dat,'Weights',Weights,'Start',start_init,'Lower',lower_bd,'Upper',upper_bd,'MaxIter',10000000,'TolFun', 10^-6, 'Robust', 'Bisquare');
figure, plot(MSD_params, time_s, MSD_vals_tot_fin)
% 
 %% 
%   boot_options=statset('UseParallel',true); 
% %  
% %      
%   param_fit_MSD = @(tau_time_2,MSD_in_2) D_val_calc;
%     [boot_CI, ~]=bootci(1000,{param_fit_MSD,time_s, MSD_vals_tot_fin}, 'Weights',Weights,'type','bca','options',boot_options);
%      Tbound_ci = boot_CI; 
%      %Tbound_err = std(bootstat_norm);
     %% 
     function[D_vals, alpha_vals, sigma_vals] = D_val_calc(tau_time,MSD_in)
     [MSD_fit_params] =  fit(tau_time, MSD_in, MSD_fit,'problem',{frame_int_seconds, texp},'Start',start_init,'Lower',lower_bd,'Upper',upper_bd,'MaxIter',10000000,'TolFun', 10^-6, 'Robust', 'Bisquare');
     D_vals = MSD_fit_params.D;
     alpha_vals = MSD_fit_params.a;
     sigma_vals = MSD_fit_params.b;
     end
 