filenames_D = uigetfile('*Tracks_D*fx_mrg*', 'Pick Tracks D files', 'Multiselect', 'on');
filenames_D = filenames_D';

num_files_D = length(filenames_D);
D_values_tot = [];
D_values_tot_t1 = [];
rog_values_tot = [];
alpha_values_tot = [];
sigma_values_tot = [];
adjr_squares_tot = [];
gof_thresh = 0.70;
pixel_um = 0.100;
frame_int_seconds = 0.5;
texp = 0.5;
for i = 1:num_files_D
    tracks = importdata(filenames_D{i});
    if isempty(tracks) == 1
        continue
    end
    num_tracks = length(tracks(:,1));
    tracks_MSDs = cell(num_tracks,6);
    D_values = zeros(num_tracks,1);
    D_values_t1 = zeros(num_tracks,1);
    alpha_values = zeros(num_tracks,1);
    sigma_values = zeros(num_tracks,1);
    adjr_squares = zeros(num_tracks,1);
    rog_values = zeros(num_tracks,1);
    for j = 1:num_tracks
        tracks_info = tracks{j,1};
        tracks_coordinates = tracks_info (:,3:4);
        track_length = length(tracks_coordinates(:,1));
        MSD = zeros(track_length-1, 3);
        %%m = t
        %k = tau
        xmean = mean(tracks_coordinates(:,1));
        ymean = mean(tracks_coordinates(:,2));
        radii_values_square = zeros(track_length,1);
        for k = 1:track_length
            radii_values_square (k,1) = (tracks_coordinates(k,1) - xmean)^2 + (tracks_coordinates(k,2) - ymean)^2;
        end
        rog_track_length = track_length; %* frame_int_seconds;
        rog = sqrt((sum(radii_values_square)/rog_track_length));
        rog = rog*pixel_um;
        for k = 1:track_length - 1
            MSD_ind = zeros(track_length - k,1);
            for m = 1:track_length - k
                MSD_ind(m,1) = (tracks_coordinates(m + k,1) - tracks_coordinates(m,1))^2 + ...
                    (tracks_coordinates(m + k, 2) - tracks_coordinates(m,2))^2;
                
            end
            MSD(k,2) = ((sum(MSD_ind))/(track_length-k))*pixel_um^2;
            MSD(k,1) = k *frame_int_seconds;
            MSD(k,3) = track_length - k;
        end
        Weights = MSD(:,3)./(sum(MSD(:,3)));
        %MSD_fit = fittype('4*D*t^a + (4*b^2 - 8*(1/6)*D*0.02)', 'dependent',{'y'},'independent',{'t'},'coefficients',{'D','a','b'});
        MSD_fit = fittype('4*D*t^a + (4*b^2 - 8*(1/6)*(Texp/Tint)*D*Tint)', 'problem', {'Tint', 'Texp'},'dependent',{'y'},'independent',{'t'},'coefficients',{'D','a','b'});
        MSD_t1 =  MSD(1,2);
        D_t1 = MSD_t1/(4*frame_int_seconds) - 0.04^2/frame_int_seconds;
        start = [0.002, 0.5, 0.04];
        lower = [0, 0.0, 0.001];
        upper = [10, 2, 0.700];
        %[MSD_params,gof,output] = fit(MSD(:,1), MSD(:,2), MSD_fit,'Start',start,'Lower',lower,'Upper',upper,'Weights',Weights,'MaxIter', 10000,'TolFun', 10^-6);
        [MSD_params,gof,output] = fit(MSD(:,1), MSD(:,2), MSD_fit,'problem',{frame_int_seconds, texp},'Start',start,'Lower',lower,'Upper',upper,'Weights',Weights,'MaxIter',10000000,'TolFun', 10^-6,'Robust','Bisquare');
        fit_params = [MSD_params.D, MSD_params.a, MSD_params.b];
        
        
        if gof.adjrsquare < gof_thresh
            continue
        end
                
        tracks_MSDs{j,1} = MSD;
        tracks_MSDs{j,2} = fit_params;
        tracks_MSDs{j,3} = gof;
        tracks_MSDs{j,4} = MSD_params;
        tracks_MSDs{j,5} = D_t1;
        tracks_MSDs{j,6} = rog;
        D_values(j,1) = MSD_params.D;
        alpha_values(j,1) = MSD_params.a;
        sigma_values(j,1) = MSD_params.b;
        adjr_squares(j,1) = gof.adjrsquare;
        D_values_t1(j,1) = D_t1;    
        rog_values(j,1) = rog;
        
        %tracks_MSDs = tracks_MSDs(~cellfun('isempty',tracks_MSDs));
        %tracks_MSDs = struct('MSD_values', MSD, 'GoF',gof,'Parameter_Values', fit_params); 
        
         %plot_array = [1:20:num_tracks];
         %if any(plot_array == j)
             %figure,
             %plot(MSD_params, MSD(:,1), MSD(:,2))
         %end
        
    end
    D_values_tot = [D_values_tot ; D_values];
    D_values_tot_t1 = [D_values_tot_t1; D_values_t1];
    rog_values_tot = [rog_values_tot;rog_values];
    alpha_values_tot = [alpha_values_tot ; alpha_values];
    sigma_values_tot = [sigma_values_tot; sigma_values];
    adjr_squares_tot = [adjr_squares_tot; adjr_squares];
    filenames_MSDs = strrep(filenames_D{i}, 'Tracks_D','MSDs');
    save(filenames_MSDs, 'tracks_MSDs');
    
end
D_values_tot = D_values_tot(D_values_tot~=0);
D_values_tot_t1 = D_values_tot_t1(D_values_tot_t1~=0);
 rog_values_tot =  rog_values_tot(rog_values_tot~=0);
alpha_values_tot = alpha_values_tot(alpha_values_tot~=0);
sigma_values_tot = sigma_values_tot(sigma_values_tot~=0);
 adjr_squares_tot = adjr_squares_tot( adjr_squares_tot~=0);  
save('Diffusion_total_fx_mrg', 'D_values_tot');
save('Diffusion_t1_fx_mrg','D_values_tot_t1');
save('ROG_TOTAL_fx_mrg', 'rog_values_tot');
            
                    
            
            
            