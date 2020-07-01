% num_tracks=length(track_molecules7);
% displacements=cell(num_tracks,1);
% tracks_bound=cell(num_tracks,1);
% displacements_bound=cell(num_tracks,1);
% tracks_unbound=cell(num_tracks,1);
% displacements_unbound=cell(num_tracks,1);
%r_thresh=0.075;
[filename_spots,  path_spots] = uigetfile('*_spots.csv*','Pick Spots file');
spots_file_1 = csvread(strcat(path_spots,filename_spots));
spots_file_2 = spots_file_1;%.data;
filename_pred = uigetfile('*Q*.mat*','Pick prediction file');
pred_file = importdata(filename_pred);
pred_file_2 = pred_file.Tracks_pred;
num_tracks = length(pred_file_2(:,1));

displacements=cell(num_tracks,1);
% tracks_bound=cell(num_tracks,1);
% displacements_bound=cell(num_tracks,1);
% tracks_unbound=cell(num_tracks,1);
% displacements_unbound=cell(num_tracks,1);
for i=1:num_tracks
    %displace=track_molecules7{1,i};
    %displace_2=displace(:,1:4);
    ID = pred_file_2(i,1);
    Dur = pred_file_2 (i,14);
    ID_find = find(spots_file_2(:,1) == ID);
    ID_spots = spots_file_2(ID_find,:);
    [~, idx] = sort(ID_spots(:,2),1);
    rev_spots = ID_spots(idx,:);
    if length(ID_find) > Dur + 1
        continue
    end
    displace_2 = rev_spots(:,3:4);
    displace_3 =zeros(length(ID_find)-1,1);
    
    for j=1:length(displace_2)
        if j+1>length(displace_2)
            break
        end
        x_displace=displace_2(j+1,1)-displace_2(j,1);
        y_displace=displace_2(j+1,2)-displace_2(j,2);
        r_displace=sqrt(x_displace^2 + y_displace^2);
        displace_3(j,1)=r_displace;
        
    end
    
 %bound=find(displace_3<=r_thresh);
 %unbound=find(displace_3>r_thresh);
%  bd=min(bound);
%  un=min(unbound);
%  bound_2=bound+1;
%  unbound_2=unbound+1;
%  
%  if isempty(unbound)==1| bd<un  
%      bound_2=[bd;bound_2];
%  end
%  if isempty(bound)==1 | un<bd
%     unbound_2=[un;unbound_2];
%  end
     
     

 
%  tracks_bound{i}=displace_2(bound_2,:);
%  tracks_unbound{i}=displace_2(unbound_2,:);
%  
 displacements{i}=displace_3;
%  displacements_bound{i}=displace_3(bound);
%  displacements_unbound{i}=displace_3(unbound);
%  
 
 
 
 
 

end
cumu_displacements=cell2mat(displacements);
cumu_displacements = nonzeros(cumu_displacements);
% cum_displacements_bound=cell2mat(displacements_bound);
% cum_displacements_unbound=cell2mat(displacements_unbound);
figure,histogram(cumu_displacements);
% figure,histogram(cum_displacements_bound);
% figure,histogram(cum_displacements_unbound);
[fy, fx] = ecdf(cumu_displacements);
plot(fx,fy);
figure, histogram (cumu_displacements, 'BinMethod','fd')

%incorporate the ability to partition tracks into bound vs unbound instead
%of just fragmenting (e.g. some 