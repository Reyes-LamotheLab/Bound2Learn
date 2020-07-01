function seg_yeast = yeast_segmentation_automated
filename=uigetfile('*SME_projection.tif','Pick your SME file');
%filename = uigetfile('*_SME_projection.tif');
user_input_std_initial = inputdlg('STD threshold','Set an Initial Standard Deviation Threshold',[1 80],{'3000'});
std_thresh = str2num(user_input_std_initial{1});
%std_thresh =1800;
%int_thresh = 
%image_read=bfopen(filename);
%% 
rd2=imread(filename); %image_read{1,1}{1,1};
rd2 = uint16(rd2);
 %figure, imshow(rd2,[min(min(rd2)), max(max(rd2))]);
%imcontrast
%% 
se=strel('disk',11,4);% default 11 and 4
tophatFiltered = imtophat(rd2,se);
%figure, imshow(tophatFiltered)
%imcontrast

    

%% 
%thresh_crit=adaptthresh(tophatFiltered,0.00005,'Statistic','Gaussian');%ddefault 0.02
thresh=imbinarize(tophatFiltered,'adaptive');
%thresh=imbinarize(rd2,'adaptive');
% figure,imshow(thresh)
%% 

bw2=bwareaopen(thresh,200);%default = 150
 %figure,imshow(bw2);

%% 
bw3=mat2gray(bw2);
bw4=imgaussfilt(bw3,0.5);%default 1.7
  %figure,imshow(bw4);
%% 
%thresh_crit2=adaptthresh(bw4,0.02,'Statistic','Gaussian');%default 0.02
thresh2=imbinarize(bw4,'adaptive');
 %figure,imshow(thresh2);

%% 
bw5 = bwareaopen(thresh2,300);%default 100
bw6 = uint16(bw5);
 %figure,imshow(bw5);
 %cc=bwconncomp(bw5)
 stats = regionprops(bw5,rd2,'Image','PixelList','PixelValues');
 PixelValues = {stats.PixelValues}.';
 PixelList = {stats.PixelList}.';

%% 
while(1)
image_replace = rd2;
bin_image = bw5;
for i = 1:length(PixelValues)
    std_int = std(double(PixelValues{i}));
    mean_int = mean(double(PixelValues{i}));
    px_coordinates = PixelList{i};
    if  std_int < std_thresh%mean_int < std_thresh%std_int < std_thresh | mean_int < int_thresh
        for j = 1:length(px_coordinates)
        image_replace(px_coordinates(j,2), px_coordinates(j,1)) = 0;
        bin_image(px_coordinates(j,2), px_coordinates(j,1)) = 0;
        end
    else
        continue
    end
end
bin_image = uint16(bin_image);
fused_image = imfuse(bin_image, rd2,'blend');
%figure,imshow(image_replace,[min(min(image_replace)), max(max(image_replace))])
%figure, imshow(bin_image,[min(min(bin_image)), max(max(bin_image))]) 
figure, imshow(rd2,[min(min(rd2)), max(max(rd2))]);
figure, imshow(fused_image)
savename=strrep(filename,'SME_projection','binary_auto');

user_input = inputdlg ('Is the segmentation good?','Yeast Segmentation',[1 50],{'NO'});
    if user_input{1} == 'Y'| user_input {1} == 'y'
        close all
        imwrite(bin_image,savename,'tif') 
        return
    else 
        close all
        default_std_thresh = num2str(std_thresh);
        user_input_std = inputdlg('Std threshold','Set a new threshold',[1 50],{default_std_thresh});
        std_thresh = str2num(user_input_std{1});
    end
end
%imwrite(bin_image,savename,'tif')
end
