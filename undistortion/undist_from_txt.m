clear
close all
clc
%%
dist_folder = 'folder_to_save_undistorted_images'
if ~exist(dist_folder, 'dir')
  mkdir(dist_folder);
end

fileID = fopen('path_to_txt_file','r');
file_content = textscan(fileID,'%s');
fclose(fileID);

paths = file_content{1};
focal = file_content{2};
distortion = file_content{3};

hfig=figure;
f = 0;
dist = 0;

for i=1:length(paths)

    Idis = imread(paths{i});

    % xi = 1.08;
    xi = distortion(i); % distortion
    dist = dist + xi;
    [ImH,ImW,~] = size(Idis);
    % f_dist = 320 * (ImW/ImH) * (ImH/299); 
    f_dist = focal(i) * (ImW/ImH) * (ImH/299); % focal length
    f = f + f_dist;
    u0_dist = ImW/2;
    v0_dist = ImH/2;
    Paramsd.f = f_dist;
    Paramsd.W = u0_dist*2;  
    Paramsd.H = v0_dist*2;
    Paramsd.xi = xi;

    Paramsund.f = f_dist;
    Paramsund.W = u0_dist*2;  
    Paramsund.H = v0_dist*2;
    
    tic
    Image_und = undistSphIm(Idis, Paramsd, Paramsund);
    toc

    %if (size(Image_und,1)~=0)
        paths_list = strsplit(paths{i}, '/');
        res1 = strsplit(paths_list{9}, '_');
        res2 = strsplit(res1{2}, '.')

        out = str2double(res2{1});
        
        filename = [sprintf('%04d', out),'.jpg'];
        fullname = fullfile(dist_folder,filename);
        imwrite(Image_und,fullname);
    %end
end