function Image_und = undistSphIm(Idis, Paramsd, Paramsund)

%Paramsund=Paramsd;
Paramsund.W=Paramsd.W*3; % size of output (undist)
Paramsund.H=Paramsd.H*3;
%Paramsund.f=Paramsd.f;


%Parameters of the camera to generate
f_dist = Paramsd.f;
u0_dist = Paramsd.W/2;  
v0_dist = Paramsd.H/2;
%
f_undist = Paramsund.f;
u0_undist = Paramsund.W/2;  
v0_undist = Paramsund.H/2;
xi = Paramsd.xi; % distortion parameters (spherical model)
[Imd.H, Imd.W, ~] = size(Idis);

tic
% 1. Projection on the image 
[grid_x, grid_y] = meshgrid(1:Paramsund.W,1:Paramsund.H);
X_Cam = grid_x./f_undist - u0_undist/f_undist;
Y_Cam  = grid_y./f_undist - v0_undist/f_undist;
Z_Cam =  ones(Paramsund.H,Paramsund.W);

%2. Image to sphere cart
xi1 = 0;
alpha_cam = ( xi1.*Z_Cam + sqrt( Z_Cam.^2 + ...
             ( (1-xi1^2).*(X_Cam.^2 + Y_Cam.^2) ) ) ) ...
             ./ (X_Cam.^2 + Y_Cam.^2 + Z_Cam.^2);

X_Sph = X_Cam.*alpha_cam;
Y_Sph = Y_Cam.*alpha_cam;
Z_Sph = (Z_Cam.*alpha_cam) - xi1;

%3. reprojection on distorted
den = xi*(sqrt(X_Sph.^2 + Y_Sph.^2 + Z_Sph.^2)) + Z_Sph;
X_d = ((X_Sph*f_dist)./den)+u0_dist;
Y_d = ((Y_Sph*f_dist)./den)+v0_dist;

%4. Final step interpolation and mapping
Image_und=zeros(Paramsund.H,Paramsund.W,3);
for c=1:3
Image_und(:,:,c) = interp2(im2double(Idis(:,:,c)), X_d, Y_d, 'cubic');
end
toc;
[Im_und.H, Im_und.W, ~] = size(Image_und);
%ROI
min(X_d(:)), max(X_d(:));
min(Y_d(:)), max(Y_d(:));
size(Idis);