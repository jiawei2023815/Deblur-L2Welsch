function deconv_test

%dir  
img = imread([dir, 'saturated_img.png']);
img = double(img)/255;

%psf  
% sigma: standard deviation for Gaussian noise (for inlier data)
% reg_str: regularization strength for sparse priors
sigma = 5/255;
reg_str = 0.003;
deblurred = deconv_outlier(img, psf, sigma, reg_str);

figure
imshow(deblurred);

