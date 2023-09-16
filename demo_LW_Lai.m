%% Remark
% This demo using L2-Welsch iterative model to deblur the Lai et al's datasets
% Lai et al's datasets includes 25 image-GT, 4 kernel-GT, and a total of 100 blurred images. It's belongs to a uniform datasets.
% This dataset originates from the following paper:
%       W.-S. Lai, J.-B. Huang, Z. Hu, N. Ahuja and M.-H. Yang,
%       A Comparative Study for Single Image Blind Deblurring,
%       IEEE CVPR 2016.
%
% This demo utilizes GPU CUDA for accelerated execution.
% Before running this demo, please make sure to install NVIDIA's CUDA parallel computing platform.
%
% Author: Jiawei Lu @IMU
% Date: 2023-05-01
% Email: 32236108@mail.imu.edu.cn

%% initialize parameters
clear;
close all;
clc;
warning('off');

% input the non-blind methods path
addpath(genpath('cho_code'));
addpath(genpath('whyte_code'));
addpath(genpath('deconv_outliers_code/code'));

% set the kernel size of each blurred image
kernel_sizes = ones(25, 1) * [31, 51, 55, 75];
kernel_sizes = gpuArray(single(kernel_sizes));

% set the gamma correction of each blurred image
gamma_correct = ones(25, 1) * [1.0, 1.0, 1.0, 1.5];
gamma_correct = gpuArray(single(gamma_correct));

% initialize the struct opts
opts.prescale = 1;
opts.xk_iter = 1; % number of iterations per layer
opts.k_thresh = 20; % normalization threshold for each layer of iterations
opts.isdisplay_est_kernel = 1; % show the deblurring process
opts.c = 0.1; % reweighted coefficient

% convert struct opts to GPU type
opts.prescale = gpuArray(single(opts.prescale));
opts.xk_iter = gpuArray(single(opts.xk_iter));
opts.k_thresh = gpuArray(single(opts.k_thresh));
opts.isdisplay_est_kernel = gpuArray(single(opts.isdisplay_est_kernel));
opts.c = gpuArray(single(opts.c));

% initialize the timer matrix
time1 = zeros(25, 4);
time2 = zeros(25, 4);
deblur_time = zeros(25, 4);
time1 = gpuArray(single(time1));
time2 = gpuArray(single(time2));
deblur_time = gpuArray(single(deblur_time));

%% main loop
for img = 1 : 1
    
    for blur = 1 : 1
        
        % input the blurred images
        blurName = ['Blurry', num2str(img), '_', num2str(blur), '.png'];
        filename_blur = ['../Data_Lai/Blurry/', blurName];
        y = im2double(imread(filename_blur));
        y = gpuArray(single(y));
        yg = rgb2gray(y);
        yg = gpuArray(single(yg));
        
        fprintf(['===================== ', 'Blurred image ', num2str(img), '_', num2str(blur), ' =====================\n']);
        
        opts.kernel_size = kernel_sizes(img, blur); % match the size of kernel with images
        opts.gamma_correct = gamma_correct(img, blur); % match the gamma correction with images
        
        % set the prior coefficients
        lambda_grad = 4e-3; lambda_surface = 4e-3;
        opts.gamma = 4e-3; opts.delta = 2e-2;
        
        % convert data to GPU type
        lambda_grad = gpuArray(single(lambda_grad));
        lambda_surface = gpuArray(single(lambda_surface));
        opts.gamma = gpuArray(single(opts.gamma));
        opts.delta = gpuArray(single(opts.delta));
        
        % blind deblurring
        tic;
        [kernel, interim_latent] = blind_deconv(yg, lambda_surface, lambda_grad, opts);
        time1(img, blur) = toc;
        fprintf(['(1) The blur kernel estimation takes ', num2str(time1(img, blur)), ' seconds.\n']);
        
        % non-blind deblurring
        opts.sigma = 5 / 255;
        opts.reg_str = 8e-3;
        opts.sigma = gpuArray(single(opts.sigma));
        opts.reg_str = gpuArray(single(opts.reg_str));
        
        tic;
        Latent_cho = deconv_outlier(y, kernel, opts.sigma, opts.reg_str); % cho's non-blind method
        %Latent_cho = whyte_deconv(y, kernel); % whyte's non-blind method
        time2(img, blur) = toc;
        deblur_time(img, blur) = time1(img, blur) + time2(img, blur);
        fprintf(['(2) The non-blind deblurring takes ', num2str(time2(img, blur)), ' seconds.\n']);
        fprintf(['(3) The blind and non-blind debluring takes ', num2str(deblur_time(img, blur)), ' seconds.\n']);
        
        % normalize the kernel
        k = kernel - min(kernel(:));
        k = k ./ max(k(:));
        
        % convert data to CPU type
        interim_latent = gather(interim_latent);
        Latent_cho = gather(Latent_cho);
        k = gather(k);
        
        % save results
        saving_path = '../Data_Lai/L2-Welsch/';
        imwrite(Latent_cho, [saving_path, 'L2-Welsch_deblur', num2str(img), '_', num2str(blur), '.png']);
        imwrite(interim_latent, [saving_path, 'L2-Welsch_interim', num2str(img), '_', num2str(blur), '.png']);
        imwrite(k, [saving_path, 'L2-Welsch_k', num2str(img), '_', num2str(blur), '.png']);
        
    end
    
end
