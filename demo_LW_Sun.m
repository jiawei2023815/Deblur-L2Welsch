%% Remark
% This demo using L2-Welsch iterative model to deblur the Sun et al's datasets
% Sun et al's datasets includes 80 image-GT, 8 kernel-GT, and a total of 640 blurred images. It's belongs to a uniform datasets
% This dataset originates from the following paper:
%       L. Sun, S. Cho, J. Wang, and J. Hays,
%       Edge-based Blur Kernel Estimation Using Patch Priors,
%       IEEE ICCP 2013.
%
% This demo utilizes GPU CUDA for accelerated execution.
% Before running this demo, please make sure to install NVIDIA's CUDA parallel computing platform.
%
% Author: Jiawei Lu @IMU
% Date: 2023-08-15
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

% set the kernel szie of each blurred image
kernel_sizes = ones(15, 1) * [19, 17, 15, 27, 13, 21, 23, 23];
kernel_sizes = gpuArray(single(kernel_sizes));

% set the gamma correction of each blurred image
gamma_correct = ones(15, 1) * [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
gamma_correct = gpuArray(single(gamma_correct));

% initialize the struct opts
opts.prescale = 1;
opts.xk_iter = 3; % number of iterations per layer
opts.k_thresh = 20; % normalization threshold for each layer of iterations
opts.isdisplay_est_kernel = 1; % show the deblurring process
opts.c = 0.1; % reweighted coefficient

% convert the strcut opts to GPU
opts.prescale = gpuArray(single(opts.prescale));
opts.xk_iter = gpuArray(single(opts.xk_iter));
opts.k_thresh = gpuArray(single(opts.k_thresh));
opts.isdisplay_est_kernel = gpuArray(single(opts.isdisplay_est_kernel));
opts.c = gpuArray(single(opts.c));

% initialize the timer matrix
time1 = zeros(80, 8);
time2 = zeros(80, 8);
deblur_time = zeros(80, 8);
time1 = gpuArray(single(time1));
time2 = gpuArray(single(time2));
deblur_time = gpuArray(single(deblur_time));

%% main loop
for img = 1 : 80
    
    for blur = 1 : 8
        
        % input the blurred images
        blurName = ['Blurry', num2str(img), '_', num2str(blur), '.png'];
        filename_blur = ['../Data_Text/Blurry/', blurName];
        y = im2double(imread(filename_blur));
        y = gpuArray(single(y));
        yg = rgb2gray(y);
        yg = gpuArray(single(yg));
        
        fprintf(['===================== ', 'Blurred image ', num2str(img), '_', num2str(blur), ' =====================\n']);
        
        % match the size of kernel and gamma correction with images
        opts.kernel_size = kernel_sizes(img, blur);
        opts.gamma_correct = gamma_correct(img, blur);
        
        % set the prior coefficients
        lambda_grad = 4e-3; lambda_surface = 4e-3;
        opts.gamma = 4e-3; opts.delta = 4e-3;
        
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
        
        % save the results
        saving_path = '../Data_Sun/Deblur_L2-Welsch/Results_1/';
        imwrite(Latent_cho, [saving_path, 'L2-Welsch_deblur', num2str(img), '_', num2str(blur), '.png']);
        imwrite(interim_latent, [saving_path, 'L2-Welsch_interim', num2str(img), '_', num2str(blur), '.png']);
        imwrite(k, [saving_path, 'L2-Welsch_kernel', num2str(img), '_', num2str(blur), '.png']);
        
    end
    
end
