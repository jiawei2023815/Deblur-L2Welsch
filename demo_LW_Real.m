%% Remark
% deblurring the real world images by L2-Welsch iterative model
%
% This demo utilizes GPU CUDA for accelerated execution.
% Before running this demo, please make sure to install NVIDIA's CUDA parallel computing platform.
%
% Author: Jiawei Lu @IMU
% Date: 2023-05-10
% Email: 32236108@mail.imu.edu.cn

%%
clear;
close all;
clc;
warning('off');

% input the non-blind methods path
addpath(genpath('cho_code'));
addpath(genpath('whyte_code'));
addpath(genpath('deconv_outliers_code/code'));

% initialize the struct opts
opts.prescale = 1;
opts.k_thresh = 20; % normalization threshold for each layer of iterations
opts.isdisplay_est_kernel = 1; % show the deblurring process
opts.c = 0.1; % reweighted coefficient

%%
% car
% lambda_grad = 4e-3; lambda_surface = 4e-3;
% opts.gamma = 4e-3; opts.delta = 4e-3;
% opts.kernel_size = 45; opts.xk_iter = 7; opts.gamma_correct = 1.0;
% blur = imread('../Real_world/car.png'); y = im2double(blur); yg = im2double(rgb2gray(blur));

% high illumination
% lambda_grad = 4e-3; lambda_surface = 4e-3;
% opts.gamma = 4e-3; opts.delta = 4e-2;
% opts.kernel_size = 131; opts.xk_iter = 4; opts.gamma_correct = 1.0;
% blur = imread('../Real_world/High-light.png'); y = im2double(blur); yg = im2double(rgb2gray(blur));

% human face
% lambda_grad = 4e-3; lambda_surface = 4e-3;
% opts.gamma = 4e-3; opts.delta = 4e-4;
% opts.kernel_size = 25; opts.xk_iter = 15; opts.gamma_correct = 1.0;
% blur = imread('./face_2.png'); y = im2double(blur); yg = im2double(rgb2gray(blur));

% saturated
lambda_grad = 4e-3; lambda_surface = 4e-3;
opts.gamma = 4e-3; opts.delta = 4e-3;
opts.kernel_size = 41; opts.xk_iter = 7; opts.gamma_correct = 1.0;
blur = imread('../Data_Text/img_CVPR2024/ker_six/CVPR24_blur.png'); y = im2double(blur); yg = im2double(rgb2gray(blur));

% convert the strcut opts to GPU
lambda_grad = gpuArray(single(lambda_grad));
lambda_surface = gpuArray(single(lambda_surface));
opts.gamma = gpuArray(single(opts.gamma));
opts.delta = gpuArray(single(opts.delta));

%% deblurring process

% blind deblurring
tic;
[kernel, interim_latent] = blind_deconv(yg, lambda_surface, lambda_grad, opts);
time1 = toc;
fprintf(['(1) The blur kernel estimation takes ', num2str(time1), ' seconds.\n']);

% non-blind deblurring
opts.sigma = 5 / 255;
opts.reg_str = 8e-3;
opts.sigma = gpuArray(single(opts.sigma));
opts.reg_str = gpuArray(single(opts.reg_str));

tic;
Latent_cho = deconv_outlier(y, kernel, opts.sigma, opts.reg_str); % cho's non-blind method
%Latent_cho = whyte_deconv(y, kernel); % whyte's non-blind method
toc;

% normailze the kernel
k = kernel - min(kernel(:));
k = k ./ max(k(:));

% convert data to CPU
interim_latent = gather(interim_latent);
Latent_cho = gather(Latent_cho);
k = gather(k);

% save the results
saving_path = './Real_world/';
imwrite(Latent_cho, [saving_path, 'L2-Welsch_CVPR2024_deblur.png']);
imwrite(interim_latent, [saving_path, 'L2-Welsch_CVPR2024_interim.png']);
imwrite(k, [saving_path, 'L2-Welsch_CVPR2024_kernel.png']);
