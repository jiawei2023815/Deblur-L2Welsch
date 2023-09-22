%% Remark
% The Code is partially created based on the method described in the following paper 
%        Jun Liu, Ming Yan, and Tieyong Zeng,
%        Surface-Aware Blind Image Deblurring,
%        IEEE TPAMI, Vol. 43, No. 3, March 2021.
%
% Author: Jiawei Lu @IMU
% Date: 2023-04-30
% Email: 32236108@mail.imu.edu.cn

function [k, lambda_surface, lambda_grad, S] = blind_deconv_main(blur_B, k, lambda_surface, lambda_grad, threshold, opts)
% this function is using to solving blind image deblurring process:
% \min_{u,h} LW(h*u-f, \xi)+lambda*||\nabla u||_{0}+mu*Phi(u)+gamma*||h||_{2}^{2}+delta*||\nabla h||_{0}
% this optimization problem can be decomposed into two subproblems:
% (1) latent image estimation problem: \min_{u} ||h*u-f||_{2}^{2}+lambda*||\nabla u||_{0}+mu*Phi(u)
% (2) blur kernel estimation: \min_{h} [1-exp(-\frac{(h*u-f)^{2}}{2})]+gamma*||h||_{2}^{2}+delta*||\nabla h||_{0}
% subproblem (1) using the function Estimate_Interim_L0Surface( )
% subproblem (2) using the function Estimate_PSF_L2L0( )

%% Input & Output

% Input:
% @blur_B: blurred image
% @k: blur kernel
% @lambda_grad: the coefficient of image prior ||\nabla u||_{0}
% @lambda_surface: the coefficient of image prior Phi(u)
% @threshold:
% @opts: inputing structure

% Ouput:
% @k: blur kernel
% @S: latent image

dx = [-1, 1; 0, 0]; dy = [-1, 0; 1, 0];
dx = gpuArray(single(dx)); dy = gpuArray(single(dy));
gamma = opts.gamma;
delta = opts.delta;

H = size(blur_B, 1); W = size(blur_B, 2);
H = gpuArray(single(H)); W = gpuArray(single(W));
blur_B_w = wrap_boundary_Liu(blur_B, opt_fft_size([H W] + size(k) - 1));
blur_B_tmp = blur_B_w(1 : H, 1 : W, :);
Bx = conv2(blur_B_tmp, dx, 'valid');
By = conv2(blur_B_tmp, dy, 'valid');

%% alternating scheme for solving the latent image estimation problem and kernel estimation problem
for iter = 1 : opts.xk_iter
    
    % solve the latent image estimation problem
    S = Estimate_Interim_L0Surface(blur_B_w, k, lambda_surface, lambda_grad, 2.0);
    S = S(1 : H, 1 : W, :);
    
    [latent_x, latent_y, threshold] = threshold_pxpy_v1(S, max(size(k)), threshold);
    
    % solve the blur kernel estimation problem
    k_prev = k; multiple = 2; multiple = gpuArray(single(multiple));
    reweight_x = omega(conv2(latent_x, k, 'same') - Bx, opts.c); % reweight mask_x
    reweight_y = omega(conv2(latent_y, k, 'same') - By, opts.c); % reweight mask_y
    k = Estimate_PSF_L2L0(Bx, By, latent_x, latent_y, gamma, delta, k_prev, multiple, reweight_x, reweight_y);
    fprintf('Pruning isolated noise in kernel...\n');
    CC = bwconncomp(gather(k), 8);
    for ii = 1 : CC.NumObjects
        currsum = sum(k(CC.PixelIdxList{ii}));
        if currsum < .1
            k(CC.PixelIdxList{ii}) = 0;
        end
    end
    
    % update the coefficients of the prior term ||\nabla u||_{0} and Phi(u)
    if lambda_surface ~= 0
        lambda_surface = max(lambda_surface / 1.1, 1e-4);
    else
        lambda_surface = 0;
    end
    
    if lambda_grad ~= 0
        lambda_grad = max(lambda_grad / 1.1, 1e-4);
    else
        lambda_grad = 0;
    end
    
    % show the blur kernel, latent image, final clean image and reweighted mask
    if opts.isdisplay_est_kernel == 1
        S(S < 0) = 0;
        S(S > 1) = 1;
        set(gcf, 'unit', 'normalized', 'position', [0.25, 0.25, 0.5, 0.5]);
        tiledlayout(2, 2);
        nexttile; imshow(blur_B, [ ]); title('Blurred Image', 'FontSize', 17, 'FontName', 'Times New Roman');
        nexttile; imshow(S, [ ]); title('Interim Latent Image', 'FontSize', 17, 'FontName', 'Times New Roman');
        nexttile; imshow(k, [ ]); title('Estimated Kernel', 'FontSize', 17, 'FontName', 'Times New Roman');
        nexttile; imshow(reweight_x + reweight_y, [ ]); title('Rewieghted Mask', 'FontSize', 18, 'FontName', 'Times New Roman');
        pause(0.01);
    end
    
end

% normalize the kernel
k(k < 0) = 0;
k = k ./ sum(k(:));

end

%% the reweighted function of kernel estimation
function Omega = omega(x, a)

Omega = exp(-0.5 * (x / a) .^ 2);

end
