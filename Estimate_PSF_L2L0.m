function psf = Estimate_PSF_L2L0(blurred_x, blurred_y, latent_x, latent_y, gamma, delta, psf_0, ~, reweight_x, reweight_y)
% using this function to solving blur kernel estimation problem:
% \min_{h} [1-exp(-\frac{(h*u-f)^{2}}{2})]+gamma*||k||_{2}^{2}+delta*||\nabla h||_{0}.
% constructing the equivalent problem by IRHQS(Iterative Reweighted Half Quadratic Splitting) algorithm:
% \min_{h, z} sum_{i, j}w_{i, j}(h*u-f)_{i, j}^{2}+gamma*||h||_{2}^{2}+delta*||z||_{0}+beta*||\nabla h-z||_{2}^{2}.
% splitting this problem into two subproblems:
% (1) h-subproblem: \min_{h} sum_{i, j}w_{i, j}(h*u-f)_{i, j}^{2}+gamma*||h||_{2}^{2}+beta*||\nabla h-z||_{2}^{2}, where w_{i, j} is the reweighted mask;
% (2) z-subproblem: \min_{z} delta*||z||_{0}+beta*||\nabla h-z||_{2}^{2}.

%% Input & output

% Input:
% @blurred_x: x-gradient of blurred image
% @blurred_y: y-gradient of blurred image
% @latent_x: x-gradinet of latent image
% @latent_y: y-gradinet of latent image
% @gamma: the coefficient of prior ||h||_{2}^{2}
% @delta: the coefficient of prior ||\nabla h||_{0}
% @psf_0:
% @multi:

% Output:
% @psf: final blur kernel

%% iteration setting
p.beta = 2 * delta;
beta_max = 100;
beta_max = gpuArray(single(beta_max));

latent_xf = fft2(latent_x);
latent_yf = fft2(latent_y);
blurred_xf = fft2(blurred_x);
blurred_yf = fft2(blurred_y);

%[M, N, K] = size(blurred_xf);

% compute |FFT(nabla_x)|^{2}+|FFT(nabla_y)|^{2}
fx = [1, -1]; fy = [1; -1]; % gradient filtering template
fx = gpuArray(single(fx)); fy = gpuArray(single(fy));
p.img_size = size(blurred_xf);
otfFx = psf2otf(gather(fx), p.img_size);
otfFy = psf2otf(gather(fy), p.img_size);
otfFx = gpuArray(single(otfFx));
otfFy = gpuArray(single(otfFy));
p.DTD = abs(otfFx) .^ 2 + abs(otfFy ) .^ 2;

% compute the w_x(i, j).*|FFT(nabla_x_u)|^{2}+w_y(i, j).*|FFT(nabla_y_u)|^{2}
%p.m = conj(latent_xf)  .* latent_xf + conj(latent_yf)  .* latent_yf;
p.m = reweight_x .* conj(latent_xf) .* latent_xf + reweight_y .* conj(latent_yf) .* latent_yf;

% compute the w_x(i, j).*FFT*(nabla_x_u).*FFT(nabla_x_f)+w_y(i, j).*FFT*(nabla_y_u).*FFT(nabla_y_f)
%b_f = conj(latent_xf)  .* blurred_xf + conj(latent_yf)  .* blurred_yf;
b_f = reweight_x .* conj(latent_xf) .* blurred_xf + reweight_y .* conj(latent_yf) .* blurred_yf;

psf_size = size(psf_0); p.psf_size = psf_size;

psf = fspecial('gaussian', psf_size, 5); % initing the kernel
psf = gpuArray(single(psf));

while p.beta < beta_max
    
    % z-subproblem
    g_x = [diff(psf, 1, 2), psf(:, 1, :) - psf(:, end, :)];
    g_y = [diff(psf, 1, 1); psf(1, :, :) - psf(end, :, :)];
    % solving z by hard thresholding operator
    t = abs(g_x) < sqrt(delta / p.beta); g_x(t) = 0;
    t = abs(g_y) < sqrt(delta / p.beta); g_y(t) = 0;
    clear t;
    
    % h-subproblem
    gx_f = psf2otf(gather(g_x), p.img_size);
    gy_f = psf2otf(gather(g_y), p.img_size);
    gx_f = gpuArray(single(gx_f));
    gy_f = gpuArray(single(gy_f));
    napla_g = conj(otfFx) .* gx_f + conj(otfFy) .* gy_f;
    A = b_f + p.beta * napla_g;
    s = A ./ (p.m + gamma + p.beta * p.DTD);
    psf = otf2psf(gather(s), p.psf_size);
    psf = gpuArray(single(psf));
    
    p.beta = p.beta * 2;
    
    % normailze the kernel
    psf(psf < max(psf(:)) * 0.05) = 0;
    psf = psf / sum(psf(:));
    psf = abs(psf);
    if delta == 0
        break;
    end
    
end

end
