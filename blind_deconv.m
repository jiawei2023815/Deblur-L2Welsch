function [kernel, interim_latent] = blind_deconv(y, lambda_surface, lambda_grad, opts)
% the function for solving blind image deblurring process:
% \min_{u,h} AL(h*u-f, \xi)+lambda*||\nabla u||_{0}+mu*Phi(u)+gamma*||h||_{2}^{2}+delta*||\nabla h||_{0}

%% Input & Output

% Input
% @y: blurred image
% @lambda_grad
% @lambda_surface
% @opts

% Output:
% @kernel
% @interim_latent

%%
% gamma correction
if opts.gamma_correct ~= 1
    y = y .^ opts.gamma_correct;
end

% setting the maximum iteration level
ret = sqrt(0.5);
ret = gpuArray(single(ret));
maxitr = max(floor(log(5 / min(opts.kernel_size)) / log(ret)), 0);
num_scales = maxitr + 1;
fprintf('Maximum iteration level is %d\n', num_scales);

% compuate the blur kernel size for each iteration layer
retv = ret .^ (0 : maxitr);
k1list = ceil(opts.kernel_size * retv);
k1list = k1list + (mod(k1list, 2) == 0);
k2list = ceil(opts.kernel_size * retv);
k2list = k2list + (mod(k2list, 2) == 0);

% iterative solution using a coarse-to-fine method
for s = num_scales : -1 : 1
    
    % estimate the kernel of each layer
    if (s == num_scales)
        % in the first iteration layer, initialize the blur kernel directly
        ks = init_kernel(k1list(s));
        k1 = k1list(s);
        k2 = k1;
    else
        % for non-first iteration layers
        % compute the blur kernel size of the current layer using function resizeKer( ) with the blur kernel size of the previous layer
        k1 = k1list(s);
        k2 = k1;
        ks = resizeKer(ks, 1 / ret, k1list(s), k2list(s));
    end
    
    % perform down-sampling on latent image
    cret = retv(s);
    ys = downSmpImC(y, cret);
    
    fprintf('Processing scale %d/%d; Kernel size %dx%d; Image size %dx%d\n', s, num_scales, k1, k2, size(ys, 1), size(ys, 2));
    if (s == num_scales)
        [~, ~, threshold] = threshold_pxpy_v1(ys, max(size(ks)));
        % Initialize the parameter:
        if threshold < lambda_grad / 10 && threshold ~= 0
            lambda_grad = threshold;
            %lambda_surface = threshold_image_v1(ys);
            lambda_surface = lambda_grad;
        end
    end
    
    % call the main function for blind deblurring to update the
    % coefficients of the blur kernel, latent image, ||\nabla u||_{0}, and the coefficients of Phi(u).
    [ks, lambda_surface, lambda_grad, interim_latent] = blind_deconv_main(ys, ks, lambda_surface, lambda_grad, threshold, opts);
    
    % centering and normalizing the kernel
    ks = adjust_psf_center(ks);
    ks(ks(:) < 0) = 0;
    sumk = sum(ks(:));
    ks = ks ./ sumk;
    
    % set elements in the kernel to 0 that satisfy a certain condition
    % if opts.k_thresh>0, set the elements less than ratio of the maximum element to threshold opts.k_thresh to 0
    % if opts.k_thresh=0, set the negative elements to 0
    if (s == 1)
        kernel = ks;
        if opts.k_thresh > 0
            kernel(kernel(:) < max(kernel(:)) / opts.k_thresh) = 0;
        else
            kernel(kernel(:) < 0) = 0;
        end
        kernel = kernel / sum(kernel(:));
    end
end

end

%% initialize the kernel, only used in the first iteration
function [k] = init_kernel(minsize)
k = zeros(minsize, minsize);
k = gpuArray(single(k));
k((minsize - 1) / 2, (minsize - 1) / 2 : (minsize - 1) / 2 + 1) = 1 / 2;
end

%% down sample
function sI = downSmpImC(I, ret)

if (ret == 1)
    sI = I;
    return
end

sig = 1 / pi * ret;

g0 = (-50 : 50) * 2 * pi;
g0 = gpuArray(single(g0));
sf = exp(-0.5 * g0 .^ 2 * sig ^ 2);
sf = sf / sum(sf);
csf = cumsum(sf);
csf = min(csf, csf(end : -1 : 1));
ii = csf > 0.05;

sf = sf(ii);
sum(sf);

I = conv2(sf, sf', I, 'valid');

[gx, gy] = meshgrid([1 : 1 / ret : size(I, 2)], [1 : 1 / ret : size(I, 1)]);

I = gather(I);
gx = gather(gx);
gy = gather(gy);
sI = interp2(I, gx, gy, 'bilinear');
sI = gpuArray(single(sI));

end

%% to adjust the size of the blur kernel, the function fixsize( ) needs to be called.
function k = resizeKer(k, ret, k1, k2)

k = imresize(k, gather(ret), 'bicubic');
k = max(k, 0);
k = fixsize(k, k1, k2);
if max(k(:)) > 0
    k = k / sum(k(:));
end

end

%% using the resizeKer function directly will obtain the blur kernel through rounding up by a multiple.
% this function can be used to adjust to the correct size.
function nf = fixsize(f, nk1, nk2)

[k1, k2] = size(f);

while((k1 ~= nk1) || (k2 ~= nk2))
    
    if (k1 > nk1)
        s = sum(f, 2);
        if (s(1) < s(end))
            f = f(2 : end, :);
        else
            f = f(1 : end - 1, :);
        end
    end
    
    if (k1 < nk1)
        s = sum(f, 2);
        if (s(1) < s(end))
            tf = zeros(k1 + 1, size(f, 2));
            tf = gpuArray(single(tf));
            tf(1 : k1, :) = f;
            f = tf;
        else
            tf = zeros(k1 + 1, size(f, 2));
            tf = gpuArray(single(tf));
            tf(2 : k1 + 1, :) = f;
            f = tf;
        end
    end
    
    if (k2 > nk2)
        s = sum(f, 1);
        if (s(1) < s(end))
            f = f(:, 2 : end);
        else
            f = f(:, 1 : end - 1);
        end
    end
    
    if (k2 < nk2)
        s = sum(f, 1);
        if (s(1) < s(end))
            tf = zeros(size(f, 1), k2 + 1);
            tf = gpuArray(single(tf));
            tf(:, 1 : k2) = f;
            f = tf;
        else
            tf = zeros(size(f, 1), k2 + 1);
            tf = gpuArray(single(tf));
            tf(:, 2 : k2 + 1) = f;
            f = tf;
        end
    end
    
    [k1, k2] = size(f);
    
end

nf = f;

end
