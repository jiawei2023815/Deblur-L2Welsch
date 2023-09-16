function S = Estimate_Interim_L0Surface(Im, kernel, lambda, mu, kappa)
% using this function to solve latent image problem:
% \min_{u} ||h*u-f||_{2}^{2}+lambda*||\nabla u||_{0}+mu*Phi(u)
% construct an equivalent optimization problem using HQS(Half Quadratic Splitting) algorithm:
% \min_{u,x,y} ||h*u-f||_{2}^{2}+lambda*||x||_{0}+mu*Phi(y)+alpha*||\nabla u-x||_{2}^{2}+beta*||u-y||_{2}^{2}
% splitting to three subproblems:
% (1) u-subproblem: \min_{u} ||h*u-f||_{2}^{2}+alpha*||\nabla u-x||_{2}^{2}+beta*||u-y||_{2}^{2}
% (2) x-subproblem: \min_{x} lambda*||x||_{0}+alpha*||\nabla u-x||_{2}^{2}
% (3) y-subproblem: \min_{y} mu*Phi(y)+beta*||u-y||_{2}^{2}

%% Input & Output

% Input:
% @Im: blurred image
% @kernel: blur kernel
% @lambda: coefficient of image gradient prior ||\nabla u||_{0}
% @mu: coefficient of prior Phi(u)
% @kappa: Update ratio in the iteration

% Output:
% @S: latent image

%%
if ~exist('kappa', 'var')
    kappa = 2.0;
end

kappa = gpuArray(kappa);
S = Im;
beta_max = 1e+2;
beta_max = gpuArray(beta_max);
fx = [1, -1]; fy = [1; -1];
fx = gpuArray(fx); fy = gpuArray(fy);
[N, M, D] = size(Im);
sizeI2D = [N, M];
otfFx = psf2otf(gather(fx), sizeI2D);
otfFy = psf2otf(gather(fy), sizeI2D);
otfFx = gpuArray(single(otfFx));
otfFy = gpuArray(single(otfFy));

kernel = gpuArray(kernel);
KER = psf2otf(gather(kernel), sizeI2D);
KER = gpuArray(single(KER));
Den_KER = abs(KER) .^ 2;

Denormin2 = abs(otfFx) .^ 2 + abs(otfFy ) .^ 2;
if D > 1
    Denormin2 = repmat(Denormin2, [1, 1, D]);
    KER = repmat(KER, [1, 1, D]);
    Den_KER = repmat(Den_KER, [1, 1, D]);
end
Normin1 = conj(KER) .* fft2(S);

alpha = 2 * lambda;
alpha_max = 2 ^ 3;
alpha_max = gpuArray(alpha_max);
Outiter = 0;

while alpha < alpha_max
    
    % solving y-subproblem
    lam_surface = 100 * lambda / alpha;
    r1 = 0.1;
    r2 = min(0.1 * lam_surface, 0.1);
    tol = 1e-3;
    Maxit = 100;
    r1 = gpuArray(single(r1));
    r2 = gpuArray(single(r2));
    tol = gpuArray(single(tol));
    Maxit = gpuArray(single(Maxit));
    u = BeltramiPD2(S, 1, lam_surface, r1, r2, tol, Maxit ); % call function BeltramiPD2( ) to solve for latent image u
    beta = 2 * mu;
    while beta < beta_max
        Outiter = Outiter + 1;
        
        % solving x-subproblem
        h = [diff(S, 1, 2), S(:, 1, :) - S(:, end, :)];
        v = [diff(S, 1, 1); S(1, :, :) - S(end, :, :)];
        t = h .^ 2 < mu / beta; h(t) = 0;
        t = v .^ 2 < mu / beta; v(t) = 0;
        clear t;
        
        % solving u-subproblem
        Normin2 = [h(:, end, :) - h(:, 1, :), -diff(h,1,2)];
        Normin2 = Normin2 + [v(end, :, :) - v(1, :, :); -diff(v,1,1)];
        Denormin = Den_KER + beta * Denormin2 + alpha;
        FS = (Normin1 + beta * fft2(Normin2) + alpha * fft2(u)) ./ Denormin;
        S = real(ifft2(FS));
        beta = beta * kappa;
        
        if mu == 0
            break;
        end
        
    end
    
    alpha = alpha * kappa;
    
end

end

function [I, k] = BeltramiPD2( I0, beta, lambda, r1, r2, tol, Maxit )
% using primal-dual algorithm to solve subproblem-Phi(u)
% \min_{u} ||u-f||_{2}^{2}+lambda*Phi(u)

% Input:
% @I0: noised image
% @beta: ?
% @lambda: coefficinet of prior Phi(u)
% @r1, r2: step size for GD
% @tol: 
% @Maxit: maximum iteration

% - beta    aspect ratio of Beltrami embedding
% Output:
% @I: denoising results
% @k: maximum iteration

beta2 = beta .^ 2;
I = I0;
phix = zeros(size(I));
phiy = zeros(size(I));
phix = gpuArray(single(phix));
phiy = gpuArray(single(phiy));

primal = zeros(Maxit + 1, 1);
primal = gpuArray(single(primal));

I_x = ForwardX( I );
I_y = ForwardY( I );
N = sqrt(1 + beta2 * ( I_x .^ 2 + I_y .^ 2 ));
primal(1) = sum(lambda * N(:) + 1 / 2 * (I(:) - I0(:)) .^ 2);

% main loop
for k = 1 : Maxit
    
    % solve phi
    I_x = ForwardX( I );
    I_y = ForwardY( I );
    
    Den = real(beta * sqrt(beta2 - phix .^ 2 - phiy .^ 2));
    Den(Den < 0) = 0;
    phix = phix + r1 * (I_x .* Den - phix);
    phiy = phiy + r1 * (I_y .* Den - phiy);
    
    proj = sqrt( phix .^ 2 + phiy .^ 2 );
    proj( proj < 1 ) = 1;
    phix = phix ./ proj;
    phiy = phiy ./ proj;
    
    % solve I
    I = I + r2 * ( I0 - I + lambda * (BackwardX(phix) + BackwardY(phiy)));
    
    % convergence?
    N = sqrt(1 + beta2 * ( I_x .^ 2 + I_y .^ 2 ));
    primal(k + 1) = sum(lambda * N(:) + 1 / 2 * (I(:) - I0(:)) .^ 2);
    
    if abs(primal(k + 1) - primal(k)) < primal(1) * tol
        break;
    end
    
end

end

%% differential operator
function [dx] = BackwardX(v)

dx = [ v(:, 1, :), v(:, 2 : end - 1, :) - v(:, 1 : end - 2, :), - v(:,end-1,:)];

end

function [dy] = BackwardY(v)

dy = [ v(1, :, :); v(2 : end - 1, :, :) - v(1 : end - 2, :, :); - v(end-1,:,:)];

end

function [dy] = ForwardY(v)

dy = zeros(size(v));
dy = gpuArray(single(dy));
dy(1 : end - 1, :, :) = v(2 : end, :, :) - v(1 : end - 1, :, :);

end

function [dx] = ForwardX(v)

dx = zeros(size(v));
dx = gpuArray(single(dx));
dx(:, 1 : end - 1, :) = v(:, 2 : end, :) - v(:, 1 : end - 1, :);

end
