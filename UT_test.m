function UT_test(type)
    if strcmp(type,'wiki')
        mu = [12.3; 7.6];
        C = [1.44 0; 0 2.89];
        nonlinfunc = @(x) to_polar(x);
    elseif strcmp(type,'sc')    
        mu = [2; 2];
        covar = 0.8;
        C = [1 covar; covar 1];
        theta = 1.9;
        k = 10;
        beta = 1.1;        
        nonlinfunc = @(x) MP2SC(x,theta,beta,k);
    end
    
    nTrial = 1000;
    samples = mvnrnd(mu',C,nTrial); 
    % size(samples)
    % scsamp = MP2SC(samples,theta,beta,k);
    scsamp = nonlinfunc(samples);
    % size(scsamp)
    mean(scsamp)
    cov(scsamp)
    corr(scsamp)
    
    [utm, utc, sp, tsp] = unscentedTransform(mu, C, nonlinfunc)
    corrcov(utc)
end

function pol = to_polar(cart)
    x = cart(:,1);
    y = cart(:,2);
    r = sqrt(x.^2 + y.^2);
    theta = atan(y./x);
    pol = [r theta];
end

function sc = MP2SC(v,theta,beta,k)
    sc = v - theta;
    sc(sc < 0) = 0;
    sc = sc.^beta;
    sc = k * sc;
    %sc = floor(sc);
end