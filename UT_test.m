function UT_test()
    mu = [2 2];
    C = [1 0.5; 0.5 1];
    theta = 1.9;
    k = 10;
    beta = 2;
    nTrial = 100000;
    
    samples = mvnrnd(mu,C,nTrial); 
    scsamp = MP2SC(samples,theta,beta,k);
    mean(scsamp)
    cov(scsamp)
    
    [utm, utc] = unscentedTransform(mu', C, @(x) MP2SC(x,theta,beta,k))
end

function pol = to_polar(cart)
    x = cart(1);
    y = cart(2);
    r = sqrt(x^2 + y^2);
    theta = atan(y/x);
    pol = [r; theta];
end

function sc = MP2SC(v,theta,beta,k)
    sc = v - theta;
    %sc(sc < 0) = 0;
    sc = k * (sc.^beta);
    %sc = floor(sc);
end