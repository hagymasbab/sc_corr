function [UT_mu, UT_cov, sigmaPoints, transformedSigmaPoints] = unscentedTransform(mu, C, nonlinfunc)
    dim = length(mu);
    srC = sqrtm(C);
    
    sigmaPoints = zeros(dim,2*dim+1);    
    for i = 1:size(sigmaPoints,2)-1
        row = ceil(i/2);
        if mod(i,2) == 0
            sgn = 1;
        else
            sgn = -1;
        end
        sigmaPoints(:,i) = sqrt(2) * sgn * srC(row,:)' + mu;        
    end
    sigmaPoints(:,end) = mu;
    
    transformedSigmaPoints = nonlinfunc(sigmaPoints')';
    UT_mu = mean(transformedSigmaPoints,2);
    UT_cov = cov(transformedSigmaPoints',1);
end