function [UT_mu, UT_cov] = unscentedTransform(mu, C, nonlinfunc)
    standardSigmaPoints = [0 -sqrt(3/2) sqrt(3/2); sqrt(2) -sqrt(1/2) -sqrt(1/2)];
    sigmaPoints = zeros(size(standardSigmaPoints));
    transformedSigmaPoints = zeros(size(sigmaPoints));
    srC = sqrtm(C);
    for i = 1:size(sigmaPoints,2)
        sigmaPoints(:,i) = srC * standardSigmaPoints(:,i) + mu;
        transformedSigmaPoints(:,i) = nonlinfunc(sigmaPoints(:,i));
    end
    UT_mu = mean(transformedSigmaPoints,2);
    UT_cov = cov(transformedSigmaPoints',1);
end