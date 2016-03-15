function sampleTransformations(nSamp,sampleSize)
    close all
    plotRow = 1;
    plotCol = 4;

    % get many bivariate normal samples as membrane potential
    nBin = 1;
    nTrial = 10000;    
    mu = [2 2];
    C = [1 0.5; 0.5 1];
    dim = length(mu);
    
    theta = 1.9;
    k = 10;
    beta = 1.1;
    
    standardSigmaPoints = [0 -sqrt(3/2) -sqrt(3/2); sqrt(2) -sqrt(1/2) -sqrt(1/2)];    
    sigmaPoints = zeros(size(standardSigmaPoints));
    transformedSigmaPoints = zeros(size(sigmaPoints));
    srC = sqrtm(C);
    UT_mean = zeros(dim,1);
    for i = 1:size(sigmaPoints,2)
        sigmaPoints(:,i) = srC * standardSigmaPoints(:,i) + mu';
        transformedSigmaPoints(:,i) = MP2SC(sigmaPoints(:,i),theta,beta,k);
        UT_mean = UT_mean + transformedSigmaPoints(:,i) / dim;
    end
    
    UT_cov = zeros(size(C));
    for i = 1:size(sigmaPoints,2)
        actDiff = transformedSigmaPoints(:,i) - UT_mean;
        UT_cov = UT_cov + (actDiff * actDiff') / dim;
    end
    UT_mean
    UT_cov    
    
    samples = reshape(mvnrnd(mu,C,nBin*nTrial),[nBin nTrial 2]);   
%     for i=1:nBin
%         cov(squeeze(samples(i,:,:)))
%     end        
    
    subplot(plotRow,plotCol,1)
    ndhist(samples(1,:,1),samples(1,:,2));
    title('V_i','FontSize',16)
    
    % rectify
    samples = samples - theta;
    samples(samples < 0) = 0;
    subplot(plotRow,plotCol,2)
    scatter(samples(1,:,1),samples(1,:,2))
    title('V_i^{\theta+}','FontSize',16)
    
    % transform with k and beta
    samples = k * (samples .^ beta);
    subplot(plotRow,plotCol,3)
    scatter(samples(1,:,1),samples(1,:,2))
    title('r_i','FontSize',16)
        
    % sum
    samples = squeeze(sum(samples,1));
    subplot(plotRow,plotCol,4)
    ndhist(samples(:,1),samples(:,2));
    title('r','FontSize',16)
    
    % floor
    samples = floor(samples);
    mean(samples)
    cov(samples)
end

function sc = MP2SC(v,theta,beta,k)
    sc = v - theta;
    sc(sc < 0) = 0;
    sc = floor(k * (sc.^beta));
end