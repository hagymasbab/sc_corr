function sampleTransformations(nSamp,sampleSize)
    close all
    plotRow = 1;
    plotCol = 4;

    % get many bivariate normal samples as membrane potential
    nBin = 100;
    nTrial = 10000;    
    mu = [1 1];
    C = [1 0.5; 0.5 1];
    
    theta = 1.9;
    k = 10;
    beta = 1.1;
    
    samples = reshape(mvnrnd(mu,C,nBin*nTrial),[nBin nTrial 2]);   
%     for i=1:nBin
%         cov(squeeze(samples(i,:,:)))
%     end        
    
    subplot(plotRow,plotCol,1)
    ndhist(samples(1,:,1),samples(1,:,2));
    
    % rectify
    samples = samples - theta;
    samples(samples < 0) = 0;
    subplot(plotRow,plotCol,2)
    scatter(samples(1,:,1),samples(1,:,2))
    
    % transform with k and beta
    samples = k * (samples .^ beta);
    subplot(plotRow,plotCol,3)
    scatter(samples(1,:,1),samples(1,:,2))
    
    % sum
    samples = squeeze(sum(samples,1));
    subplot(plotRow,plotCol,4)
    ndhist(samples(:,1),samples(:,2));
    
    % floor
end