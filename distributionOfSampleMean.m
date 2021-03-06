function distributionOfSampleMean(nSamp,shape,scale)
    close all

    lsc = load('../majom/data/SC_nat_atoc100a01_bin10.mat');
    spikeCount = squeeze(sum(lsc.spikeCount(1,:,1:50),3))';
    N = length(spikeCount);
    
%     trueMean = shape*scale;
%     trueVar = shape*scale^2;
    trueMean = mean(spikeCount);
    trueVar = var(spikeCount);

    trueSD = sqrt(trueVar);
    
    sampleSizes = [20 50 100];
    nSampSize = length(sampleSizes);
    plotCol = 2;    
    
    for i=1:nSampSize
        sampleSize = sampleSizes(i);
        
        sampleMeans = [];
        sampleVars = [];
        for s=1:nSamp
            %actSamp = gamrnd(shape,scale,[sampleSize 1]);
            actSamp = spikeCount(chooseKfromN(sampleSize,N));
            sampleMeans = [sampleMeans; mean(actSamp)];
            sampleVars = [sampleVars; var(actSamp)];
        end

%         subplot(nSampSize,plotCol,(i-1)*plotCol+1)
%         x = linspace(0,5,100);
%         plot(x,gampdf(x,shape,scale),'LineWidth',2);        
%         if i==1
%             title(sprintf('Gamma pdf shape=%d scale=%d',shape,scale),'FontSize',16)
%         end

        subplot(nSampSize,plotCol,(i-1)*plotCol+1)
        histogram(sampleMeans,50,'Normalization','pdf')
        hold on
        limits = xlim();
        x = linspace(limits(1),limits(2),100);
        pdf = zeros(size(x));        
        for j = 1:length(x)
            pdf(j) = normpdf(x(j),trueMean,trueSD/sqrt(sampleSize));
        end
        plot(x,pdf,'LineWidth',3)            
        plot(trueMean * ones(2,1), ylim(), 'k', 'LineWidth', 3);
        if i==1
            title(sprintf('Sample means from %d samples',nSamp),'FontSize',16)
        end
        ylabel(sprintf('sample size = %d',sampleSize),'FontSize',16)
        
        subplot(nSampSize,plotCol,(i-1)*plotCol+2)
        histogram(sampleVars,50,'Normalization','pdf')
        hold on
        limits = xlim();
        x = linspace(limits(1),limits(2),100);
        pdf = zeros(size(x));
        gpdf = zeros(size(x));
        for j = 1:length(x)
            pdf(j) = pearsonType3(x(j),sampleSize,trueVar);
            gpdf(j) = gampdf(x(j), (sampleSize-1)/2, 2*trueVar/sampleSize);
        end
        plot(x,pdf,'LineWidth',3)
        plot(x,gpdf,'y','LineWidth',3)
        plot(trueVar * ones(2,1), ylim(), 'k', 'LineWidth', 3);
        if i==1
            title(sprintf('Sample variances from %d samples',nSamp),'FontSize',16)
        end
    end


    function pdf = pearsonType3(s,N,sigma)
        nomin = (N / (2 * sigma)) ^ ((N-1)/2);
        denom = gamma((N-1)/2);
        poly = s ^ ((N-3)/2);
        expon = exp(-N*s / (2*sigma));
        pdf = (nomin / denom) * poly * expon;
    end
end