function distributionOfSampleMean(nSamp,shape,scale)
    close all

    trueMean = shape*scale;
    trueVar = shape*scale^2;
    trueSD = sqrt(shape*scale^2);
    
    sampleSizes = [20 50 100];
    nSampSize = length(sampleSizes);
    plotCol = 2;
    
    for i=1:nSampSize
        sampleSize = sampleSizes(i);
        
        sampleMeans = [];
        sampleVars = [];
        for s=1:nSamp
            actSamp = gamrnd(shape,scale,[sampleSize 1]);
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
        for j = 1:length(x)
            pdf(j) = pearsonType3(x(j),sampleSize,trueVar);
        end
        plot(x,pdf,'LineWidth',3)
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