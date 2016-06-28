function ag = singleMultiCorr()
    load('ecker_data_v1_binned_static')
    sess = 8;
    [nUnit, nCond, nBin, nTrial] = size(data{sess}.spikes);
    grouplabels = unique(data{sess}.tetrode);
    nChannel = length(grouplabels);
    groupspikes = zeros(nChannel, nCond, nBin, nTrial);
    for ch = 1:nChannel
        trains = data{sess}.spikes(data{sess}.tetrode == grouplabels(ch),:,:,:);
        groupspikes(ch,:,:,:) = sum(trains,1);
    end      
    
    colors = {'blue','red','yellow','magenta','green','cyan'};
    
    within_group_corrs = [];
    across_group_corrs = [];
    
    wg = cell(1,nChannel);
    ag = cell(nChannel,nChannel);
    for g = 1:nChannel
        wg{g} = [];
        for g2 = 1:nChannel
            ag{g,g2} = [];
        end
    end
    
    for c = 1:nCond
        SC_unit = squeeze(sum(data{sess}.spikes(:,c,:,:),3));
        CM_unit = corr(zscore(SC_unit'));
        SC_group = squeeze(sum(groupspikes(:,c,:,:),3));
        CM_group = corr(zscore(SC_group'));
        for u = 1:nUnit
            g1 = find(grouplabels == data{sess}.tetrode(u));
            for u2 = u+1:nUnit
                g2 = find(grouplabels == data{sess}.tetrode(u2));
                if g1 == g2
                    within_group_corrs = [within_group_corrs; CM_unit(u,u2)];
                    wg{g1} = [wg{g1}; CM_unit(u,u2)];
                else
                    across_group_corrs = [across_group_corrs; CM_group(g1,g2) CM_unit(u,u2)];
                    ag{g1,g2} = [ag{g1,g2}; CM_group(g1,g2) CM_unit(u,u2)];
                    ag{g2,g1} = [ag{g2,g1}; CM_group(g1,g2) CM_unit(u,u2)];
                end
            end
        end
    end
    
    subplot(2,2,1)
    hist(within_group_corrs,linspace(-1,1,40));
    xlim([-1 1])
    subplot(2,2,2)
    scatter(across_group_corrs(:,1),across_group_corrs(:,2))
    xlabel('channel')
    ylabel('unit')
    
    for g1 = 1:nChannel   
        if sum(data{sess}.tetrode == grouplabels(g1)) == 1
            continue
        end
        subplot(2,2,3)
        hi = histogram(wg{g1},linspace(-0.5,0.5,40));        
        xlim([-0.5 0.5])
        hi.Normalization = 'probability';
        hi.FaceColor = colors{g1};
        hold on
        for g2 = g1+1:nChannel
            if sum(data{sess}.tetrode == grouplabels(g2)) == 1
                continue
            end
            subplot(2,2,4)
            scatter(ag{g1,g2}(:,1),ag{g1,g2}(:,2),'MarkerFaceColor',colors{g1},'MarkerEdgeColor',colors{g2},'LineWidth',2);
            hold on
        end
    end
end