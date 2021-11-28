function testClusters(IDX, data, pattern, e)
    figure('Position',[10 10 900 600]);
    hold on;
    for index = 1 : length(pattern)
        cluster_dataset = zeros(size(data,1),size(data,2),'double');
        entry = 1;
        for row = 1 : size(IDX,1)
            if IDX(row) == pattern(index)
                cluster_dataset(entry,:) = data(row,:);
                entry = entry + 1;
            end
        end
        entry = entry - 1;
        cluster_dataset = cluster_dataset(1:entry,:);
        scatter(cluster_dataset(:,1),cluster_dataset(:,2),[],e(index,:),'o','filled');
    end
end