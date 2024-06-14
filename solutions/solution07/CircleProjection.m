%% Initialization
% Generate points on a circle
X = cos(0:0.1:2*pi);
Y = sin(0:0.1:2*pi);
labels = [1 ones(1, length(X))*2];

% and one point in the centre
X = [0 X];
Y = [0 Y];
data = [X; Y]';

% Notes about the comparison here:
%   - Unfortunately, setting a seed does not work for the t-SNE approach. The results still differ. This makes reproducibility a bit harder
%   - The loss of SNE changes much less compared to t-SNE. It seems sufficient to only run the algorithm a few times
%   - On the other hand, t-SNE has much higher variations of the loss. Therefore, more runs are performed here
%   - This comparison is not strictly fair but it is assumed that in both cases a global optimum is reached (for SNE because not much changes and for
%   t-SNE because of the many repetitions)

%% SNE plot
rng(1337, 'combRecursive');
bestLoss = realmax;

for i = 1:20
    [projCurrent, loss] = sne(data, 1, 3);
    loss
    if loss < bestLoss
        bestLoss = loss;
        projSNE = projCurrent;
    end
end
bestLoss

%% t-SNE plot
rng(1337, 'combRecursive'); % @Matlab: why does this not work?
lossesTSNE = zeros(1000, 1);
projectionsTSNE = cell(length(lossesTSNE), 1);

parfor i = 1:length(lossesTSNE)
    [projCurrent, loss] = tsne(data, 'Algorithm', 'Exact', 'NumDimensions', 1, 'Perplexity', 3);
    lossesTSNE(i) = loss;
    projectionsTSNE{i} = projCurrent;
end

[loss, idx] = min(lossesTSNE);
loss
projTSNE = projectionsTSNE{idx};

%% Save data and let Mathematica do the visualization
save('comparison_result.mat', 'data', 'projTSNE', 'projSNE');

%%
figure;
gscatter(projTSNE, ones(length(projTSNE), 1), labels);
