function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);   %(300,2)

% You need to return the following variables correctly.
centroids = zeros(K, n);    %(3,2)


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%
% C(thiscen) = idx(k)
for thiscen = 1:K   % dealing this centroid
    point_num = 0;
    for thisPoint = 1:m   % dealing this point
        if idx(thisPoint) == thiscen
            centroids(thiscen, :) = centroids(thiscen, :) + X(thisPoint, :);
            point_num = point_num + 1;
        end        
    end
    centroids(thiscen, :) = centroids(thiscen, :)/point_num;  
end








% =============================================================


end

