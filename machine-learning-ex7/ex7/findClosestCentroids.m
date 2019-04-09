function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%
% 
% Set K
K = size(centroids, 1);  % three centroids

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);    % a training set containing 300 Xs

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%
% (16384,3)   (16,3)
m = size(X,1);  % 16384
cur_norm_set = zeros(K,1);   % 16,1
min_norm = 0;
for thisx = 1:m
    for thiscen = 1:K
        cur_norm_set(thiscen) = norm(X(thisx,:) - centroids(thiscen,:)).^2;
    end
    min_norm = min(cur_norm_set);
    tmp = find(cur_norm_set == min_norm);
    idx(thisx) = tmp(1);  % in case there are the same norm in cur_norm_set
    cur_norm_set = zeros(K,1);
    min_norm = 0;
end






% =============================================================

end

