function Z = projectData(X, U, K)
%PROJECTDATA Computes the reduced data representation when projecting only 
%on to the top k eigenvectors
%   Z = projectData(X, U, K) computes the projection of 
%   the normalized inputs X into the reduced dimensional space spanned by
%   the first K columns of U. It returns the projected examples in Z.
%
m = size(X, 1);
% You need to return the following variables correctly.
Z = zeros(size(X, 1), K);  % size(X, 1) m examples

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the projection of the data using only the top K 
%               eigenvectors in U (first K columns). 
%               For the i-th example X(i,:), the projection on to the k-th 
%               eigenvector is given as follows:
%                    x = X(i, :)';
%                    projection_k = x' * U(:, k);
%
for i = 1:m
    x = X(i, :)';  % (1,2)' = (2,1)
    Z(i,:) = U(:,1:K)' * x;   % (2,1)' * (2,1) = (1,2)*(2,1) = (1,1)  
end



% =============================================================

end