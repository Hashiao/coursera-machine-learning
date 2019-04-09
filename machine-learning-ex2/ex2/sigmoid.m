function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));
[mRow, nCol] = size(z);
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).
for iRow = 1:mRow
    for jCol = 1:nCol
        g(iRow, jCol) = 1/(1+exp(1)^(-1*z(iRow, jCol)));
    end
end

% =============================================================

end
