function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples
m_inv = 1/m;
n = length(theta);
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));  % (n, 1)

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
J = 0.5*m_inv*sum((X*theta - y).^2);
J = J + 0.5*lambda*m_inv*sum(theta(2:n).^2);

%grad = m_inv*sum((X*theta - y).*X);  % without Reg
%grad(2:end) = grad(2:end) + lambda*m_inv*theta(2:end);


temp = theta;
temp(1) = 0;
grad = ((X*theta - y)' *X)' + lambda*temp;
grad = m_inv * grad;

% =========================================================================

grad = grad(:);

end
