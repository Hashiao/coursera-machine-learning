function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
m_inv = 1/m;
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
for i = 1 : m
    J = J + m_inv*(-1*y(i)*log( sigmoid(X(i,:)*theta)) - (1-y(i))*log(1 - sigmoid(X(i,:)*theta)));
end

Reg_J = 0;

for itheta = 2 : size(theta)
    Reg_J = Reg_J + lambda*0.5*m_inv*theta(itheta)*theta(itheta);
end

J = J + Reg_J;

for itheta = 1 : size(theta)
    for i = 1 : m
        grad(itheta) = grad(itheta) + m_inv*(sigmoid(X(i,:)*theta) - y(i))*X(i, itheta);
    end
    if itheta ~= 1
        grad(itheta) = grad(itheta) + lambda*m_inv*theta(itheta);
    end
end







% =============================================================

end
