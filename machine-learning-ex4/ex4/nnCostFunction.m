function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));  % (25, 401)
Theta2_grad = zeros(size(Theta2));  % (10, 26)

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%% part one : costFuntion
%  Feedforward the neural network and return the cost in the
%  variable J
X = [ones(m, 1) X];   % (5000, 400) -> (5000, 401)
a_1 = X;
Z_2 = a_1 * Theta1';    % (5000, 25)
a_2 = sigmoid(Z_2);  
a_2 = [ones(m, 1) a_2];   % (5000, 26)
Z_3 = a_2 * Theta2';      % (5000, 10)
a_3 = sigmoid(Z_3);       % (5000, 10)

% map the vector y to matrix Y_map
Y_map = zeros(m, num_labels);  % 每一行是 只有一个是1
for i = 1:m
    Y_map(i, y(i)) = 1;   % 每一行 是一个y
end    
% log(h(x_i)_k ) = a_3;  
J = (1/m)*sum(sum(-1*Y_map.*log(a_3) - (1-Y_map).*log(1 - a_3)));

% Regularization J 
Theta1_noBias = Theta1(:,2:end);  % (25, 400)
Theta2_noBias = Theta2(:,2:end);  % (10, 25)
J = J + lambda*0.5*(1/m)*(sum(sum(Theta1_noBias.^2)) + sum(sum(Theta2_noBias.^2)));


%% part two : grad
% Implement the backpropagation algorithm to compute the gradients
% Theta1_grad and Theta2_grad.

% set DELTA (l, i, j) == 0
DELTA_1 = zeros(size(Theta1));  % (25, 401)
DELTA_2 = zeros(size(Theta2));  % (10, 26)
for im = 1:m
   % forward prop
   delta_3 = (a_3(im,:) - Y_map(im,:))'; % (10, 1) column vector    Theta2_noBias
   delta_2 = (Theta2_noBias')*delta_3 .* sigmoidGradient(Z_2(im,:))';  % (25, 1) column vector
   DELTA_2 = DELTA_2 + delta_3*(a_2(im,:));   % (10,26)
   DELTA_1 = DELTA_1 + delta_2*(a_1(im,:));   % (25,401)  the former is output_num  the rear is input_num
end
Theta1_grad = DELTA_1/m;   % (25, 401)
Theta2_grad = DELTA_2/m;   % (10, 26)


% Regularization Theta_grad
Theta1_Reg = lambda*Theta1_noBias/m; % (25, 400)
Theta1_Reg = [zeros(size(Theta1_Reg, 1), 1) Theta1_Reg];  % map it to (25, 401)
Theta1_grad = Theta1_grad + Theta1_Reg;

Theta2_Reg = lambda*Theta2_noBias/m; % (10, 25)
Theta2_Reg = [zeros(size(Theta2_Reg, 1), 1) Theta2_Reg];  % map it to (10, 26)
Theta2_grad = Theta2_grad + Theta2_Reg;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
