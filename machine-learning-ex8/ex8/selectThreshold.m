function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%
%  yval(307,1)   yval==1 means it's anamalous
yval_inv = 1 - yval;
m = length(yval);
bestEpsilon = 0;
bestF1 = 0;
F1 = 0;
thispredictions = zeros(m,1);
stepsize = (max(pval) - min(pval)) / 1000;   % pval = G(Xval, mu, sigma2) [0,0.08]
for epsilon = min(pval):stepsize:max(pval)
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions
    thispredictions = (pval < epsilon);
    true_positive = sum(thispredictions .* yval);
    % true_positive = sum((thispredictions == 1) & (yval == 1));
    false_positive = sum(thispredictions .* yval_inv);
    false_nagtive = sum((1 - thispredictions) .* yval);
    
    prec = true_positive/(true_positive + false_positive);
    rec = true_positive/(true_positive + false_nagtive);

    F1 = 2*prec*rec/(prec + rec);

    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
