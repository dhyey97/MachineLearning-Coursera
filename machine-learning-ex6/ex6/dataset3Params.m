function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;
error=1.0;
C_list=[0.01 0.03 0.1 0.3 1 3 10 30];
err_res=zeros(length(C_list)^2,3);
k=0;
predictions=zeros(length(yval),1);
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


for i=1:length(C_list)
  for j=1:length(C_list)
    k=k+1;
     C_i=C_list(i);
    sigma_j=C_list(j);
    model= svmTrain(X, y, C_i, @(x1, x2) gaussianKernel(x1, x2, sigma_j));
    predictions=svmPredict(model,Xval);
    error=mean(double(predictions ~= yval));
    err_res(k,:)=[error C_i sigma_j];
    
  endfor
endfor
sort_res=sortrows(err_res,1);
C=sort_res(1,2);
sigma=sort_res(1,3);



% =========================================================================

end
