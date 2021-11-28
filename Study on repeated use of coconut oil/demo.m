%Author: Yasiru Ranasinghe

close all;
clear all;
clc
%% Load sample data
trialID = randperm(1,1) ; % Generate a random sample number
trial = load(strcat('pca_sample',num2str(trialID),'.mat')) ; % Load data structure
% fprintf('Current train sample is from %s\n', strcat('pca_sample',num2str(trialID),'.mat')) ;
trial = double(trial.pca_sample) ; % Extract sample data

clearvars trialID
%% Parameter initialization
ACC = zeros(2,2,'double') ;
train_size = 50 + 5 * (randperm(4,1) - 1) ; % Generate a random sample size
frames = size(trial,3) ; sBands = size(trial,2) ;

%% Construct train, test and validation datasets
train_indices = sort(randperm(size(trial,1), train_size)) ;
test_indices = setdiff(1:size(trial,1), train_indices) ;
train_set = permute(trial(train_indices,:,:),[2,1,3]) ;
test_set = permute(trial(test_indices,:,:),[2,1,3]) ;
% Construct the training dataset
train_set = reshape(train_set, sBands, numel(train_set)/sBands)' ;
% Construct the testing daataset
test_set = reshape(test_set, sBands, numel(test_set)/sBands)' ;

%% Construct dimension reduction and transformation matrix via LDA
fprintf('\n--------------------TRAINING PHASE--------------------\n')
fprintf('Current train sample size is %d\n', train_size) ;
startT = tic() ;
class_mu = zeros(frames/3, sBands, 'double') ; % class means matrix
for class = 1 : size(class_mu,1)
    start_row = 1 + train_size*3*(class-1) ;
    stop_row = train_size*3*class ;
    class_mu(class, :) = mean(train_set(start_row:stop_row,:), 1) ;
end

data_mu = mean(class_mu, 1) ; % train dataset mean

% Between classes scatter matrix
Sb = train_size*3*(class_mu-repmat(data_mu,size(class_mu,1),1))'*...
    (class_mu-repmat(data_mu,size(class_mu,1),1));
Sw = train_set;
for class = 1 : size(class_mu,1)
    start_row = 1 + train_size*3*(class-1) ;
    stop_row = train_size*3*class ;
    Sw(start_row:stop_row,:)...
    = Sw(start_row:stop_row,:) - repmat(class_mu(class,:),train_size*3,1) ;
end

Sw = Sw'*Sw ; % Within classes scatter matrix
[Fvec,Fval] = eigs(Sw\Sb,5) ; % Eigen decomposition
Fval = Fval*ones(size(Fval,1),1) ; 
red_dims = min(3,size(Fvec,2)) ;

fprintf('Linear Discriminant analysis is performed.\n')

transMatrix = real(Fvec(:,1:red_dims)) ; % Transformation matrix
red_class_mu = class_mu * transMatrix ; % Adjusting for representation
if red_class_mu(1,1) < red_class_mu(6,1)
    transMatrix(:,1) = -transMatrix(:,1) ;
end
if red_class_mu(1,2) > red_class_mu(2,2)
    transMatrix(:,2) = -transMatrix(:,2) ;
end
red_train_set = train_set * transMatrix ; % Dimesnsion reduced dataset

figure('Position',[10 10 900 600]) ;
% ColorMap for the figures with all six classes
sixColMap = [1, 0.4353, 0; 0.3010, 0.7450, 0.9330;...
            0.4660, 0.6740, 0.1880; 0.4940, 0.1840, 0.5560;...
            1, 0, 0; 0, 0.4470, 0.7410] ;
        
for class = 1 : size(class_mu,1)
    start_row = 1 + train_size*3*(class-1) ;
    stop_row = train_size*3*class ;
    scatter(red_train_set(start_row:stop_row,1),...
        red_train_set(start_row:stop_row,2),[],...
        sixColMap(class,:),'o','filled') ;
    hold on
end
axis on
legend('Pure oil','1 time heated','2 time heated',...
    '3 time heated','4 time heated','5 time heated');
title('Training dataset on reduced dimension') ;
%% Spectral clustering and sigma sweep

% Construct disparity matrix
D = zeros(size(train_set,1), 'double') ;
for row_index = 1 : size(D,1)
    for column_index = 1 : row_index
        d = red_train_set(row_index,:) - red_train_set(column_index,:) ;
        D(row_index,column_index) = d*d' ;
    end
end
D = D + D' ;
clearvars row_index column_index d
fprintf('Construction of disparity matrix is performed.\n')
% Construct affinity matrix
fprintf('Sigma sweep is performing...\n')
sigma_sweep = linspace(1,50,100) ;
sigma_sweep = double(sigma_sweep) ;
  
digit = num2str(numel(sigma_sweep)) ;
fprintf('Current sigma index : %s/100',digit) ;
mode_val = zeros(size(sigma_sweep,2),5,'double') ;
L = zeros(size(D,1),'double') ;
for sigma_index = 1 : size(sigma_sweep,2)
    ULC(digit,sigma_index) ;
    sigma = sigma_sweep(sigma_index) ;
    A = exp(-D/sigma) ;
    for diagonal_index = 1 : size(D,1)
        A(diagonal_index,diagonal_index) = 0 ;
        L(diagonal_index,diagonal_index)...
        = ones(1,size(D,1))*A(:,diagonal_index) ;
    end
    L = L^0.5 ;    
    A = (L\A)/L ;
    val = eigs(A,10) ;
    for mode = 1:9
        mode_val(sigma_index,mode) = abs(val(mode+1)-val(mode)) ;
    end
end
fprintf('\n') ;

%% Compute with optimum sigma values
sigma_index = find(mode_val(:,6) == max(mode_val(:,6))) ;
optimum_sigma = mean(sigma_sweep(sigma_index)) ;
L = zeros(size(train_set,1),'double') ;
A = exp(-D/optimum_sigma) ;
for diagonal_index = 1 : size(train_set,1)
        A(diagonal_index,diagonal_index) = 0 ;
        L(diagonal_index,diagonal_index)...
             = ones(1,size(train_set,1))*A(:,diagonal_index) ;
end
A = L - A ;
L = L^0.5 ;
A = (L\A)/L ;
[vec,~] = eig(A,'vector') ;
vec = L\vec ;
vec = round(vec*1e4)*1e-4 ;
    
%% 4 mode clustering
spectral_vectors = vec(:,1:4) ;
IDX = kmeans(spectral_vectors,4) ;
stopTr = toc(startT) ;

% Creating the groundtruth for training dataset
IDX = reshape(IDX, numel(IDX)/6, 6) ;
pattern = mean(IDX, 1) ; % Pattern index after kmeans

groundTruth = [1,2,2,3,4,4] ; % Default groundtruth vector
if abs(pattern(3) - pattern(2)) > abs(pattern(3) - pattern(4))
    groundTruth(3) = groundTruth(4) ;
end
if abs(pattern(5) - pattern(4)) < abs(pattern(5) - pattern(6))
    groundTruth(5) = groundTruth(4) ;
end

pattern = round(pattern) ;
dumPat = pattern ;
if ~isempty(setdiff(1:4, pattern)) 
    for index = 2 : numel(groundTruth)
        for jndex = 1 : index - 1
            if (pattern(index) == pattern(jndex)) &&...
                    (groundTruth(index) ~= groundTruth(jndex))
                dumPat(index) = 0 ; break
            end
        end
    end
end

groundTruth = dumPat ;
IDX = reshape(IDX, numel(IDX), 1) ;
anomaly = red_train_set(IDX~=...
    kron(groundTruth',ones(3*train_size,1)),:) ;

fourColMap = [1,1,0; 1, 0.7490, 0; 1, 0.4353, 0; 0.5020, 0, 0] ;
testClusters(IDX,red_train_set,pattern([1,2,4,6]), fourColMap);
figure(2) ; hold on ;
scatter(anomaly(:,1), anomaly(:,2),'ko','filled') ;
axis on
ACC(1,2) = 1-size(anomaly,1)/size(D,1);
legend('0^{th} umbrella cluster','1^{st} umbrella cluster',...
    '2^{nd} umbrella cluster','3^{rd} umbrella cluster');
title('Training dataset 4 mode clasification') ;
%% 6 mode clustering
spectral_vectors = vec(:,1:6) ;
IDX = kmeans(spectral_vectors,6) ;

% Creating the groundtruth for training dataset
IDX = reshape(IDX, numel(IDX)/6, 6) ;
pattern = mean(IDX, 1) ; % Pattern index after kmeans
groundTruth = 1:6 ; % Default groundtruth vector

pattern = round(pattern) ;
dumPat = pattern ;
if ~isempty(setdiff(1:6, pattern)) 
    for index = 2 : numel(groundTruth)
        for jndex = 1 : index - 1
            if (pattern(index) == pattern(jndex))
                dumPat([index,jndex]) = [0,0] ; break
            end
        end
    end
end

groundTruth = dumPat ;

IDX = reshape(IDX, numel(IDX), 1) ;
anomaly = red_train_set(IDX~=...
    kron(groundTruth',ones(3*train_size,1)),:) ;

testClusters(IDX,red_train_set,pattern, sixColMap);
figure(3) ; hold on ;
scatter(anomaly(:,1), anomaly(:,2),'ko','filled') ;
axis on
ACC(1,1) = 1-size(anomaly,1)/size(D,1);
legend('Pure oil','1 time heated','2 time heated',...
    '3 time heated','4 time heated','5 time heated');
title('Training dataset 6 mode clasification') ;
fprintf('Spectral clustering is performed.\n')
%% Validation of the algorithm
fprintf('\n--------------------VALIDATION PHASE--------------------\n')
startT = tic() ;
test_size = 500 ;
fprintf('Current test sample size is %d\n', test_size) ;
test_set = test_set(sort(randperm(size(test_set,1),test_size)),:) ;
test_set = test_set * transMatrix ; % Dimesnsion reduced dataset
fprintf('Dimension reduction is performed.\n')

% Construct disparity matrix
D = zeros(size(train_set,1), 'double') ;
for row_index = 1 : size(D,1)
    for column_index = 1 : row_index
        d = red_train_set(row_index,:) - red_train_set(column_index,:) ;
        D(row_index,column_index) = d*d' ;
    end
end
D = D + D' ;
clearvars row_index column_index d
fprintf('Construction of disparity matrix is performed.\n')
L = zeros(size(train_set,1),'double') ;
A = exp(-D/optimum_sigma) ;
for diagonal_index = 1 : size(train_set,1)
        A(diagonal_index,diagonal_index) = 0 ;
        L(diagonal_index,diagonal_index)...
             = ones(1,size(train_set,1))*A(:,diagonal_index) ;
end
A = L - A ;
L = L^0.5 ;
A = (L\A)/L ;
[vec,val] = eig(A,'vector') ;
vec = L\vec ;
vec = round(vec*1e4)*1e-4 ;
    
%% 4 mode clustering
spectral_vectors = vec(:,1:4) ;
IDX = kmeans(spectral_vectors,4) ;
stopVa = toc(startT) ;

% Creating the groundtruth for training dataset
IDX = reshape(IDX, numel(IDX)/6, 6) ;
pattern = mean(IDX, 1) ; % Pattern index after kmeans

groundTruth = [1,2,2,3,4,4] ; % Default groundtruth vector
if abs(pattern(3) - pattern(2)) > abs(pattern(3) - pattern(4))
    groundTruth(3) = groundTruth(4) ;
end
if abs(pattern(5) - pattern(4)) < abs(pattern(5) - pattern(6))
    groundTruth(5) = groundTruth(4) ;
end

pattern = round(pattern) ;
dumPat = pattern ;
if ~isempty(setdiff(1:4, pattern)) 
    for index = 2 : numel(groundTruth)
        for jndex = 1 : index - 1
            if (pattern(index) == pattern(jndex)) &&...
                    (groundTruth(index) ~= groundTruth(jndex))
                dumPat(index) = 0 ; break
            end
        end
    end
end

groundTruth = dumPat ;
IDX = reshape(IDX, numel(IDX), 1) ;
anomaly = red_train_set(IDX~=...
    kron(groundTruth',ones(3*train_size,1)),:) ;

fourColMap = [1,1,0; 1, 0.7490, 0; 1, 0.4353, 0; 0.5020, 0, 0] ;
testClusters(IDX,red_train_set,pattern([1,2,4,6]), fourColMap);
figure(4) ; hold on ;
scatter(anomaly(:,1), anomaly(:,2),'ko','filled') ;
axis on
ACC(2,2) = 1-size(anomaly,1)/size(D,1);
legend('0^{th} umbrella cluster','1^{st} umbrella cluster',...
    '2^{nd} umbrella cluster','3^{rd} umbrella cluster');
title('Validation dataset 4 mode clasification') ;
fprintf('Spectral clustering is performed.\n\n')
%% 6 mode clustering
spectral_vectors = vec(:,1:6) ;
IDX = kmeans(spectral_vectors,6) ;

% Creating the groundtruth for training dataset
IDX = reshape(IDX, numel(IDX)/6, 6) ;
pattern = mean(IDX, 1) ; % Pattern index after kmeans
groundTruth = 1:6 ; % Default groundtruth vector

pattern = round(pattern) ;
dumPat = pattern ;
if ~isempty(setdiff(1:6, pattern)) 
    for index = 2 : numel(groundTruth)
        for jndex = 1 : index - 1
            if (pattern(index) == pattern(jndex))
                dumPat([index,jndex]) = [0,0] ; break
            end
        end
    end
end

groundTruth = dumPat ;

IDX = reshape(IDX, numel(IDX), 1) ;
anomaly = red_train_set(IDX~=...
    kron(groundTruth',ones(3*train_size,1)),:) ;

testClusters(IDX,red_train_set,pattern, sixColMap);
figure(5) ; hold on ;
scatter(anomaly(:,1), anomaly(:,2),'ko','filled') ;
axis on
ACC(2,1) = 1-size(anomaly,1)/size(D,1);
legend('Pure oil','1 time heated','2 time heated',...
    '3 time heated','4 time heated','5 time heated');
title('Validation dataset 6 mode clasification') ;
%% Print results to the command window
ACC = ACC*100 ;
fprintf('--------------------COMPUTATIONAL RESULTS--------------------\n\n') ;
fprintf('Time elpased for training phase : %0.4f sec\n', stopTr) ;
fprintf('4-mode clustering accuracy : %0.2f\n', ACC(1,2)) ;
fprintf('6-mode clustering accuracy : %0.2f\n\n', ACC(1,1)) ;
fprintf('Time elpased for validation phase : %0.4f sec\n', stopVa) ;
fprintf('4-mode clustering accuracy : %0.2f\n', ACC(2,2)) ;
fprintf('6-mode clustering accuracy : %0.2f\n', ACC(2,1)) ;