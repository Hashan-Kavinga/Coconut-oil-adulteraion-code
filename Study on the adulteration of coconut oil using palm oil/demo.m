%Author: Sanjaya Lakmal

%%
%%Initialization
clear all;
close all;

class1 = [];
class2 = [];
class3 = [];
class4 = [];
class5 = [];
class6 = [];
class7 = [];
class8 = [];
class9 = [];
%%
%Constructing a functional relationship between Bhattacharrya distance

%load the preprocessed datasets
for sample_no = 1:5
set_no = num2str(sample_no);
load(['calibration_set' set_no '.mat']);
sample = double(pca_sample);
sample = sample(:,1:9,:);

class1 = [class1; sample(:,:,1);sample(:,:,2);sample(:,:,3)];
class2 = [class2; sample(:,:,4);sample(:,:,5);sample(:,:,6)];
class3 = [class3; sample(:,:,7);sample(:,:,8);sample(:,:,9)];
class4 = [class4; sample(:,:,10);sample(:,:,11);sample(:,:,12)];
class5 = [class5; sample(:,:,13);sample(:,:,14);sample(:,:,15)];
class6 = [class6; sample(:,:,16);sample(:,:,17);sample(:,:,18)];
class7 = [class7; sample(:,:,19);sample(:,:,20);sample(:,:,21)];
class8 = [class8; sample(:,:,22);sample(:,:,23);sample(:,:,24)];
class9 = [class9; sample(:,:,25);sample(:,:,26);sample(:,:,27)];
end

Total_Data = [ class1;class2; class3; class4;class5;class6;class7;class8;class9];
Data(:,:,1)=class1;
Data(:,:,2)=class2;
Data(:,:,3)=class3;
Data(:,:,4)=class4;
Data(:,:,5)=class5;
Data(:,:,6)=class6;
Data(:,:,7)=class7;
Data(:,:,8)=class8;
Data(:,:,9)=class9;

% fisher discriminant analysis

no_classes = 9;
no_dimension = 9;
classwise_mean=mean(Data);
total_mean=mean(Total_Data);

Sb=zeros(no_dimension);
for i=1:no_classes
    u=classwise_mean(:,:,i)-total_mean;
    Sb=Sb+ (u'*u);
end

Sw=zeros(no_dimension);
for i=1:no_classes
   Sw=Sw+ cov(Data(:,:,i)) ;
end

[U,S,V] = svd(Sw\Sb);
eigVal = ones(1, no_dimension, 'single') * S;

%Project 9D dataset to a 5D subspace
bigvec=U(:,1:5);
projected_Data=[];
for i=1:no_classes
    x=bigvec'*Data(:,:,i)';
    projected_Data=[projected_Data; x];
end    

for i=0:8       
Pclass(i+1,:,:) = [projected_Data(5*i+1,:);projected_Data(5*i+2,:);projected_Data(5*i+3,:);projected_Data(5*i+4,:);projected_Data(5*i+5,:)]';
end

%Calculate Bhattacharrya distance wrt authentic coconut oil 
rng(100);
ref_class = reshape(Pclass(1,randperm(13500,900),:),[900,5]);
for i=1:9
    for j=0:14
        comp_class = reshape(Pclass(i,(900*j+1):900*(j+1),:),[900,5]);
        d(i,j+1) = b_distance(ref_class,comp_class);

    end
end

%normalize the data
m_d = max(d(:));
d = d/m_d;
d_mean = mean(d,2);
d(:,16) = d_mean;
t =[0, 5, 10, 15, 20, 25, 30, 35, 40]/100;

%plot the calibration data
figure(1);
hold on;
for i=1:15
     scatter(t,d(:,i),'r','Filled');
end

%fitting the data to a curve
grid on;
[xData, yData] = prepareCurveData( t, d_mean );
ft = fittype( 'poly2' );
opts = fitoptions( 'Method', 'LinearLeastSquares' );
opts.Lower = [-Inf -Inf  0];
opts.Upper = [Inf Inf  0];
[fitresult, gof] = fit( xData, yData, ft, opts );
h = plot( fitresult, 'b');
set(h,'LineWidth',2);
xlabel('Adulteration level (V_{palm oil} / V_{total})');
ylabel('Normalized Bhattacharyya Distance');
title('The variation of normalized Bhattacharyya distance with adulteration level');
legend off;


%%
%validation of the proposed model

%load the validation dataset
Data =[];
load('validation_set.mat');
sample = double(pca_sample);
sample = sample(:,1:9,:);

for i=0:15
    Data(:,:,i+1) =sample(:,:,5*i+1);
end

%project the dataset to the same 5D subspace 
projected_Data=[];
for i=1:16 
    x=bigvec'*Data(:,:,i)';
    projected_Data=[projected_Data; x];
end   

Pclass=[];
for i=0:15
Pclass(i+1,:,:) = [projected_Data(5*i+1,:);projected_Data(5*i+2,:);projected_Data(5*i+3,:);projected_Data(5*i+4,:);projected_Data(5*i+5,:)]';
end

d=[];
for i=1:16
        comp_class = reshape(Pclass(i,:,:),[900,5]);
        d(i,1) = b_distance(ref_class,comp_class) ;
end

%normalize the data
d = d/m_d;
t =[2, 4, 6, 8, 12, 14, 16, 18, 22, 24, 26, 28, 32, 34, 36, 38]/100;

%plot the validation set and the curve on same figure
figure(2);
hold on;
scatter(t,d(:,1),'r','Filled');
h = plot( fitresult, 'b');
set(h,'LineWidth',2);
xlabel('Adulteration level (V_{palm oil} / V_{total})');
ylabel('Normalized Bhattacharyya Distance');
title('Validation of the proposed algorithm');
grid on;
legend off;

%Calculate R2 and MSE of the validation set
y = d(:,1)';
yCalc = 1.016*t.^2+ 2.045*t;
Rsq = 1 - sum((y - yCalc).^2)/sum((y - mean(y)).^2);
mse = immse(y,yCalc);
m = msgbox(sprintf('R^2  = %0.4f\nMSE = %0.4f', Rsq, mse),'Performance','help');

