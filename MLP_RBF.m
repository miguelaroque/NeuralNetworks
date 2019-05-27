%%%%%%%%%%%%%%%%%%%%%%%%% Read Files %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The goal of this section is to read the MNIST Fashion Dataset
clear all, close all, clc

data = importdata("phpnBqZGZ.csv");
trainData = data.data(1:60000,1:784)';
trainLabel = data.data(1:60000,end);

testData = data.data(60001:end,1:784)';
testLabel = data.data(60001:end,end);

ims = reshape(trainData,[28,28,60000]);


% commented plot to check if images are rotated
%%% Plots %%%%%
% figure 
% imagesc(ims(:,:,1)); colormap(gray)
% figure
% imagesc(ims(:,:,1)'); colormap(gray)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%% Feature Analysis %%%
% Some relevant information is extrated from the analysis done. It is
% possible to identify patterns in the pixel intensity of the images. For
% each class, the resulting average pixel intensity reveal similarities
% among some classes -> see plots to be sure 

% find indices of each class
class0idx = find(trainLabel == 0);
class1idx = find(trainLabel == 1);
class2idx = find(trainLabel == 2);
class3idx = find(trainLabel == 3);
class4idx = find(trainLabel == 4);
class5idx = find(trainLabel == 5);
class6idx = find(trainLabel == 6);
class7idx = find(trainLabel == 7);
class8idx = find(trainLabel == 8);
class9idx = find(trainLabel == 9);

% plot histogram tha count samples for each class (6000 for each)
figure
histogram(trainLabel,'BinWidth',0.1)
ylabel('Sample Counts')
xlabel('Class')
grid on
% legend('Bar Count')
ylim([0,8000])
xlim([-1,10])

% Extract average pixel intensity among each class, i.e, for each pixel, an
% average is done across all images labelled as the same class

class0MatrixAvg = mean(trainData(:,class0idx),2);
class1MatrixAvg = mean(trainData(:,class1idx),2);
class2MatrixAvg = mean(trainData(:,class2idx),2);
class3MatrixAvg = mean(trainData(:,class3idx),2);
class4MatrixAvg = mean(trainData(:,class4idx),2);
class5MatrixAvg = mean(trainData(:,class5idx),2);
class6MatrixAvg = mean(trainData(:,class6idx),2);
class7MatrixAvg = mean(trainData(:,class7idx),2);
class8MatrixAvg = mean(trainData(:,class8idx),2);
class9MatrixAvg = mean(trainData(:,class9idx),2);

% Figure with subplots from all the similar groups identified:
%  1-> ('T-shirt','Pullover','Dress','Coat','Shirt')
%  2-> ('Trousers')
%  3-> ('Sandal','Sneaker','Ankle boot')
%  4-> ('Bag')
figure
subplot(2,2,1)
hold on
plot(class0MatrixAvg,'k','Linewidth',2)
plot(class2MatrixAvg,'b','Linewidth',2)
plot(class3MatrixAvg,'c','Linewidth',2)
plot(class4MatrixAvg,'m','Linewidth',2)
plot(class6MatrixAvg,'k--','Linewidth',2)
legend('T-shirt','Pullover','Dress','Coat','Shirt')
ylabel('Average Gray-level')
xlabel('Pixel (feature)')
ylim([0 280])
grid on
hold off

subplot(2,2,2)
hold on
plot(class1MatrixAvg,'r','Linewidth',2)
legend('Trousers')
ylabel('Average Gray-level')
xlabel('Pixel (feature)')
ylim([0 280])
grid on
hold off

subplot(2,2,3)
hold on
plot(class5MatrixAvg,'y','Linewidth',2)
plot(class7MatrixAvg,'r--','Linewidth',2)
plot(class9MatrixAvg,'m--','Linewidth',2)
legend('Sandal','Sneaker','Ankle boot')
ylabel('Average Gray-level')
xlabel('Pixel (feature)')
ylim([0 280])
grid on
hold off

subplot(2,2,4)
hold on
plot(class8MatrixAvg,'b--','Linewidth',2)
legend('Bag')
ylabel('Average Gray-level')
xlabel('Pixel (feature)')
ylim([0 280])
grid on
hold off

%% K-fold analysis for best model (net) choice

% K- fold is widely used in the model evaluation. In this section you
% should run manually different parameters to obtain different model metrics
% (accuracy, recall, sensibility, etc). You should aggregate the results of each run 
% in a table and decide the best model to go ahead.

%%%%%%%%%%%%%%%%%%%%%%% MLP or DNN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
MT = full(ind2vec(trainLabel'+1));

% for model validation a good practice to be used is the K-fold cross
% validation. Matlab is resource consuming so only a part of the
% dataset is used to validate the possible model
% Net hidden layers / hidden neurons to analyze

% Occam's Razor

% [256 128 64] % too dense I think
% [128 64] % test this one
% [64 32] % test this one
% [128] % test this one
% [64] % test this one

% K-fold cross validation (10 folds)
% dataset to test under the k-fold approach where only half of the dataset
% is analyzed to improve time of execution of the dataset. Nevertheless the
% scalability of the overal dataset is taken into account. 

trainDataKfold = [trainData(:,class0idx(1:3000)),...
                  trainData(:,class1idx(1:3000)),...
                  trainData(:,class2idx(1:3000)),...
                  trainData(:,class3idx(1:3000)),...
                  trainData(:,class4idx(1:3000)),...
                  trainData(:,class5idx(1:3000)),...
                  trainData(:,class6idx(1:3000)),...
                  trainData(:,class7idx(1:3000)),...
                  trainData(:,class8idx(1:3000)),...
                  trainData(:,class9idx(1:3000))];
              
trainLabelKfold=[trainLabel(class0idx(1:3000));...
                 trainLabel(class1idx(1:3000));...
                 trainLabel(class2idx(1:3000));...
                 trainLabel(class3idx(1:3000));...
                 trainLabel(class4idx(1:3000));...
                 trainLabel(class5idx(1:3000));...
                 trainLabel(class6idx(1:3000));...
                 trainLabel(class7idx(1:3000));...
                 trainLabel(class8idx(1:3000));...
                 trainLabel(class9idx(1:3000))];

p = randperm(30000); % 3000 * 10
trainDataKfold = trainDataKfold(:,p);
trainLabelKfold = trainLabelKfold(p);
K = 10;
CVO = cvpartition(trainLabelKfold,'k',10);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Definition of MLP net (using custom parameters, the patternet should be considered your own MLP)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
net = patternnet([128]); % to define your own net, I advise just to modify this patternet. Defining your own from the start can be tricky. 
net.divideFcn = 'dividerand';
net.performFcn = 'mse'; % mse is recommended
net.layers{1}.transferFcn = 'logsig'; % check and see 'hardlim', 'tansig'
net.layers{2}.transferFcn = 'softmax'; % check and see 'hardlim', 'tansig'
% net.layers{X}.transferFcn = 'logsig'; Where X is the layer
% net.layers{3}.transferFcn = 'logsig';
% net.trainFcn = 'trainlm'; % BE CAREFUL WITH THIS METRIC 'trainscg' is
% recommended you can try others but some will crash your computer (like 'trainlm')
net.divideParam.trainRatio = 0.8;
net.divideParam.valRatio = 0.2;
net.divideParam.testRatio = 0;
net.trainparam.epochs = 100;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Definition of RBF net. by definition it only have 2 layers, a radial basis layer
% and an output layer (Iam not sure if it must be purelin). 
% I am modifying a patternet to obtain the RBF net
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
RBFnet = patternnet([64]);
RBFnet.divideFcn = 'dividerand';
RBFnet.layers{1}.transferFcn = 'radbas'; % radial basis function layer
RBFnet.layers{2}.transferFcn = 'purelin'; % check and see 'softmax'
RBFnet.divideParam.trainRatio = 0.8;
RBFnet.divideParam.valRatio = 0.2;
RBFnet.divideParam.testRatio = 0;
RBFnet.trainparam.epochs = 100;


MT2 = full(ind2vec(trainLabelKfold'+1));


% Metrics for model evaluation MLP (if necessary, define more metrics)
acc = [];
prec = [];
sensi = [];
speci = [];
F_meas = [];

% Metrics for model evaluation RBF (if necessary, define more metrics)
RBFacc = [];
RBFprec = [];
RBFsensi = [];
RBFspeci = [];
RBFF_meas = [];

% Running K-fold for MLP and RBF
for i = 1:CVO.NumTestSets
   
    % Define partitions of K-fold to iterate
    
    trIdx = CVO.training(i);
    teIdx = CVO.test(i);
    
    trDataAux = trainDataKfold(:,trIdx);
    trLabel = MT2(:,trIdx);
    teDataAux = trainDataKfold(:,teIdx);
    teLabel = MT2(:,teIdx);
    teLabelvec = vec2ind(teLabel);
    
    %%%%%%%% Running MLP %%%%%%%%%%%%%
    
    clear net_incr % avoid incremental training
    [net_incr,tr] = train(net,trDataAux,trLabel);
    yPred = round(net_incr(teDataAux));
    
%     yPred = round(sim(net_incr,teDataAux));
    %index accuracy metrics for model choice
    yPredSparse = vec2ind(yPred);
    evalStats = Evaluate(teLabelvec', yPredSparse'); % EVAL = [accuracy sensitivity specificity precision recall f_measure gmean];
    
    plotconfusion(teLabel,yPred)
    
    acc = [acc;evalStats(1)];
    sensi = [sensi;evalStats(2)];
    speci = [speci;evalStats(3)];
    prec = [prec;evalStats(4)];
    F_meas = [F_meas;evalStats(6)];
    
%     (if necessary use ROC metrics)
%     % ROC curve for each class 
%     for ii = 1:10 % nr of classes
%         [X,Y,T,AUCauxClass] = perfcurve(teLabelvec', yPredSparse',ii); % if you want to see the curves per class use: figure; plot(X,Y)
%     end
    
    
    %%%%%%%% Running RBF %%%%%%%%%%%%%
    
    clear RBFnet_incr % avoid incremental training
    
    [RBFnet_incr,tr] = train(RBFnet,trDataAux,trLabel);
    RBFyPred = round(RBFnet_incr(teDataAux));
    RBFyPredSparse = vec2ind(RBFyPred);
    RBFevalStats = Evaluate(teLabelvec', RBFyPredSparse'); % EVAL = [accuracy sensitivity specificity precision recall f_measure gmean];
    
    plotconfusion(teLabel,RBFyPred)
    
    RBFacc = [RBFacc;RBFevalStats(1)];
    RBFsensi = [RBFsensi;RBFevalStats(2)];
    RBFspeci = [RBFspeci;RBFevalStats(3)];
    RBFprec = [RBFprec;RBFevalStats(4)];
    RBFF_meas = [RBFF_meas;RBFevalStats(6)];
    
end

% After manually check the best model, you csn proceed to classification
% with those parameters using an hold-out method for all dataset.


%% Classification of MNIST Fashion dataset using MLP or RBF net (Run 2 times one for MLP, other for RBF)

classificationModel = 'MLP';

% After the analysis of the best model via K-Fold, a hold-out method (70% for train, 
% 15% for validation and 15% for test from the training dataset) technique 
% is applied to train the full dataset with the best model:
% (I AM SUPPOSING THAT THE NET WITH 2 HIDDEN LAYER AND [128 64] IS BETTER): (probably it is not!!)
% choose the best model from the previous analysis
% Change network parameters:
if strcmp(classificationModel,'MLP')
    final_net = patternnet([128 64]);
    final_net.divideFcn = 'dividerand';
    final_net.performFcn = 'mse';
    final_net.layers{1}.transferFcn = 'logsig';
    final_net.layers{2}.transferFcn = 'logsig';
else
    final_net = patternnet([64]); % just one layer here
    final_net.divideFcn = 'dividerand';
    final_net.performFcn = 'mse';
    final_net.layers{1}.transferFcn = 'radbas'; %and that layer must be RBF
end
% net.layers{2}.transferFcn = 'tansig';
final_net.divideParam.trainRatio = 0.7;
final_net.divideParam.valRatio = 0.3;
final_net.divideParam.testRatio = 0;
final_net.trainparam.epochs = 300;
% final_net.plotFcns = {'plotperform', 'plottrainstate', 'ploterrhist', 'plotconfusion', 'plotroc'};


[netTrained,tr] = train(final_net,trainData,MT);
yPredFinal = round(sim(netTrained,testData));

yPredFinalvec = vec2ind(yPredFinal);
testLabelvec = [testLabel+1]';

evalStatsFinal = Evaluate(testLabelvec', yPredFinalvec');
accuracy = evalStatsFinal(1);
sensibility = evalStatsFinal(2);
specificity = evalStatsFinal(3);
precision = evalStatsFinal(4);
F_measure = evalStatsFinal(6);


figure
hold on
AUCauxClassFinal = [];
for jj = 1:10 % nr of classes
    [Xf,Yf,Tf,AUCaux3] = perfcurve(testLabelvec', yPredFinalvec',jj); % if you want to see the curves per class use: figure; plot(X,Y)
    rgb = rand(1,3);
    plot(Xf,Yf,'LineWidth',2,'Color',rgb)
end
legend('Class 1','Class2','Class3','Class4','Class5','Class6','Class7','Class8','Class9','Class10')
hold off


