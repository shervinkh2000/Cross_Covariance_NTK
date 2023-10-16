#\\\ Standard libraries:
import numpy as np
import torch; torch.set_default_dtype(torch.float64)
import torch.nn as nn
import torch.optim as optim
import copy
import matplotlib.pyplot as plt
import pandas as pd
import os
import random


#\\\ Own libraries:
import myModules

########################################################


print(os.getcwd())

path = os.getcwd() + '/timeseries_data.txt'

df = pd.read_csv(path, header = None)



df = pd.DataFrame(df[0].str.split(' ').tolist())

df = df.apply(pd.to_numeric)


data = df.to_numpy()
############################################

########################training####################################

All_losses_test = dict()
All_losses_train = dict()


All_losses_test['Cxx_GraphFilter'] = []
All_losses_train['Cxx_GraphFilter'] = []

All_losses_test['Cxy_GraphFilter'] = []
All_losses_train['Cxy_GraphFilter'] = []

All_losses_test['Cxx_twolayer_GNN'] = []
All_losses_train['Cxx_twolayer_GNN'] = []

All_losses_test['Cxy_twolayer_GNN'] = []
All_losses_train['Cxy_twolayer_GNN'] = []


for g in range(10):
    nEpochs = 3
    batchSize = 5
    learningRate = 0.025*0.5

    N = np.shape(data)[1]
    nsamples = np.shape(data)[0]

    nTrain = 1000

    nTest = 100

    delta_t = 1

    perm = np.random.permutation(nsamples - 1 - delta_t)


    #################train data#################

    X = np.zeros((nTrain, N))
    Y = np.zeros((nTrain, N))

    for j in range(nTrain):
        X[j, :] = data[perm[j], :]
        Y[j, :] = data[perm[j] + delta_t, :]

    X = X/(np.linalg.norm(X, axis = 1, keepdims = True))

    X = np.expand_dims(X, axis = 1)
    Y = np.expand_dims(Y, axis = 1)

    xTrain = torch.from_numpy(X)
    yTrain = torch.from_numpy(Y)


    #################test data#################

    X = np.zeros((nTest, N))
    Y = np.zeros((nTest, N))

    for j in range(nTest):
        X[j, :] = data[perm[j + nTrain], :]
        Y[j, :] = data[perm[j + nTrain] + delta_t, :]

    X = X/(np.linalg.norm(X, axis = 1, keepdims = True))

    X = np.expand_dims(X, axis = 1)
    Y = np.expand_dims(Y, axis = 1)

    xTest = torch.from_numpy(X)
    yTest = torch.from_numpy(Y)


    ##################shift operators#########################
    Xt = (xTrain.numpy()).reshape((nTrain, -1))
    Yt = (yTrain.numpy()).reshape((nTrain, -1))


    print(np.shape(Xt))
    print(np.shape(Yt))

    Cxy = Xt.T@Yt

    Cxy = Cxy/np.linalg.norm(Cxy)

    Cxx = Xt.T@Xt

    Cxx = Cxx/np.linalg.norm(Cxx)

    R = np.random.randn(N, N)
    R = R/np.linalg.norm(R)



    ################################
    ######## LOSS FUNCTION #########
    ################################

    def MSELoss(yHat,y):
        mse = nn.MSELoss()
        return mse(yHat, y)    



    ################################
    ######## ARCHITECTURES #########
    ################################

    architectures = dict()
    learningRates = dict()
    losses_test = dict()


    K = 2
    F = 50


    arch = myModules.GraphFilter(Cxx, K, 1, 1)
    architectures['Cxx_GraphFilter'] = arch
    learningRates['Cxx_GraphFilter'] = 50*learningRate

    arch = myModules.GraphFilter(Cxy, K, 1, 1)
    architectures['Cxy_GraphFilter'] = arch
    learningRates['Cxy_GraphFilter'] = 50*learningRate

    arch = nn.Sequential(myModules.GraphFilter(Cxx, K, 1, F), torch.nn.LeakyReLU(), myModules.GraphFilter(Cxx ,K, F, 1))
    architectures['Cxx_twolayer_GNN'] = arch
    learningRates['Cxx_twolayer_GNN'] = learningRate

    arch = nn.Sequential(myModules.GraphFilter(Cxy, K, 1, F), torch.nn.LeakyReLU(), myModules.GraphFilter(Cxy ,K, F, 1))
    architectures['Cxy_twolayer_GNN'] = arch
    learningRates['Cxy_twolayer_GNN'] = learningRate



    ################################
    ########### TRAINING ###########
    ################################

    validationInterval = 30

    nValid = int(np.floor(0.1*nTrain))
    xValid = xTrain[0:nValid,:,:]
    yValid = yTrain[0:nValid,:,:]
    xTrain = xTrain[nValid:,:,:]
    yTrain = yTrain[nValid:,:,:]
    nTrain = xTrain.shape[0]


    # Declaring the optimizers for each architectures
    optimizers = dict()
    for key in architectures.keys():
        optimizers[key] = optim.Adam(architectures[key].parameters(), lr = learningRates[key])

    if nTrain < batchSize:
        nBatches = 1
        batchSize = [nTrain]
    elif nTrain % batchSize != 0:
        nBatches = np.ceil(nTrain/batchSize).astype(np.int64)
        batchSize = [batchSize] * nBatches
        while sum(batchSize) != nTrain:
            batchSize[-1] -= 1
    else:
        nBatches = np.int(nTrain/batchSize)
        batchSize = [batchSize] * nBatches
    batchIndex = np.cumsum(batchSize).tolist()
    batchIndex = [0] + batchIndex

    epoch = 0 # epoch counter

    # Store the training...
    lossTrain = dict()
    costTrain = dict()
    lossValid = dict()
    costValid = dict()

    lossTrainSmooth = dict()

    lossTest = dict()
    # ...and test variables
    lossTestBest = dict()
    costTestBest = dict()
    lossTestLast = dict()
    costTestLast = dict()

    bestModel = dict()

    for key in architectures.keys():
        lossTrain[key] = []
        costTrain[key] = []
        lossValid[key] = []
        costValid[key] = []

        lossTest[key] = []

        lossTrainSmooth[key] = []


    while epoch < nEpochs:
        randomPermutation = np.random.permutation(nTrain)
        idxEpoch = [int(i) for i in randomPermutation]
        print("")
        print("Epoch %d" % (epoch+1))

        batch = 0

        while batch < nBatches:
            # Determine batch indices
            thisBatchIndices = idxEpoch[batchIndex[batch]
                                        : batchIndex[batch+1]]

            # Get the samples in this batch
            xTrainBatch = xTrain[thisBatchIndices,:,:]
            yTrainBatch = yTrain[thisBatchIndices,:,:]

            if (epoch * nBatches + batch) % validationInterval == 0:
                print("")
                print("    (E: %2d, B: %3d)" % (epoch+1, batch+1),end = ' ')
                print("")

            for key in architectures.keys():
                # Reset gradients
                architectures[key].zero_grad()

                # Obtain the output of the architectures
                yHatTrainBatch = architectures[key](xTrainBatch)

                # Compute loss
                lossValueTrain = MSELoss(yHatTrainBatch, yTrainBatch)

                # Compute gradients
                lossValueTrain.backward()

                # Optimize
                optimizers[key].step()

                costValueTrain = np.sqrt(lossValueTrain.item())

                lossTrain[key] += [lossValueTrain.item()]
                costTrain[key] += [costValueTrain]

                # Print:
                if (epoch * nBatches + batch) % validationInterval == 0:
                    with torch.no_grad():
                        # Obtain the output of the GNN
                        yHatValid = architectures[key](xValid)
                        yHatTest = architectures[key](xTest)
                        yHatTrain = architectures[key](xTrain)
                    # Compute loss
                    lossValueValid = MSELoss(yHatValid, yValid)

                    lossValueTest = MSELoss(yHatTest, yTest)

                    lossValueTrainSmooth = MSELoss(yHatTrain, yTrain)

                    # Compute accuracy:
                    costValueValid = np.sqrt(lossValueValid.item())

                    lossValid[key] += [lossValueValid.item()]
                    costValid[key] += [costValueValid]

                    lossTrainSmooth[key] += [lossValueTrainSmooth.item()]

                    lossTest[key] += [lossValueTest.item()]

                    print("\t" + key + ": %6.4f / %7.4f [T]" % (
                            costValueTrain,
                            lossValueTrain.item()) + " %6.4f / %7.4f [V]" % (
                            costValueValid,
                            lossValueValid.item()))

                    # Saving the best model so far
                    if len(costValid[key]) > 1:
                        if costValueValid <= min(costValid[key]):
                            bestModel[key] =  copy.deepcopy(architectures[key])
                    else:
                        bestModel[key] =  copy.deepcopy(architectures[key])

            batch+=1

        epoch+=1

    print("")

    for key in architectures.keys():
        All_losses_test[key] += [lossTest[key]]
        All_losses_train[key] += [lossTrainSmooth[key]]

    ##########################
    ########## PLOT ##########
    ##########################



#########################################
plt.figure()

legend = []

for key in All_losses_test.keys():

    legend += [key]

    #if it is a filter
    if key == 'Cxx_GraphFilter':
        plt.errorbar(np.arange(np.shape(np.array(All_losses_test[key]))[1]), np.mean(np.array(All_losses_test[key]), axis = 0), yerr = np.std(np.array(All_losses_test[key]), axis = 0), capsize = 2, alpha = 1, errorevery=(0, 4), color = 'lightsalmon')
    if key == 'Cxy_GraphFilter':
        plt.errorbar(np.arange(np.shape(np.array(All_losses_test[key]))[1]), np.mean(np.array(All_losses_test[key]), axis = 0), yerr = np.std(np.array(All_losses_test[key]), axis = 0), capsize = 2, alpha = 1, errorevery=(1, 4), color = 'firebrick')
    if key == 'Cxx_twolayer_GNN':
        plt.errorbar(np.arange(np.shape(np.array(All_losses_test[key]))[1]), np.mean(np.array(All_losses_test[key]), axis = 0), yerr = np.std(np.array(All_losses_test[key]), axis = 0), capsize = 2, alpha = 1, errorevery=(2, 4), color = 'powderblue')
    if key == 'Cxy_twolayer_GNN':
        plt.errorbar(np.arange(np.shape(np.array(All_losses_test[key]))[1]), np.mean(np.array(All_losses_test[key]), axis = 0), yerr = np.std(np.array(All_losses_test[key]), axis = 0), capsize = 2, alpha = 1, errorevery=(3, 4), color = 'steelblue')


plt.title('test loss average')
plt.legend(legend)
plt.show()

plt.close()
#########################################

plt.figure()

legend = []

for key in All_losses_train.keys():

    legend += [key]

    #if it is a filter
    #if it is a filter
    if key == 'Cxx_GraphFilter':
        plt.errorbar(np.arange(np.shape(np.array(All_losses_train[key]))[1]), np.mean(np.array(All_losses_train[key]), axis = 0), yerr = np.std(np.array(All_losses_train[key]), axis = 0), capsize = 2, alpha = 1, errorevery=(0, 4), color = 'lightsalmon')
    if key == 'Cxy_GraphFilter':
        plt.errorbar(np.arange(np.shape(np.array(All_losses_train[key]))[1]), np.mean(np.array(All_losses_train[key]), axis = 0), yerr = np.std(np.array(All_losses_train[key]), axis = 0), capsize = 2, alpha = 1, errorevery=(1, 4), color = 'firebrick')
    if key == 'Cxx_twolayer_GNN':
        plt.errorbar(np.arange(np.shape(np.array(All_losses_train[key]))[1]), np.mean(np.array(All_losses_train[key]), axis = 0), yerr = np.std(np.array(All_losses_train[key]), axis = 0), capsize = 2, alpha = 1, errorevery=(2, 4), color = 'powderblue')
    if key == 'Cxy_twolayer_GNN':
        plt.errorbar(np.arange(np.shape(np.array(All_losses_train[key]))[1]), np.mean(np.array(All_losses_train[key]), axis = 0), yerr = np.std(np.array(All_losses_train[key]), axis = 0), capsize = 2, alpha = 1, errorevery=(3, 4), color = 'steelblue')


plt.title('train loss average')
plt.legend(legend)
plt.show()

plt.close()