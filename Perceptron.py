'''
Jeremy Scott
A11180132
Percepton - from Homework #1
'''
import numpy
import matplotlib.pyplot as plt

'''
Before z-scoring or training Let us get the Data into a usable form so that:
    x_train/x_test -> [1,x1,x2,x3,x4] @ index n;  n-> the nth example
    t_train/t_test -> [label x5] @ index n 
'''
#get training/testing data
with open('iris_train','r') as f_train:         
    trainDataRows = f_train.read().split('\n')  
with open('iris_test','r') as f_test:
    testDataRows = f_test.read().split('\n')
    
#split data by newline    
trainDataList   = []
testDataList    = []
for i in range(len(trainDataRows)):
    trainDataList.append(trainDataRows[i].split(','))
    if i < len(testDataRows):   #since testing on less example's than training
        testDataList.append(testDataRows[i].split(','))
        
#Data into dictionaries. nth example # is key. [1,x1..x4] is value. 
#label in separate list, t for given labels here, y for predicted labels later
x_train = {}    #vector of training inputs, X0 = 1 always
t_train = []    #label for each training example "TEACHER"
x_test  = {}    #vector of testing inputs, X0 = 1
t_test  = []    #correct label for each test example (Compare Prediction to this)
for i in range(len(trainDataList)):
    x_train[i]=[]
    x_train[i].append(float(1))
    t_train.append(trainDataList[i][4])
    if i < len(testDataList):
        x_test[i]=[]
        x_test[i].append(float(1))
        t_test.append(testDataList[i][4])
    for j in range(0,4):
        x_train[i].append(float(trainDataList[i][j]))
        if i < len(testDataList):
            x_test[i].append(float(testDataList[i][j]))
#data is easily accessible now

'''
Part A) 
    Z-score the training data features
'''
#errorSquared: part of sum in numerator of standard deviation formula --tested GOOD
def errorSquared(number, average):
    return (number - average)**2

#average: average a list of numbers --tested GOOD      
def average(list_numbers):
    list_sum = float(0)
    N = len(list_numbers)
    for n in list_numbers:
        list_sum += n
    return list_sum/N

#standardDeviation: of a list of numbers, used in Z-score --tested GOOD
def standardDeviation(list_numbers):
    #make sure these are floats
    list_numbers = [float(i) for i in list_numbers]
    av = average(list_numbers)
    sd_sum = 0
    for i in list_numbers:
        sd_sum += errorSquared(i,av)
    #divide that by N (or N-1?)
    return (sd_sum/len(list_numbers))**(0.5)
    
#zScore: xi - feature data, ui - average of feature data, si -stdDev
def zScore(xi,ui,si):
    return (xi-ui)/si

#z-score all the training data
print 'Part A) \nZ-score the training data \nx_train before z-score: ',x_train
print 'x_test before z-score: ',x_test
cols = {} #column dict:  key = k/feature, value = list of column entries
for k in range(1,5): #k = feature number
    cols[k]=[]
    for i in range(len(x_train)):  #i = column number
        cols[k].append(x_train[i][k])
    ui  = average(cols[k])
    si  = standardDeviation(cols[k])
    for i in range(len(x_train)):
        x_train[i][k] = round(zScore(x_train[i][k],ui,si),4)     #zScore training data
        if(i < len(x_test)):
            x_test[i][k] = round(zScore(x_test[i][k],ui,si),4)   #zScore testing data
print 'x_train after Z-scored: ',x_train
print 'x_test after Z-scored: ',x_test

#replace old training columns with new z-scored training columns
for k in range(1,5):                #k = feature number
    cols[k]=[]
    for i in range(len(x_train)):   #i = column number
        cols[k].append(x_train[i][k])
'''
Part B) 
    6 plots for each dataset.  Each plot a comparison of a pair of features
    uncomment one at a time to see good graphs
'''
#training data

print 'Part B) \nPlot for testing data should pop up, if not, uncomment the section'
#Shows the 6 plots, one at a time
# for j in range(1,4):
#     for k in range(j+1,5):    #indexing for 6 pairs of features to plot
#         plt.plot(cols[j][0:35],cols[k][0:35], 'or')
#         plt.plot(cols[j][35:],cols[k][35:], 'o')
#         plt.xlabel('x%s'%j)
#         plt.ylabel('x%s'%k)
#         plt.axis([-4, 4, -4, 4])    #all z-scores should fit in this range
#         plt.title('Compare feature %s against feature %s'%(j,k))
#        plt.show()

#testing data
print 'Plot for training data should pop up, if not, uncomment the section'
#replace old training columns with new z-scored training columns
# for k in range(1,5): #k = feature number
#     cols[k]=[]
#     for i in range(len(x_test)):  #i = column number
#         cols[k].append(x_test[i][k])
# 
# #Shows the 6 testing plots, one at a time
# for j in range(1,4):
#     for k in range(j+1,5):    #indexing for 6 pairs of features to plot
#         plt.plot(cols[j][0:15],cols[k][0:15], 'or')
#         plt.plot(cols[j][15:],cols[k][15:], 'o')
#         plt.xlabel('x%s'%j)
#         plt.ylabel('x%s'%k)
#         plt.axis([-4, 4, -4, 4])    #all z-scores should fit in this range
#         plt.title('Compare feature %s against feature %s'%(j,k))
#         plt.show()         

'''
PERCEPTRON ALGORITHM
Part C)
    train the perceptron to classify the training data, noting the improvement
    in learning over epochs.  I'll be plotting this
'''
#function wTx:  calculate the dot product given weights and inputs
def wTx(w,x):
    summation = float(0)
    for i in range(len(w)):
        summation += w[i]*x[i]
    return summation


#TRAINING
#initialize weights, w[0] is -threshold or +bias, and learning rate
#set w0 to 50 to FORCE error after first iteration
w = [10,0.0,0.0,0.0,0.0]
Yj = float(0)
#alpha in w's update rule
learningRate = .01
#save epoch numbers and learning correct/(correct+incorrect) ratios
plotEpoch = []
plotLearning = []
for epochNum in range(0,20):
    #for all examples j in training set update the classifier
    for j in range(0, len(x_train)):
        #calculate Yj
        Yj = 1 if wTx(w,x_train[j]) >= 0 else 0
        Tj = 1 if t_train[j] == 'Iris-setosa' else 0
        #update weights
        for e in range(len(w)):
            w[e] = w[e]+ learningRate*(Tj-Yj)*x_train[j][e]
    
    #test Perceptron classifier w on x_train after each epoch
    correct = 0
    incorrect = 0
    for i in range(len(x_train)):
        if ((wTx(w,x_train[i])>=0 and t_train[i] == 'Iris-setosa') 
            or (wTx(w,x_train[i])<0 and t_train[i] == 'Iris-versicolor')):
            correct += 1.0
        else: incorrect += 1.0       
    plotEpoch.append(epochNum+1)
    plotLearning.append(100*correct/(correct+incorrect))
    #print 100*correct/(correct+incorrect)
    
plt.plot(plotEpoch,plotLearning)
plt.xlabel("Epoch number")
plt.ylabel("Percent Correct")
plt.axis([0,21,25,105])
plt.title("Learning improvement of perception with heavy bias(50) over 20 training cycles")
plt.show()

'''
Part D)
    test the classifer on the test data, report percentage accurate
        since I opted to try both 50 and 0 bias i'll report both
'''
correct = 0
incorrect = 0
for i in range(len(x_test)):
    if ((wTx(w,x_test[i])>=0 and t_test[i] == 'Iris-setosa') 
        or (wTx(w,x_test[i])<0 and t_test[i] == 'Iris-versicolor')):
        correct += 1.0
    else: incorrect += 1.0       

print 'percent correct on test data:', 100*correct/(correct+incorrect)
     
 


         
     
     
     
     
     
     
     
     
    
    
    
    
    
    
    