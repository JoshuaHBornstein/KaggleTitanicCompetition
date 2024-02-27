import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

 

import os
#import tensorflow as tf
#import tensorflow_decision_forests as tfdf
#import keras, tensorflow

#load data
datapath = "/Users/joshBornstein/Documents/CS589/titanic/"
train = pd.read_csv(datapath + "train.csv")
feature_names = train.columns.values


#general data exploration
def broadData(train):
    print("Survival Rate: " + str(np.sum(train['Survived'] == 1) / len(train['Survived'])))

    for f in ["Sex", "Pclass"]:
        print("Survival Rate by " + f)
        print(train[[f, "Survived"]].groupby([f], as_index=False).mean().sort_values(by="Survived", ascending=False))
        print("")


    #ageGroups = [10, 20, 30, 40, 50, 100, -1]
    #ageGroups = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, -1]
    ageGroups = [15, 30, 50, 100, -1]
    ageGroupSize = []
    ageGroupSurvived = []
    for i in range(len(ageGroups) - 1):
        if i == 0:
            ageGroupSize.append(np.sum(train['Age'] <= ageGroups[i]))
            ageGroupSurvived.append(np.sum(train['Survived'][train['Age'] <= ageGroups[i]] == 1))
        else:
            ageGroupSize.append(np.sum(train['Age'] <= ageGroups[i]) - np.sum(train['Age'] <= ageGroups[i-1]))
            ageGroupSurvived.append(np.sum(train['Survived'][train['Age'] <= ageGroups[i]] == 1) - np.sum(train['Survived'][train['Age'] <= ageGroups[i-1]] == 1))
    #add unknown age group
    ageGroupSize.append(np.sum(train['Age'].isnull()))
    ageGroupSurvived.append(np.sum(train['Survived'][train['Age'].isnull()] == 1))
    for i in range(len(ageGroups)):
        if i == 0:
            print("Size of Age Group: 0-" + str(ageGroups[i]) + ": " + str(ageGroupSize[i]))
            print("Survival Rate of Age Group: 0-" + str(ageGroups[i]) + ": " + str(ageGroupSurvived[i]/ageGroupSize[i]))
            print()
        else:
            print("Size of Age Group: " + str(ageGroups[i-1]) + "-" + str(ageGroups[i]) + ": " + str(ageGroupSize[i]))
            print("Survival Rate of Age Group: " + str(ageGroups[i-1]) + "-" + str(ageGroups[i]) + ": " + str(ageGroupSurvived[i]/ageGroupSize[i]))
            print()

    #plot survival rate by age group
    plt.bar([str(ageGroups[i - 1]) + "-" + str(ageGroups[i]) if i > 0 else "0-" + str(ageGroups[i]) for i in range(len(ageGroups))], [ageGroupSurvived[i]/ageGroupSize[i] for i in range(len(ageGroups))])
    plt.title("Survival Rate by Age Group")
    plt.show()

    #

broadData(train)

def dataSelectedForFeature(d):
  #pclass
  class1 = d[d['Pclass'] == 1]
  class2 = d[d['Pclass'] == 2]
  class3 = d[d['Pclass'] == 3]
  broadData(class1)
  broadData(class2)
  broadData(class3)

  #embarked
  s = d[d['Embarked'] == 'S']
  c = d[d['Embarked'] == 'C']
  q = d[d['Embarked'] == 'Q']
  print("Presenting Data for Embarked = S")
  broadData(s)
  print("Presenting Data for Embarked = C")
  broadData(c)    
  print("Presenting Data for Embarked = Q")
  broadData(q)

  noCabin = d[d['Cabin'].isnull()]
  yesCabin = d[d['Cabin'].notnull()]
  print("Presenting Data for Cabin = Nan")
  broadData(noCabin)
  print("Presenting Data for Cabin != Nan")
  broadData(yesCabin)
  
  


#issue with age groups, 177 people have no age, 177/891 = 20% of data
#issue with cabin, 687 people have no cabin, 687/891 = 77% of data
#issue with embarked, 2 people have no embarked, 2/891 = 0.2% of data

#statistical analysis on whether having a recorded age is independent of survival
"""
import scipy.stats as stats
numAgeRecord = 714 #len(train['Age']) - np.sum(train['Age'].isnull())
numAgeNotRecord = 177 #np.sum(train['Age'].isnull())
numSurvivedAgeRecord = 290 
numSurvivedAgeNotRecord = 52
numDiedAgeRecord = 424
numDiedAgeNotRecord = 125
obs = np.array([[numSurvivedAgeRecord, numSurvivedAgeNotRecord], [numDiedAgeRecord, numDiedAgeNotRecord]])
chi2, p, dof, expected = stats.chi2_contingency(obs)
print("Chi Squared Test for Independence of Age Recorded and Survival")
print("Chi Squared: " + str(chi2))
print("P Value: " + str(p))
print("Degrees of Freedom: " + str(dof))
print("Expected: " + str(expected))
"""
#p value is 0.007, reject null hypothesis that age recorded and survival are independent, therefore, age recorded is a good predictor of survival, thus rather than replace missing ages with the mean age, I will replace them with a new category "U" for unknown

#statistical analysis on whether having a recorded cabin is independent of survival
"""
import scipy.stats as stats
numCabinNotRecord = 687 #np.sum(train['Cabin'].isnull()) 
numCabinRecord =  204 #len(train['Cabin']) - numCabinNotRecord
numSurvivedCabinRecord = 136
numSurvivedCabinNotRecord = 206
numDiedCabinRecord = 68
numDiedCabinNotRecord = 481
obs = np.array([[numSurvivedCabinRecord, numSurvivedCabinNotRecord], [numDiedCabinRecord, numDiedCabinNotRecord]])
chi2, p, dof, expected = stats.chi2_contingency(obs)
print("Chi Squared Test for Independence of Cabin Recorded and Survival")
print("Chi Squared: " + str(chi2))
print("P Value: " + str(p))
print("Degrees of Freedom: " + str(dof))
print("Expected: " + str(expected))"""
#p value is 6.7e-21 (basically 0), reject null hypothesis that cabin recorded and survival are independent, therefore, cabin recorded is a good predictor of survival, thus rather than replace missing cabins with the mean cabin, I will replace them with a new category "U" for unknown, I will also lump all recorded cabins into 1 category

#determine if passenger is alone
"""
alone = []
for i in range(len(train)):
    if train['SibSp'][i] == 0 and train['Parch'][i] == 0:
        alone.append(1)
    else:
        alone.append(0)
alone = np.array(alone)
train['Alone'] = alone
"""
#determine if passenger is a child and information regarding parental relationships
"""
counterC = 0
counterP = 0
allP = np.sum(train['Parch'])
child = []
for i in range(len(train)):
    if train['Age'][i] < 18:
        if train['Parch'][i] > 0:
            counterC += 1
            counterP += train['Parch'][i]
        child.append(1)
    else:
        child.append(0)
child = np.array(child)
train['Child'] = child
print("Number of under 18 year olds: " + str(np.sum(child)))
print("Number of children: " + str(counterC))
print("Number of parents of under 18 year olds: " + str(counterP))
print("Number of parents/children relationships in total: " + str(allP))
"""
#number of under 18 year olds is 113, number of under 18 year olds with parch relationship is 81, number of parch relations of under 18 year olds is 119, number of parent/children relationships in total is 340/2 = 170

#split data into training and testing 
validation = testData
np.random.seed(0)
train = train.sample(frac=1)
train = train.reset_index(drop=True)
descriptive_train = train[0:int(0.8*len(train))]
descriptive_test = train[int(0.8*len(train)):]
y = train['Survived'].values

#process train before making it into a numpy array

#check if data contains survived 
if 'Survived' in train.columns:
    train = train.drop(['Survived'], axis=1)
    y = train['Survived'].values
else:
    y = np.zeros(len(train))


train = train.drop(['Survived','PassengerId', 'Name', 'Ticket'], axis=1)
validation = validation.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
#while I am dropping names, it is not because I know they are a bad predictor of survival, but that is a guess
#Using a BERT model to represent the names likely would be useless.
#The names best use is for determining famial relationships
#The names also include prefixes but most of that information is included in age and sex 
#I am also dropping ticket number because it is not clear how to use it
#I am also dropping passenger id because I suspect it is not a good predictor of survival
#I drop survived because it is the target variable
feature_names = train.columns.values
#filling in missing age with mean age is a bad idea since whether or not age is recorded is a good predictor of survival
#replace age with categorical variable for age group using the following splits: 0-15, 15-30, 30-50, 50-100, U 
#the splits were optimized to maximize the difference in survival rate between age groups
#0-15: 0.59, 15-30: 0.36, 30-50: 0.42, 50-100: 0.34, U: 0.29 
train['Age'] = train['Age'].fillna(-1)
train['Age'] = [0 if i <= 15 else 1 if i <= 30 else 2 if i <= 50 else 3 if i <= 100 else 4 for i in train['Age']]

#placing the unkown age group at the end might seem counterintuitive but it is because the unknown age group has the lowest survival rate and 
#what I am doing is techincally a non-linear feature-transformation

#replace Cabin with binary variable for whether or not cabin is recorded
#i also tried using the first letter of the cabin as a categorical variable but it did not improve the model
train['Cabin'] = train['Cabin'].fillna('U')
train['Cabin'] = [0 if i == 'U' else 1 for i in train['Cabin']]
#add unknown category for 2 passengers with missing embarked
#train['Embarked'] = train['Embarked'].fillna('U')
#replace unkown embarked with mode
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])
train["Fare"] = train["Fare"].fillna(train["Fare"].mean())

validation['Age'] = validation['Age'].fillna(-1)    
validation['Age'] = [0 if i <= 15 else 1 if i <= 30 else 2 if i <= 50 else 3 if i <= 100 else 4 for i in validation['Age']]
validation['Cabin'] = validation['Cabin'].fillna('U')
validation['Cabin'] = [0 if i == 'U' else 1 for i in validation['Cabin']]
validation['Embarked'] = validation['Embarked'].fillna(validation['Embarked'].mode()[0])
validation["Fare"] = validation["Fare"].fillna(validation["Fare"].mean())

#train['Fare'] = train['Fare'].fillna(train['Fare'].mean())
le = preprocessing.LabelEncoder()
ohe = preprocessing.OneHotEncoder()
#train = train.dropna()
print(feature_names)

#turn train into a numpy array
x = train.values
validationX = validation.values

print(x.shape)
#print(train.head())
#encode gender, embarked, and age group as one-hot vectors and normalize fare (pclass categories make sense as is)
le.fit(x[:, 1])
x[:, 1] = le.transform(x[:, 1])
validationX[:, 1] = le.transform(validationX[:, 1])
le.fit(x[:, 7])
x[:, 7] = le.transform(x[:, 7])
validationX[:, 7] = le.transform(validationX[:, 7])


y = y[train.index.values] #reorder y to match x
x_train = x[0:int(0.8*len(x))]
y_train = y[0:int(0.8*len(y))]
x_test = x[int(0.8*len(x)):]
y_test = y[int(0.8*len(y)):]    

#remove Nan from all data
x_train = np.nan_to_num(x_train)
x_test = np.nan_to_num(x_test)
validationX = np.nan_to_num(validationX)
x = np.nan_to_num(x)

#split age at 0, 10, 20, 30, 40, 50, U and define new features
#split sex in 2
#split class in 3
#define x_trans as a matrix of 0s and 1s with n columns and one-hot encoding of feature


"""
x_trans = np.zeros((x.shape[0], 12))
for i in range(len(x)):
    if x[i, 0] == 1:
        x_trans[i, 0] = 1
    if x[i, 0] == 2:
        x_trans[i, 1] = 1
    if x[i, 0] == 3:
        x_trans[i, 2] = 1

for i in range(len(x)):
    if x[i, 1] == 0:
        x_trans[i, 3] = 1
    else:
        x_trans[i, 4] = 1

age_splits = [10, 20, 30, 40, 50, 100]
flag = True
for i in range(len(x)):
    for j in range(len(age_splits)):
        if x[i, 2] <= age_splits[j] and flag:
            x_trans[i, j + 5] = 1
            flag = False
    if flag:
        x_trans[i, 11] = 1
    flag = True

#split x_trans into 80% training and 20% validation
x_trans_train = x_trans[0:int(0.8*len(x_trans))]
x_trans_test = x_trans[int(0.8*len(x)):]
"""

def plot_categorical(x, y, title):
    unique = np.unique(x)
    odd = [1 + 2*i for i in range(len(unique))]
    even = [2 + 2*i for i in range(len(unique))]
    plt.bar(odd, [np.sum(y[x == i] == 0) for i in unique], label="0")
    plt.bar(even, [np.sum(y[x == i] == 1) for i in unique], label="1")
    label = [str(i) for i in unique]
    plt.xticks(odd, label)
    plt.title(title)
    plt.legend()
    plt.show()

def plot_continuous(x, y, title): 
    plt.scatter(x[y == 0], y[y == 0], label="0")
    plt.scatter(x[y == 1], y[y == 1], label="1")
    plt.title(title)
    plt.legend()
    plt.show()

"""
plot_categorical(x[:,0], y, "Pclass")
plot_categorical(x[:, 1], y, "Sex")
plot_categorical(x[:,2], y, "Age")
plot_categorical(x[:,3], y, "SibSp")
plot_categorical(x[:,4], y, "Parch")
plot_continuous(x[:,5], y, "Fare")
plot_categorical(x[:,6], y, "Cabin")
plot_categorical(x[:,7], y, "Embarked")
"""

#determine survival rate of wives/sisters and husbands/brothers
"""totalWives = np.sum(x[:, 1][x[:, 3] > 0] == 0)
totalSingleWomen = np.sum(x[:, 1][x[:, 3] == 0] == 0)
totalHusbands = np.sum(x[:, 1][x[:, 3] > 0] == 1)
totalSingleMen = np.sum(x[:, 1][x[:, 3] == 0] == 1)
aliveWives = np.sum(y[x[:, 3] > 0][x[:, 1][x[:, 3] > 0] == 0] == 1)
aliveSingleWomen = np.sum(y[x[:, 3] == 0][x[:, 1][x[:, 3] == 0] == 0] == 1)
aliveHusbands = np.sum(y[x[:, 3] > 0][x[:, 1][x[:, 3] > 0] == 1] == 1)
aliveSingleMen = np.sum(y[x[:, 3] == 0][x[:, 1][x[:, 3] == 0] == 1] == 1)
print("Survival rate wives/sisters: " + str(aliveWives/totalWives))
print("Survival rate of husbands/brothers: " + str(aliveHusbands/totalHusbands))
print("Survival rate of single women: " + str(aliveSingleWomen/totalSingleWomen))
print("Surival rate of single men " + str(aliveSingleMen/totalSingleMen))"""
#interestingly, single women have a higher survival rate than married woman while married men have a higher survival rate than single men
#it may be due to unseen confounding variables such as class or age
#regardless, my machine learning model does not care if a coorelation is due to a confounding variable or not, it just cares that there is a correlation

#linear model to establish a baseline and get a sense of feature importance
def linearModel(x_train, y_train, x_test, y_test):
    lr = linear_model.LogisticRegression(max_iter=1000)
    lr.fit(x_train, y_train)
    print("Logistic Regression Score: " + str(lr.score(x_test, y_test)))
    print("Logistic Regression Coefficients: " + str(lr.coef_))
    print("Logistic Regression Most Important Features: " + str(np.argsort(np.abs(lr.coef_))[0]))
    print("Logistic Regression Most Important Features: " + str(feature_names[np.argsort(np.abs(lr.coef_))[0]]))

#train a linear model for each feature
def oneFeatureLR(x_train, y_train, x_test, y_test):
    for i in range(x_train.shape[1]):
        lr = linear_model.LogisticRegression()
        lr.fit(x_train[:,i].reshape(-1,1), y_train)
        print("Logistic Regression Score (only feature " + feature_names[i] + "): " + str(lr.score(x_test[:,i].reshape(-1,1), y_test)))
        print("Logistic Regression Coefficients: " + str(lr.coef_))

def linearModelWithNormalization(x_train, y_train, x_test, y_test):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train_normal = scaler.transform(x_train)
    x_test_normal = scaler.transform(x_test)
    lr = linear_model.LogisticRegression(max_iter=1000)
    lr.fit(x_train_normal, y_train)
    print("Logistic Regression Score: " + str(lr.score(x_test_normal, y_test)))
    print("Logistic Regression Coefficients: " + str(lr.coef_))
    print("Logistic Regression Most Important Features: " + str(np.argsort(np.abs(lr.coef_))[0]))
    print("Logistic Regression Most Important Features: " + str(feature_names[np.argsort(np.abs(lr.coef_))[0]]))


def splitLinearModelByMaleTrainedFemaleAlive(x_train, y_train, x_test, y_test):
    #female model
    female_y_pred = np.ones(len(x_test[:, 1] == 0)) 
    female_y_score = np.s
    
    #male model
    lr = linear_model.LogisticRegression(max_iter=1000)
    lr.fit(x_train[x_train[:, 1] == 1], y_train[x_train[:, 1] == 1])
    male_y_pred = lr.predict(x_test[x_test[:, 1] == 1])
    male_y_score = lr.score(x_test[x_test[:, 1] == 1], y_test[x_test[:, 1] == 1])

    #combine predictions
    y_pred = []
    femIter = 0
    maleIter = 0
    for i in range(len(x_test)):
        if x_test[i, 1] == 0:
            y_pred.append(female_y_pred[femIter])
            femIter += 1
        else:
            y_pred.append(male_y_pred[maleIter])

    y_pred = np.array(y_pred)
    print("Logistic Regression Score: " + str(np.sum(y_pred == y_test) / len(y_test)))


                









    



    
            


oneFeatureLR(x_train, y_train, x_test, y_test)
linearModel(x_train, y_train, x_test, y_test)

#dummy model for gender
def dummyModelAllDead(x_test, y_test):
    y_pred = []
    for i in range(len(x_test)):
        y_pred.append(0)
    y_pred = np.array(y_pred)
    print("All Dead Dummy Model Score: " + str(np.sum(y_pred == y_test) / len(y_test)))

def dummyModelSex(x_test, y_test):
    y_pred = []
    for i in range(len(x_test)):
        if x_test[i, 1] == 0:
            y_pred.append(1)
        else:
            y_pred.append(0)
    y_pred = np.array(y_pred)
    print("Gender Dummy Model Score: " + str(np.sum(y_pred == y_test) / len(y_test)))

#dummy model for class
def dummyModel2(x_test, y_test):
    y_pred = []
    for i in range(len(x_test)):
        if x_test[i, 0] == 1:
            y_pred.append(1)
        else:
            y_pred.append(0)
    y_pred = np.array(y_pred)
    print("Class Dummy Model Score: " + str(np.sum(y_pred == y_test) / len(y_test)))

#dummy model for age under 15
def dummyModel4(x_test, y_test):
    y_pred = []
    for i in range(len(x_test)):
        if x_test[i, 2] >= .59:
            y_pred.append(1)
        else:
            y_pred.append(0)
    y_pred = np.array(y_pred)
    print("Age Dummy Model Score: " + str(np.sum(y_pred == y_test) / len(y_test)))


#dummy model that predicts survival if women, child under 10, or first class
def dummyModel3(x_test, y_test):
    y_pred = []
    for i in range(len(x_test)):
        if x_test[i, 0] == 1 or x_test[i, 1] == 0 or x_test[i, 2] >= .59:
            y_pred.append(1)
        else:
            y_pred.append(0)
    y_pred = np.array(y_pred)
    print("Combine Dummy Model Score: " + str(np.sum(y_pred == y_test) / len(y_test)))

#dummy model for women and children
def dummyModel5(x_test, y_test):
    y_pred = []
    for i in range(len(x_test)):
        if x_test[i, 1] == 0 or x_test[i, 2] >= .59:
            y_pred.append(1)
        else:
            y_pred.append(0)
    y_pred = np.array(y_pred)
    print("Women + Children Dummy Model Score: " + str(np.sum(y_pred == y_test) / len(y_test)))

#dummy model for if cabin is recorded
def dummyModel6(x_test, y_test):
    y_pred = []
    for i in range(len(x_test)):
        if x_test[i, 6] == 1:
            y_pred.append(1)
        else:
            y_pred.append(0)
    y_pred = np.array(y_pred)
    print("Cabin Dummy Model Score: " + str(np.sum(y_pred == y_test) / len(y_test)))

#dummy model women survive, men survive if they are first class and have a cabin
def dummyModel8(x_test, y_test):
    y_pred = []
    for i in range(len(x_test)):
        if x_test[i, 1] == 0:
            y_pred.append(1)
        elif x_test[i, 0] == 1 and x_test[i, 6] == 1:
            y_pred.append(1)
        else:
            y_pred.append(0)
    y_pred = np.array(y_pred)
    print("Dummy model sex, cabin, class: " + str(np.sum(y_pred == y_test) / len(y_test)))


print(x_test[:10])
print(y_test[:10])
dummyModelAllDead(x_test, y_test)
dummyModelSex(x_test, y_test)
dummyModel2(x_test, y_test)
dummyModel3(x_test, y_test)
dummyModel4(x_test, y_test)
dummyModel5(x_test, y_test)
dummyModel6(x_test, y_test)


#results are really bad, equivalent to assuming everyone died

#remove cabin and embarked
x_train = np.delete(x_train, [6,7], axis=1)
x_test = np.delete(x_test, [6,7], axis=1)

#try a decision tree
from sklearn import tree
dt = tree.DecisionTreeClassifier()
dt.fit(x_train, y_train) 
#dt.fit(x_train, y_train)
print("Decision Tree Score (no embarked or cabin): " + str(dt.score(x_test, y_test)))
print("Decision Tree Feature Importances: " + str(dt.feature_importances_))

#try a decision tree uising age, gender, and class



    




#try a random forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(x_train, y_train)
print("Random Forest Score (no embarked or cabin): " + str(rf.score(x_test, y_test)))
print("Random Forest Feature Importances: " + str(rf.feature_importances_))

#results are still bad, try a neural network
nn = MLPClassifier()
nn.fit(x_train, y_train)
y_test_pred = nn.predict(x_test)        

print("Neural Network Score (no embarked or cabin): " + str(nn.score(x_test, y_test)))



#try a neural network with 2 hidden layers
nn = MLPClassifier(hidden_layer_sizes=(8,8))
nn.fit(x_train, y_train)
print("Neural Network Score 2 (no embarked or cabin): " + str(nn.score(x_test, y_test)))

nn2 = MLPClassifier(hidden_layer_sizes=(8,8,8))
nn2.fit(x_train, y_train)
print("Neural Network Score 3 (no embarked or cabin): " + str(nn2.score(x_test, y_test)))

#perform cross validation on hyperparamters for neural network
#tune hidden layer size, activation function, solver, alpha, learning rate, learning rate init

def nnCrossValidation():
    hiddenLayerSizes = [(8,8), (8,8,8), (8,8,8,8), (8,8,8,8,8), (12, 12 ,12), (12, 12, 12, 12), (12, 12, 12, 12, 12)]
    activation = ['identity', 'logistic', 'tanh', 'relu']
    solver = ['lbfgs', 'sgd', 'adam']
    alpha = [0.0001, 0.001, 0.01, 0.1, 1]
    learningRate = ['constant', 'invscaling', 'adaptive']
    learningRateInit = [0.00001, 0.0001, 0.001, 0.01]
    bestScore = 0
    bestParams = None
    for h in hiddenLayerSizes:
        for a in activation:
            for s in solver:
                for al in alpha:
                    for l in learningRate:
                        for li in learningRateInit:
                            nn = MLPClassifier(hidden_layer_sizes=h, activation=a, solver=s, alpha=al, learning_rate=l, learning_rate_init=li)
                            nn.fit(x_train, y_train)
                            score = nn.score(x_test, y_test)
                            if score > bestScore:
                                bestScore = score
                                bestParams = (h, a, s, al, l, li)
    print("Best Score: " + str(bestScore))
    print("Best Params: " + str(bestParams))


nn = MLPClassifier(hidden_layer_sizes=(10, 10), activation='tanh', solver='adam', alpha=0.1, max_iter=2000)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x)
x_normal = scaler.transform(x)
testX, junk, junk1, junk2, junk3, junk4, test, test_feature_names = manualPreprocessing(testData)
print(testX[-1])
testX_normal = scaler.transform(testX)

#remove Nan from test data
testX_normal = np.nan_to_num(testX_normal)

nn.fit(x_normal, y)



y_pred = nn.predict(testX_normal)

"""

#engineer new features
#determine #parents/children and #siblings/spouses
#determine if passenger is alone
#determine if passenger is a child
#determine if passenger is a parent
#determine if passenger is a spouse
"""




#try a random forest with new features
#try a neural network with new features


"""
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras import regularizers

model = Sequential()
model.add(Dense(8, activation='relu', input_dim=6))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=32)
score = model.evaluate(x_test, y_test, batch_size=32)
print("Neural Network Score 2 (no embarked or cabin): " + str(score[1]))

"""


#used fixed-shape universal approximators employing the kernel trick to learn a function from the data
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

def kernelRidgeRegression(x_train, y_train, x_test, y_test):
    kr = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), cv=5,
                      param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3],
                                  "gamma": np.logspace(-2, 2, 5)})
    kr.fit(x_train, y_train)
    print("Kernel Ridge Regression Score: " + str(kr.score(x_test, y_test)))
    print("Kernel Ridge Regression Coefficients: " + str(kr.coef_))
    print("Kernel Ridge Regression Most Important Features: " + str(np.argsort(np.abs(kr.coef_))[0]))
    print("Kernel Ridge Regression Most Important Features: " + str(feature_names[np.argsort(np.abs(kr.coef_))[0]]))



def sparseMatrixProcessing(x1, x2):
    #use the following categories for fare 
    None




def manualPreprocessingOHE(d1, d2, ageGroups=[15, 30, 50, 100]):
  train = d1
  validation = d2
  np.random.seed(0)
  train = train.sample(frac=1)
  train = train.reset_index(drop=True)
  descriptive_train = train[0:int(0.8*len(train))]
  descriptive_test = train[int(0.8*len(train)):]
  #process train before making it into a numpy arrar
  y = train['Survived'].values
  train = train.drop(['Survived'], axis=1)
  train = train.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
  validation = validation.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
  feature_names = train.columns.values
  train['Age'] = train['Age'].fillna(-1)
  newAge = []
  for i in range(len(train)):
    if train['Age'][i] == -1:
      newAge.append(len(ageGroups))
    else:
      for j in range(len(ageGroups)):
        if train['Age'][i] <= ageGroups[j]:
          newAge.append(j)
          break

  train['Age'] = newAge
  #train['Age'] = [0 if i <= 15 else 1 if i <= 30 else 2 if i <= 50 else 3 if i <= 100 else 4 for i in train['Age']]
  train['Cabin'] = train['Cabin'].fillna('U')
  train['Cabin'] = [0 if i == 'U' else 1 for i in train['Cabin']]
  train['Embarked'] = train['Embarked'].fillna('U')
  train["Fare"] = train["Fare"].fillna(train["Fare"].mean())
  train['Fare'] = [0 if i <= 10 else 1 if i <= 30 else 2 if i <= 100 else 3 for i in train['Fare']]
  validation['Age'] = validation['Age'].fillna(-1)
  validation['Age'] = [0 if i <= 15 else 1 if i <= 30 else 2 if i <= 50 else 3 if i <= 100 else 4 for i in validation['Age']]
  validation['Cabin'] = validation['Cabin'].fillna('U')
  validation['Cabin'] = [0 if i == 'U' else 1 for i in validation['Cabin']]
  validation['Embarked'] = validation['Embarked'].fillna(validation['Embarked'].mode()[0])
  validation["Fare"] = validation["Fare"].fillna(validation["Fare"].mean())
  validation['Fare'] = [0 if i <= 10 else 1 if i <= 30 else 2 if i <= 100 else 3 for i in validation['Fare']]
  validationX = validation.values


  le = preprocessing.LabelEncoder()
  ohe = preprocessing.OneHotEncoder()
  x = train.values

  le.fit(x[:, 1])
  x[:, 1] = le.transform(x[:, 1])
  validationX[:, 1] = le.transform(validationX[:, 1])
  le.fit(x[:, 7])
  x[:, 7] = le.transform(x[:, 7])
  validationX[:, 7] = le.transform(validationX[:, 7])
  y = y[train.index.values] #reorder y to match x
  #replace x and validatio nX with sparse matrices using one-hot encoding
  #concatenate x and validationX to ensure that they have the same number of columns
  concatX = np.concatenate((x, validationX), axis=0)
  ohe.fit(concatX)
  x = ohe.transform(x)
  validationX = ohe.transform(validationX)
   





  x_train = x[0:int(0.8*len(x))]
  y_train = y[0:int(0.8*len(y))]
  x_test = x[int(0.8*len(x)):]
  y_test = y[int(0.8*len(y)):]





  #x = np.nan_to_num(x)
  #validationX = np.nan_to_num(validationX)

  return x, y, x_train, y_train, x_test, y_test, train, feature_names, validationX, validation

testData = pd.read_csv("test.csv")
x, y, x_train, y_train, x_test, y_test, train, feature_names, validationX, validation = manualPreprocessingOHE(train, testData)

ageGroupings = [[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 100], [10, 20, 30, 40, 50, 100], [15, 30, 45, 60, 100], [20, 40, 100], [30, 100], [15, 30, 50, 100]]
for ag in ageGroupings:
    lr = linear_model.LogisticRegression(max_iter=1000)
    lr.fit(x_train, y_train)
    print("Logistic Regression Score: " + str(lr.score(x_test, y_test)))
    print("Logistic Regression Coefficients: " + str(lr.coef_))
    print("Logistic Regression Most Important Features: " + str(np.argsort(np.abs(lr.coef_))[0]))
    print("Logistic Regression Most Important Features: " + str(feature_names[np.argsort(np.abs(lr.coef_))[0]]))

import tensorflow_decision_forests as tfdf
import tensorflow as tf

#concatenate pd_train and pd_test
concat = pd.concat([train, validation], axis=0)
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train, label="Survived")
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(validation)

#normalize the data
#train_ds = tf.keras.utils.normalize(train_ds, axis=1)
#test_ds = tf.keras.utils.normalize(test_ds, axis=1)


# Use Kernel Methods to learn a function from the data 
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.model_selection import GridSearchCV  

def gaussianProcessRegression(x_train, y_train, x_test, y_test):
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    gp.fit(x_train, y_train)
    #make predictions
    y_pred, sigma = gp.predict(x_test, return_std=True)
    threshold = 0.5
    y_pred[y_pred >= threshold] = 1
    y_pred[y_pred < threshold] = 0
    print("Gaussian Process Regression Score: " + str(np.sum(y_pred == y_test) / len(y_test)))




#SVM
from sklearn.svm import SVC 
def svm(x_train, y_train, x_test, y_test, kernel='linear'):
    svclassifier = SVC(kernel=kernel)
    svclassifier.fit(x_train, y_train)
    y_pred = svclassifier.predict(x_test)
    print("SVM Score: " + str(np.sum(y_pred == y_test) / len(y_test)))

#hyperparameter tuning for SVM
def svmHyperparameterTuning(x_train, y_train, x_test, y_test):
    from sklearn.model_selection import GridSearchCV
    parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
    svc = SVC()
    clf = GridSearchCV(svc, parameters)
    clf.fit(x_train, y_train)
    print("SVM Hyperparameter Tuning Score: " + str(clf.score(x_test, y_test)))
    print("SVM Hyperparameter Tuning Best Parameters: " + str(clf.best_params_))


#visual comparison of model performances
def plot_accuracies(x_train, y_train, x_test, y_test):
    #compare linear model, decision tree, random forest, neural network, kernel ridge regression, gaussian process regression, svm
    models = [linear_model.LogisticRegression(max_iter=1000), tree.DecisionTreeClassifier(), RandomForestClassifier(n_estimators=100), MLPClassifier(), KernelRidge(kernel='rbf', gamma=0.1), GaussianProcessRegressor(kernel=C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)), n_restarts_optimizer=9), SVC(kernel='linear')]
    modelNames = ["Logistic Regression", "Decision Tree", "Random Forest", "Neural Network", "Kernel Ridge Regression", "Gaussian Process Regression", "SVM"]
    scores = []
    for i in range(len(models)):
        models[i].fit(x_train, y_train)
        scores.append(models[i].score(x_test, y_test))
    plt.bar(modelNames, scores)
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Model")
    plt.show()
    #linear model
    #decision tree
    #random forest
    #neural network
    
#tune hyperparamters for decision tree
def decisionTreeHyperparameterTuning(x_train, y_train, x_test, y_test):
    from sklearn.model_selection import GridSearchCV
    parameters = {'criterion':('gini', 'entropy'), 'max_depth':[1, 10, 100, 1000, 10000], 'max_features':('auto', 'sqrt', 'log2')}
    dt = tree.DecisionTreeClassifier()
    clf = GridSearchCV(dt, parameters)
    clf.fit(x_train, y_train)
    print("Decision Tree Hyperparameter Tuning Score: " + str(clf.score(x_test, y_test)))
    print("Decision Tree Hyperparameter Tuning Best Parameters: " + str(clf.best_params_))