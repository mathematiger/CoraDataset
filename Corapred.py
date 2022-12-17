import copy
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score  # accuracy score for multi-classification is subset accuracy
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline  # for building pipelines to e.g. preprocess data
from statistics import mean
# Import different Classifiers
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
#Import Graph-package
import networkx as nx

#timing
import time


# Read Data from cora.content File and create Dataframe from it
df = pd.read_csv(r'/Users/dominikkohler/Downloads/cora/cora.content', sep="\t", header=None)
dfX = df.copy()
dfX = dfX.drop(dfX.columns[-1], axis=1)
dfy = df.iloc[:, -1:]
list_of_labels = df.iloc[:,-1].unique()


#Create Graph Data:
dfgraph = pd.read_csv(r'/Users/dominikkohler/Downloads/cora/cora.cites', sep="\t", header=None)
dfgraph.columns=["passive","active"]
G = nx.from_pandas_edgelist(dfgraph,source = "active",target = "passive", create_using=nx.DiGraph()) #to build graph and use graph framework in python

#Export to TSV
def df_to_tsv(dfr):
    dfr.to_csv('Results.tsv', sep="\t")


#Kfold CV kommt in 2 Funktionen vor:
splitnumber = 10
kf = KFold(n_splits=splitnumber, shuffle=True, random_state=2)
rowlengthofdf = df.shape[0]
index_X = np.arange(0, rowlengthofdf)


#TODO: Feature Enginering: look for "subset Columns" (all 1s of one Column is already in one other column) ; Extract most relevant features, maybe PCA

#dictionary for possible models for 1st stage prediction
C = 10
classifiers = {
    # "L2 logistic (Multinomial)": LogisticRegression(C=C, penalty="l2", solver="saga", multi_class="multinomial", max_iter=100, n_jobs=-1), #bad results..
    "Gauß SVC": make_pipeline(preprocessing.StandardScaler(), SVC(C=C)),
    "Decision Tree": DecisionTreeClassifier(max_depth=20),
    "XGBoost": GradientBoostingClassifier(n_estimators=23, learning_rate=1.0, max_depth=6, random_state=0),
    "XGBoost2": GradientBoostingClassifier(n_estimators=50, learning_rate=0.5, max_depth=6, random_state=0),
    #   "KNN": KNeighborsClassifier(n_neighbors=5),
}
#str_func defines local classifier, func_local_list defines the list of possible local update functions are called e.g. not updating at all, updating with normalized countlink, ....

def find_algorithm(str_func, func_local_list):
    #score results:
    list_results = [pd.DataFrame(index=range(len(dfX.index)), columns=range(len(func_local_list)))] * len(
        classifiers)  # list of all Predictions, one Dataframe for each classifier for the first (global) prediction, each DF stores Data for each local Classifier
    score_bestpred = pd.DataFrame(index=range(len(dfX.index)),
                                  columns=['Predictions'])  # list of best prediction in total
    #use classifier list from above
    n_classifiers = len(classifiers)
    #Store scores of classifiers:
    scorestore =np.arange(n_classifiers*len(func_local_list)*splitnumber).reshape(n_classifiers, len(func_local_list), splitnumber).astype(dtype = np.double)
    # loop for Crossvalidation
    for i, (train_index, test_index) in enumerate(kf.split(index_X)):
        X_train, X_test = dfX.iloc[train_index], dfX.iloc[test_index]
        y_train, y_test = dfy.iloc[train_index].values.ravel(), dfy.iloc[test_index].values.ravel()
        # loop for fitting and scoring Classifiers
        for j, (name, clf) in enumerate(classifiers.items()):
            start = time.time()
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
            score = accuracy_score(y_test, pred)
            #scorestore[j][i] = round(score, 3)
            end = time.time()
            print(name, "Score: ",score, "Time: ", end-start) #print time needed
    #maxscore = 0
    #maxscorealg = "not chosen yet"
    #compare all scores from all models and return algorithm with best score
    #print("Average Score on all Folds:")
    #for i, (name, clf) in enumerate(classifiers.items()):
    #    print(name, "Score:",mean(scorestore[i]))
    #    if(mean(scorestore[i]) > maxscore):
    #        maxscorealg = clf


    #for each algorithm from above, we update the labels locally
            # insert pred in dfinfo
            dfinfo.drop('predilabel', axis=1)
            dfinfo['predilabel'] = copy.deepcopy(df.iloc[:, -1:])
            for ihelp, jhelp in enumerate(test_index):
                dfinfo.at[jhelp, 'predilabel'] = pred[ihelp]


            for indfunc, func_label_compute in enumerate(func_local_list):
                updated = True
                maxsteps = 2  #TODO: replace with 20
                currentsteps = 0
                while(updated): #loop, until nothing can be made better; Maybe not the fastest solution..
                    print("update cycle of new nodes for algo: ", func_label_compute.__name__, currentsteps)
                    currentsteps=currentsteps+1
                    updated = False
                    for index in test_index:
                        helplabel = func_label_compute(index, classifiers_local[str_func])
                        if(dfinfo.at[index,'predilabel'] != helplabel):
                            dfinfo.at[index,'predilabel'] = helplabel
                            updated = True
                    if(currentsteps == maxsteps):
                        updated=False

                for itwo, jtwo in enumerate(test_index):    #Maybe here the labels are not in the correct order?
                    pred[itwo] = dfinfo.at[jtwo, 'predilabel']
                score = accuracy_score(y_test, pred)
                scorestore[j][indfunc][i]=round(score, 3)
                #print(score, scorestore[j][indfunc][i])
            # insert end-prediction label in Dataframe to export it to TSV
                for ihelp, jhelp in enumerate(test_index):
                    list_results[j].at[jhelp, indfunc] = pred[ihelp]
    maxscore = 0
    maxscorealg = 0
    maxscorealg_global = 0
    # compare all scores from all models and return algorithm with best score
    print("Average Score on all Folds:")
    for j, (name, clf) in enumerate(classifiers.items()):
        for ithree, nameloc in enumerate(func_local_list):
            print(nameloc.__name__,name, "Score:", mean(scorestore[j][ithree]))
            if (mean(scorestore[j][ithree]) > maxscore):
                maxscorealg = ithree
                maxscorealg_global = j
                maxscore=mean(scorestore[j][ithree])
    score_bestpred = list_results[j][maxscorealg]

    print("Number of NaNs in Predictions:", score_bestpred.isnull().sum())
    df_to_tsv(score_bestpred) #export to TSV
    return(maxscore)



#create df with: Node name - label - number of links to this node - list with nodes this node links to (or at least number of links from this node)
def create_dfinfo():
    dfinfo = copy.deepcopy(df.iloc[:,[0]])  #deepcopy, st. the code can be run twice without having variables initialized as left-overs
    dfinfo.columns=["Paper"] #store papernames in first column
    dfinfo['predilabel'] = copy.deepcopy(df.iloc[:, -1:]) #ddepcopy, if labels are changed
    return dfinfo
#function to calculate prdilabel and numbreceivinglinks
dfinfo = create_dfinfo()
#print(dfinfo)
#print(G.in_edges(35))


#functions for adjusting node labels with graph data

#forloop: use the 10-fold cv from before and in each fold, learn the labels by the algorithm chosen above and then calculate the most probable class by a new method



def compute_countlinkold(lab, index):
    print("compute_coutnlinkold aufgerufen")
    #find all papers, which are cited by this paper and have the label lab
    name = dfinfo.iloc[index, 0]
    count = 0
    for citing in G.in_edges(name):
        helpindex = df.index[df[0]==citing[0]].tolist()
        labofciting = dfinfo.iloc[helpindex[0], 1]
        #print(labofciting)
        if(labofciting == lab):
            count=count+1
    return count


def compute_df_countlink_onenode(adjnode):
    #countlinklist = [0]*len(list_of_labels)
    #create df with columns = list_of_labels
    dfcountlink = pd.DataFrame(columns=list_of_labels)
    dfcountlink.loc[0,list_of_labels[1]] = 1 # create new row
    dfcountlink.loc[0,:] = 0 #insert initial values as 0
    for citing in G.out_edges(adjnode):
        helpindex = df.index[df[0] == citing[0]].tolist()
        labofciting = dfinfo.iloc[helpindex[0], 1]
        dfcountlink.loc[0,labofciting] = dfcountlink.loc[0,labofciting]+1/len(list(G.out_edges(citing[0])))
    return dfcountlink


def return_nodelabel_adjnode(adjnode):
    helpindex = df.index[df[0] == adjnode[0]].tolist()
    labofciting = dfinfo.iloc[helpindex[0], 1]
    return labofciting


classifiers_local = {
     #   "Gauß SVC": make_pipeline(preprocessing.StandardScaler(), SVC(C=1)),
    "Linear SVC" : make_pipeline(preprocessing.StandardScaler(), LinearSVC()),
    "Decision Tree": DecisionTreeClassifier(max_depth=5),
"XGBoost": GradientBoostingClassifier(learning_rate=0.5, max_depth=5, random_state=0),
    #   "KNN": KNeighborsClassifier(n_neighbors=5),
}
def compute_label(index, clf):
    name = dfinfo.iloc[index, 0]
    #build new Dataframe for predicting
    dfcountlinknode = pd.DataFrame(columns=list_of_labels)
    df_labels_local = pd.DataFrame(columns = ["labels"])
    if(len(G.in_edges(name))!=0): #check if there exists incoming link
        for citing in G.in_edges(name): #Paper, which are cited by Paper name
            dfcountlinknode = pd.concat([dfcountlinknode, compute_df_countlink_onenode(citing[0])])
            df_labels_local.loc[len(df_labels_local.index)]=return_nodelabel_adjnode(citing)
            # Countlink for current node:
            dftest_local = compute_df_countlink_onenode(name)
            # Use Algorithm on this Set:
        clf.fit(dfcountlinknode, df_labels_local.values.ravel())
        predlabel = clf.predict(dftest_local).tolist()[0]

    else:
        predlabel = dfinfo.loc[index, 'predilabel']
    return predlabel
#print(compute_label(100, classifiers_local["L2 logistic (Multinomial)"])) #for testing


def compute_normalized_countlinkold(lab,index):
    name = dfinfo.iloc[index, 0]
    count = 0
    for citing in G.in_edges(name):
        helpindex = df.index[df[0] == citing[0]].tolist()
        labofciting = dfinfo.iloc[helpindex[0], 1]
        if (labofciting == lab):
            denom = len(list(G.out_edges(citing[0]))) #normalize: take not 1 for each paper, but 1 over "how often this paper was cited"
            count = count + 1/denom
    return count

def give_identity(index,clf):
    return dfinfo.at[index,'predilabel']

func_list= [give_identity, compute_label]

#Values of below function:
#   str_func is for the local Classifier,
#   clf is the Classifier for predicting the first labels only based on the relational data
#   func_label_compute gives a list of functions how to compute a new label of the given node


scorepredlabel = pd.DataFrame(index=range(len(dfX.index)), columns=range(len(func_list))) #two DFs to store Results in order to export to TSV
score_bestpred =  pd.DataFrame(index=range(len(dfX.index)), columns=['Predictions'])
def localpredictions(str_func, clf, func_local_list): #creating of folds and train/test sets could have been done in an own formula
    scorestorelocal = np.arange(len(func_local_list) * splitnumber).reshape(len(func_local_list), splitnumber).astype(dtype=np.double)
    #
    for i, (train_index, test_index) in enumerate(kf.split(index_X)):
        print("Beginning fold" i)

        dfinfo.drop('predilabel', axis=1)
        dfinfo['predilabel'] = copy.deepcopy(df.iloc[:, -1:])
        X_train, X_test = dfX.iloc[train_index], dfX.iloc[test_index]
        y_train, y_test = dfy.iloc[train_index].values.ravel(), dfy.iloc[test_index].values.ravel()
        dfinfotrain = dfinfo.iloc[train_index]
        dfinfotest = dfinfo.iloc[test_index]
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)

        #insert pred in dfinfo
        for ihelp, jhelp in enumerate(test_index):
            dfinfo.at[jhelp, 'predilabel'] = pred[ihelp]


        #loop over different functions how to replace one node

        for indfunc, func_label_compute in enumerate(func_local_list):
            updated = True
            maxsteps = 2  #TODO: replace with 20
            currentsteps = 0
            while(updated): #loop, until nothing can be made better; Maybe not the fastest solution..
                print("update cycle of new nodes for algo: ", func_label_compute.__name__, currentsteps)
                currentsteps=currentsteps+1
                updated = False
                for index in test_index:
                    helplabel = func_label_compute(index, classifiers_local[str_func])
                    if(dfinfo.at[index,'predilabel'] != helplabel):
                        dfinfo.at[index,'predilabel'] = helplabel
                        updated = True
                if(currentsteps == maxsteps):
                    updated=False

            for itwo, jtwo in enumerate(test_index):    #Maybe here the labels are not in the correct order?
                pred[itwo] = dfinfo.at[jtwo, 'predilabel']
            score = accuracy_score(y_test, pred)
            scorestorelocal[indfunc][i]=round(score, 3)
        # insert end-prediction label in Dataframe to export it to TSV
            for ihelp, jhelp in enumerate(test_index):
                scorepredlabel.at[jhelp, indfunc] = pred[ihelp]

    maxscore = 0
    maxscorealg = 0
    # compare all scores from all models and return algorithm with best score
    print("Average Score on all Folds:")
    for ithree, name in enumerate(func_local_list):
        print(name.__name__, "Score:", mean(scorestorelocal[ithree]))
        if (mean(scorestorelocal[ithree]) > maxscore):
            maxscorealg = ithree
            maxscore=mean(scorestorelocal[ithree])
    score_bestpred = scorepredlabel[maxscorealg]
    print(score_bestpred)
    return(maxscore)

#faster to compute:
#print("Total Score as endresult:",localpredictions("Decision Tree", classifiers["Decision Tree"]))

print("Best Score:",find_algorithm("Decision Tree", func_list))

#print("Best average Score as endresult:",localpredictions("Decision Tree", find_algorithm(), func_list))