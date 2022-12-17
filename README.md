# CoraDataset
Using page-rank like link statistics for prediction

First, the idea behind the algorithm is explained, later on the code


Idea of Algorithm is Based on the two papers:
    - [Qing Lu, and Lise Getoor. "Link-based classification." ICML, 2003.]    (https://linqspub.soe.ucsc.edu/basilic/web/Publications/2003/lu:icml03/)
    - [Prithviraj Sen, et al. "Collective classification in network data." AI Magazine, 2008.](https://linqspub.soe.ucsc.edu/basilic/web/Publications/2008/sen:aimag08/)
Main Idea in general: Make an initial prediction on the keywords-data and then use a second classifier to use the links in between papers iteratively, updating each node in every step according to the current labels of other papers

Main Idea from first Paper: Find secondary features and train a new classifier for each iteration in ICA using new features created from the links (especially count-link)

Main Idea from second paper: During ICA, update the nodes only based on a classifier trained and evaluated on "local" connections of this node
    
My main "own" Contribution: Instead of using count-link from the first paper (counting adjacent nodes of the same label and hence creating 7 new features (as there are 7 possible different labels), I use "Normalized Count-link" and only count outgoing links of a node (papers, which are cited by the paper of the node, coming from the idea "The topic of the paper is probably what the paper is based on and not, in which topic the paper is used"), and instead of counting weight "1" for each paper, I count like in page-rank "1/papers which cite this paper".
    
The setup: There is a 10-crossfold validation, where 9 sets are used to train an algorithm which predicts the labels of the tenth set. The scoring is measured by subset accuracy.

The Goal of the project: to predict new labels not only based on their features but also on their links in between each other. The goal is more to show a setup of how to find a good Algorithm for predictions than making the most accurate prediction, using ideas from the two papers named above and to show some own creativity by this.

How to predict the (approx) correct label:
First, we train a Machine Learning algorithm on the training data and evaluate a label for each element of the test set. 

For each algorithm (not currently implemented though, currently there are two loops over the same folds, one for finding the best algorithm, one for the method described below, but this leads to overfitting), we want to update each predicted label by observing the nodes connected to this element. 

We herefore calculate for each node N (the element whose label is going to be predicted) calculate "Link Statistics - a 'normalized' pagerank-like version of -count-link" from the Paper above (explained in the next paragraph) for each adjacent node, where N points to (hence the paper N cites the paper of the adjacent node), called node of first degree and each node adjacent to an adjacent node N_a, which the adjacent node N_A points to (hence, N_a cites them) (called nodes of 2nd degree). Then we use the link-statics of the nodes of second degree to train a "local" algorithm how to predict the label of their adjacent node (one node of first degree). Then we use this trained algorithm to predict from the Link statistics of the nodes of first degree the label of the wanted node.

'normalized' pagerank-like countlink: For one node 'N', we compute 7 new features: For each possible label (7 in total). We then look at each adjacent node A_i, which have a link to 'N' (Hence, N cites them), and calculate "1 divided by 'The number of nodes, which have a link from this node A_i' ". Then we sum over all this calculated numbers for all adjacent nodes of N with a given label L. This is the feauture of N to the label L and we do this for all possible labels.

Using the local predictions, we update the labels iteratively for each node in the test set, for a maximum of iterations or until no label is changed any longer. 

As Score, we take the average accuracy score of every fold and we save the predictions on each fold to a DF, which can be exported to a TSV.


Second part: Implementation:
    Plese run the whole code at first, then only the line
    "print(find_algorithm("Decision Tree", func_list))" at the end
    Global Classifier: Is chosen from the dict "classifiers" in line 52
    Local Classifer: Is here only "Decision Tree" as called in the function at the end (can be used for more different local classifiers with only minor modifications of the code, but I am running out of time )
    How to use the local Classifier: given with func_list from line 240, currently: 
      1. "normalized count-link" as explained above
      2. Identity: Do not update at all in order to get a comparison of using normalized count-link
    the function "df_to_tsv" in line 37 stores the best predictions (the prediction of a model trained on 9 Folds and predicts the 10th fold, iterated over all folds) as a TSV (CSV with sep="\t")
    
    
     
     
     
What is left to do:

For the first part of the algorithm, there is no big tuning process and no feature engineering, like finding relevant keywords for each label and only looking at these, looking for double columns, etc. 

Also, for the local predictions, more algorithms for comparison reasons can be implemented.
