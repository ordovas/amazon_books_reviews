from sklearn.base import BaseEstimator  # Base class for all estimators in scikit-learn.
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np

class BestClassifier(BaseEstimator):

    def __init__(self,X,y, models=["LogisticRegression", "LinearSVC","SGDClassifier",
                             "KNeighborsClassifier","GaussianNB","RandomForestClassifier"]):
        """
        A Custom BaseEstimator that chooses the best model and the best hyperparameters for a dataset
        """
        # Stores in self the list of the models that we are going to check
        self.models = models
        # Stores for each model (key) which parameters and which values are we coing to search
        # for the best model analysis
        self.parameters = {}
        self.parameters['LogisticRegression']={"l1_ratio":np.linspace(0,1,5),"C":np.logspace(-4,2,10)}
        self.parameters['LinearSVC']={"C":np.logspace(-4,2,10), "penalty":["l1", "l2"]}
        self.parameters['SGDClassifier']={"penalty":["elasticnet"],"alpha":np.logspace(-4,2,10),
                                          "l1_ratio":np.linspace(0,1,10)} 
        self.parameters['KNeighborsClassifier']={"n_neighbors":[3,5,10,20,50,100],
                                                 "metric":["euclidean","minkowski","manhattan","chebyshev"]}
        self.parameters['GaussianNB']={'var_smoothing': np.logspace(0,-9, num=20)}
        self.parameters['RandomForestClassifier']={'n_estimators': [5,10,25],
                                                   'max_features': ['auto', 'sqrt', 'log2'],
                                                   'min_samples_split':[2,5,10,20,30],
                                                   'max_depth' : [None,5,10,25,50,100,250,500],
                                                   'criterion' :['gini', 'entropy']}
        # This function decides what is the best model and which are the best parameters
        self.decide(X,y) 
        
        

    # This function, for each method, does a grid search to choose the best parameter among 1 method
    def grid(self,  X, y,classifier_type: str = 'LogisticRegression'):
        # Logistic regression grid search
        print("Analyzing "+classifier_type)
        if classifier_type == 'LogisticRegression':
            self.classifier_ = LogisticRegression(penalty="elasticnet", max_iter=10000,class_weight="balanced",solver="saga")
            search=GridSearchCV(self.classifier_ , self.parameters['LogisticRegression'],
                                n_jobs=-1, cv=5,verbose=0)
            search.fit(X, y)
            self.classifier_ = LogisticRegression(penalty="elasticnet",max_iter=10000,solver="saga",
                                                  class_weight="balanced",**search.best_params_)

            
            
        # Linear SVC grid search    
        elif classifier_type == 'LinearSVC':
            self.classifier_ = LinearSVC(max_iter=10000,class_weight="balanced")
            search=GridSearchCV(self.classifier_ , self.parameters['LinearSVC'], n_jobs=-1, cv=5,verbose=0)
            search.fit(X, y)
            self.classifier_ = LinearSVC(max_iter=10000,class_weight="balanced",**search.best_params_)

            
        # SGD Classifier grid search    
        elif classifier_type == 'SGDClassifier':
            self.classifier_ = SGDClassifier(max_iter=10000,class_weight="balanced")
            search=GridSearchCV(self.classifier_ , self.parameters['SGDClassifier'], n_jobs=-1, cv=5,verbose=0)
            search.fit(X, y)
            self.classifier_ = SGDClassifier(max_iter=10000,class_weight="balanced",**search.best_params_)

            
        # K-nearest neighbors classifier grid search    
        elif classifier_type == 'KNeighborsClassifier':
            self.classifier_ = KNeighborsClassifier()
            search=GridSearchCV(self.classifier_ , self.parameters['KNeighborsClassifier'], n_jobs=-1, cv=5,verbose=0)
            search.fit(X, y)
            self.classifier_ = KNeighborsClassifier(**search.best_params_)

            
        # Naive Bayes classifier grid search    
        elif classifier_type == 'GaussianNB':
            self.classifier_ = GaussianNB()
            search=GridSearchCV(self.classifier_ , self.parameters['GaussianNB'], n_jobs=-1, cv=5,verbose=0)
            search.fit(X, y)
            self.classifier_ = GaussianNB(**search.best_params_)

            
        # Random Forest classifier grid search    
        elif classifier_type == 'RandomForestClassifier':
            self.classifier_ = RandomForestClassifier(class_weight="balanced")
            search=GridSearchCV(self.classifier_ , self.parameters['RandomForestClassifier'], n_jobs=-1, cv=5,verbose=0)
            search.fit(X, y)
            self.classifier_ = RandomForestClassifier(class_weight="balanced",**search.best_params_)
            
        # Error mensage if the method is not listed    
        else:
            raise ValueError('Unkown classifier type.')

        print(f"Score = {search.best_score_}")
        # Returns the best classifier function and its validation score
        return self.classifier_.fit(X, y),search.best_score_
    
    
    
    # This function search for each method its best fitting parameter to choose the best model
    def decide(self,X,y):
        # It begins with score=0
        score=0
        # Loop to obtain the best model for each method
        for model in self.models:
            clf,score_t=self.grid(X, y,model)
            # If it is the best score until now, we store the 
            # best model as the best classifier, and store its validation score
            if score < score_t:
                score=score_t
                self.bestclassifier_=clf
        
        print("Best model:")
        print(clf)
        
        
    # When we've got the best classifier, these functions common to all classifiers can be used
    # for our ML exercises.
    def fit(self, X, y=None):
        return self.bestclassifier_.fit(X, y)
    
    def predict(self, X, y=None):
        return self.bestclassifier_.predict(X)
    
    def predict_proba(self, X):
        return self.bestclassifier_.predict_proba(X)

    def score(self, X, y):
        return self.bestclassifier_.score(X, y)