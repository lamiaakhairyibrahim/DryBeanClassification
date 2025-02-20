from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split  , GridSearchCV 
from sklearn.metrics import confusion_matrix , classification_report
import xgboost


class models:
    def __init__(self  , x , y):
        self.x =x
        self.y = y
        self.split_data()
    
    def split_data(self ):
        self.x_train , self.x_test , self.y_train , self.y_test =  train_test_split(self.x , self.y , test_size= 0.2 , random_state= 0)

    def rand_forest(self ):
        par_grad = {"n_estimators": [50 , 100, 150 , 200] , 'criterion' :['gini', 'entropy'] , 'max_depth' : [5,10,15,20 , 25]}
        model = RandomForestClassifier(class_weight='balanced')
        grid_search = GridSearchCV(estimator=model , param_grid= par_grad , cv= 5 , n_jobs= -1 , scoring= 'f1_macro' )
        grid_search.fit(self.x_train , self.y_train)
        self.best_model = grid_search.best_estimator_
        self.test_pred()
        return self.best_model 
    
    
    def svc_(self ):
        svr_parameters = {'kernel':['linear', 'poly', 'rbf', 'sigmoid'],'C':[0.01,0.1,1,10,100]}
        # grid search pitch
        model_ = SVC(class_weight = 'balanced' )
        grid_search = GridSearchCV(estimator=model_ , param_grid= svr_parameters  , cv= 5 , n_jobs= -1 , scoring= 'f1_macro' )
        grid_search.fit(self.x_train , self.y_train)
        self.best_model = grid_search.best_estimator_
        self.test_pred()
        return self.best_model 
    
    
    def test_pred(self):
        self.y_pred = self.best_model.predict(self.x_test)
        conf_matr = confusion_matrix(self.y_test , self.y_pred)
        print(conf_matr)
        conf_matr_report = classification_report(self.y_test ,self.y_pred)
        print(conf_matr_report)



