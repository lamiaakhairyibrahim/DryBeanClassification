import pandas as pd
from src.preprocessing import preprocess
from src.model import models

class collect:
    def __init__(self ,path_data):
        self.path_data = path_data
        self.coll()
    
    def coll(self):
        data = pd.read_csv(self.path_data)
        pdata = preprocess(data)
        pgdata = pdata.get_data()
        print(pgdata.info())
        x = pgdata.drop(columns=['Class'])
        y = pgdata["Class"]
        mod = models(x , y)
        best_model = mod.rand_forest()
        print(best_model)
        best_model_ = mod.svc_()
        print(best_model_)

