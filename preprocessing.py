import pandas as pd
from sklearn.preprocessing import LabelEncoder , StandardScaler
from sklearn.decomposition import PCA
from src.visualization import visual
class preprocess:
        def __init__(self ,data):
            self.data = data
            self.process()
        def process(self):
            print(f"sample from the data:\n{self.data.head()}")
            print(f"dimensions of the data =  {self.data.shape}")
            print(f"information about the data:")
            print(self.data.info())
            print(f"names of types of beans : {self.data["Class"].unique()}")
            print(f"number of missing value in the data:  \n {self.data.isnull().sum()}")
            print(f"the number of duplicated samples in the dataset: {self.data.duplicated().sum()}")
            self.data = self.data.drop_duplicates(ignore_index= True)  # = data.drop_duplicates().reset_index(drop=True)
            print(f"the number of duplicated samples in the dataset after remove duplicated data : {self.data.duplicated().sum()}")
            print(f"dimensions of the data after editing: {self.data.shape}")
            vdata = visual()
            vdata.hest(self.data["Class"])
            print("Convert the label from category TO numerical by label encoding")
            labopj = LabelEncoder()
            self.data["Class"] = labopj.fit_transform(self.data["Class"])
            vdata.heatmap(self.data)
            stand_sc = StandardScaler()
            x = self.data.drop(["Class"], axis = 1)
            vdata.box_plot(x)
            y = self.data["Class"]
            x_stand_sc = stand_sc.fit_transform(x)
            pca_opj = PCA()
            pca_opj.fit(x_stand_sc)
            explain_variance = pca_opj.explained_variance_ratio_
            cumulative_variance = explain_variance.cumsum()
            print(f"Cumulative percentage of variance :{cumulative_variance}")
            n_com = (cumulative_variance >= 0.95).argmax() + 1
            print(f"number of component for PCA is : {n_com}")
            pca_opj_x = PCA(n_components= n_com)
            pca_opj__x = pca_opj_x.fit_transform(x_stand_sc)
            pca_df = pd.DataFrame(pca_opj__x , columns= [f'PC{i+1}' for i in range(n_com)])
            self.data = pd.concat([pca_df , y] , axis = 1)

        def get_data(self):
               return self.data
     
          
    

