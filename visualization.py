import matplotlib.pyplot as plt 
import seaborn as sns

class visual:
    def __init__(self):
        pass

    def hest(self ,data):
        plt.hist(data , bins= 30 , color= "green")
        plt.show()

    def heatmap(self , data):
        corr = data.corr()
        sns.heatmap(corr)
        plt.show()
        
    def box_plot(self , data):
        sns.boxplot(data )
        plt.xticks(rotation = 90)
        plt.show()


        