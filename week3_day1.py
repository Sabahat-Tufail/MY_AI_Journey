import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data ={
    'Hours_studied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Exam_score': [55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
}

df=pd.DataFrame(data)
df