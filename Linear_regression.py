# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 15:46:44 2021

@author: ZYH
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression,Lasso,Ridge,LassoCV,RidgeCV
from sklearn.model_selection import train_test_split

from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt  
import joblib

'''
使用LinearRegression、Lasso、LassoCV、Riage，RidgeCV五种来测试并保存最优解
输出最优确定系数值
'''
def fun1(brand,data1,result):
    if len(data1)<10:
        return
    LR=LinearRegression()
    X_train,X_test,Y_train,Y_test=train_test_split(data1.iloc[:,[1,3,4,5,6,7,8]],data1.iloc[:,2],train_size=.8)
    
    lasso=Lasso(alpha=0.001)
    lasso.fit(X_train,Y_train)
    lasso_score=lasso.score(X_test, Y_test)
    
    lassocv=LassoCV()
    lassocv.fit(X_train,Y_train)
    lassocv_score=lassocv.score(X_test, Y_test)
    
    ridge=Ridge(alpha=0.001)
    ridge.fit(X_train,Y_train)
    ridge_score=ridge.score(X_test, Y_test)

    ridgecv=RidgeCV()
    ridgecv.fit(X_train,Y_train)
    ridgecv_score=ridgecv.score(X_test, Y_test)
   
    LR.fit(X_train,Y_train)
    a=LR.intercept_
    b=LR.coef_
    score=LR.score(X_test, Y_test)
    
    result_max=max(lasso_score,lassocv_score,ridge_score,ridgecv_score,score)
    
    f=open("./result/"+brand+data1["model"].unique()[0]+".pkl",'w')
    if result_max==lasso_score:
        joblib.dump(lasso, "./result/"+brand+data1["model"].unique()[0]+".pkl")
    if result_max==lassocv_score:
        joblib.dump(lassocv, "./result/"+brand+data1["model"].unique()[0]+".pkl")
    if result_max==ridge_score:
        joblib.dump(ridge, "./result/"+brand+data1["model"].unique()[0]+".pkl")
    if result_max==ridgecv_score:
        joblib.dump(ridgecv, "./result/"+brand+data1["model"].unique()[0]+".pkl")
    if result_max==score:
        joblib.dump(LR, "./result/"+brand+data1["model"].unique()[0]+".pkl")
    print(brand,data1["model"].unique()[0],"R2:",result_max)    
    
filenames=os.listdir()

"""
读取数据
地址为相对路径
读取数据以及文件名
"""
data=[]
name=[]
co_result=[]
i=0
for i in filenames:
    if i.endswith("csv"):
        file=open(i,'r')
        content=pd.read_csv(file)
        for j in range(0,len(content)):
            if content.iloc[j,3]=="Manual":
                content.iloc[j,3]=1
            elif content.iloc[j,3]=="Automatic":
                content.iloc[j,3]=2
            else:
                content.iloc[j,3]=3
        for j in range(0,len(content)):
            if content.iloc[j,5]=="Petrol":
                content.iloc[j,5]=1
            elif content.iloc[j,5]=="Diesel":
                content.iloc[j,5]=2
            else:
                content.iloc[j,5]=3       
            
        data.append(content)
        name.append(i[0:-4])
        
if not os.path.exists("./result"):
    os.mkdir("./result")

result=locals()
#按品牌、按型号进行循环
for i in range(0,len(name)):
    classinformation=data[i]["model"].unique()
    for temp_class in classinformation:
        temp_data=data[i][data[i]['model'].isin([temp_class])]
        result["data"+temp_class.strip()]=temp_data  
        fun1(name[i],result["data"+temp_class.strip()],co_result)
   





