import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from Fisher import FLD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error #均方误差
from sklearn.metrics import mean_absolute_error #平方绝对误差
from sklearn.metrics import r2_score#R square
#调用



tmp = pd.read_csv('./embryonic_data_10genes.txt', sep='\t', index_col=0)

## 读取数据
E3 = (tmp[[x for x in tmp.columns.tolist() if x.find('E3') != -1]])
E5 = (tmp[[x for x in tmp.columns.tolist() if x.find('E5') != -1]])
E4 = (tmp[[x for x in tmp.columns.tolist() if x.find('E4') != -1]])
E6 = (tmp[[x for x in tmp.columns.tolist() if x.find('E6') != -1]])
E7 = (tmp[[x for x in tmp.columns.tolist() if x.find('E7') != -1]])

# print(E3,E5)
## 获得X与Y
train_E3 = E3.values.T;
train_E4 = E4.values.T;
train_E5 = E5.values.T;
train_E6 = E6.values.T;
train_E7 = E7.values.T;
value_E3 = np.ones(train_E3.shape[0])*3
value_E4 = np.ones(train_E4.shape[0])*4
value_E5 = np.ones(train_E5.shape[0])*5
value_E6 = np.ones(train_E6.shape[0])*6
value_E7 = np.ones(train_E7.shape[0])*7

## label
label_E3 = np.zeros(train_E3.shape[0])
label_E5 = np.ones(train_E5.shape[0])
# print(train_E3)
## 打乱样本顺序

train_E3E5_ori = np.concatenate((train_E3,train_E5),axis=0)
label_E3E5_ori = np.concatenate((label_E3,label_E5),axis=0)
train_E3ToE7 = np.concatenate((train_E3,train_E4,train_E5,train_E6,train_E7),axis = 0)
value_E3ToE7 = np.concatenate((value_E3,value_E4,value_E5,value_E6,value_E7),axis = 0)


shuffle = np.column_stack((train_E3E5_ori,label_E3E5_ori))
np.random.shuffle(shuffle)
# print(shuffle)
train_E3E5 = shuffle[:,0:10]
label_E3E5 = shuffle[:,10]
shuffle2 = np.column_stack((train_E3ToE7,value_E3ToE7))
np.random.shuffle(shuffle2)
train_E3ToE7 = shuffle2[:,0:10]
value_E3ToE7 = shuffle2[:,10]

# print (train_E3E5,label_E3E5)
## logestic回归
LR1 = LogisticRegression(C=1.0,penalty='l2',solver='liblinear',max_iter=1000)
# print(train_E3E5)
[rows, cols] = train_E3E5.shape
for i in range(rows):
    for j in range(cols):
        # if train_E3E5[i, j] != 0:
            train_E3E5[i, j] = np.log(train_E3E5[i, j]+1)
print(train_E3E5)
LR1.fit(train_E3E5,label_E3E5)
predict4 = LR1.predict(train_E4)
predict6 = LR1.predict(train_E6)
predict7 = LR1.predict(train_E7)
print('E4LR预测分类比：'+str(np.size([var0 for var0 in predict4 if var0 == 0])/np.size([var1 for var1 in predict4 if var1 == 1])))
print('E6LR预测分类比：'+str(np.size([var0 for var0 in predict6 if var0 == 0])/np.size([var1 for var1 in predict6 if var1 == 1])))
print('E7LR预测分类比：'+str(np.size([var0 for var0 in predict7 if var0 == 0])/np.size([var1 for var1 in predict7 if var1 == 1])))
scoresLR = cross_validate(LR1, train_E3E5, label_E3E5, cv=10,scoring='f1')
print('LR socre：'+str(scoresLR['test_score']))
print('mean:'+str(np.mean(scoresLR['test_score'])))

plt.title('LR Score')
plt.ylim(0, 1.5)
plt.bar(range(len(scoresLR['test_score'])), scoresLR['test_score'])
for i in range(len(scoresLR['test_score'])):
    plt.text(i, scoresLR['test_score'][i]+0.05, '%.4f' % scoresLR['test_score'][i], ha='center', va= 'bottom',fontsize=7)
plt.show()
print("-"*50)
print("自己实现的FLD并预测E3、E5前10个的结果")
##自己实现的FLD方法并测试
fld = FLD(train_E3,train_E5)
w = fld.fit()  # 调用函数，得到参数w
for i in range(10):
    out1 = fld.judge(train_E5[i])  # 判断所属的类别
    out2 = fld.judge(train_E3[i])  # 判断所属的类别
    print(out1,out2)


print("-"*50)
##sklearn中的线性判别函数
LDA = LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
              solver='svd', store_covariance=False, tol=0.0001)
LDA.fit(train_E3E5, label_E3E5)
scoresLDA = cross_validate(LDA, train_E3E5, label_E3E5, cv=10,scoring='f1')
print('FLD socre：'+str(scoresLDA['test_score']))
print('mean:'+str(np.mean(scoresLDA['test_score'])))

predictlda4 = LDA.predict(train_E4)
predictlda6 = LDA.predict(train_E6)
predictlda7 = LDA.predict(train_E7)
print('E4LDA预测分类比：'+str(np.size([var0 for var0 in predictlda4 if var0 == 0])/np.size([var1 for var1 in predictlda4 if var1 == 1])))
print('E6LDA预测分类比：'+str(np.size([var0 for var0 in predictlda6 if var0 == 0])/np.size([var1 for var1 in predictlda6 if var1 == 1])))
print('E7LDA预测分类比：'+str(np.size([var0 for var0 in predictlda7 if var0 == 0])/np.size([var1 for var1 in predictlda7 if var1 == 1])))






print("-"*50)
plt.title('FLD Score')
plt.ylim(0, 1.5)
plt.bar(range(len(scoresLDA['test_score'])), scoresLDA['test_score'])
for i in range(len(scoresLDA['test_score'])):
    plt.text(i, scoresLDA['test_score'][i]+0.05, '%.4f' % scoresLDA['test_score'][i], ha='center', va= 'bottom',fontsize=7)
plt.show()



print("-"*50)


linreg = LinearRegression(normalize=True)
[rows, cols] = train_E3ToE7.shape
for i in range(rows):
    for j in range(cols):
        # if train_E3ToE7[i, j] != 0:
            train_E3ToE7[i, j] = np.log(train_E3ToE7[i, j]+1)





linreg.fit((train_E3ToE7), (value_E3ToE7))
print('线性回归：参数'+str(linreg.coef_) , '\n截距'+str(linreg.intercept_))

plt.xlabel('num')
plt.ylabel('lable')

plt.plot(np.arange(100), linreg.predict(train_E3ToE7[1:101]),'r', label='broadcast')
plt.plot(np.arange(100), value_E3ToE7[1:101],'b', label='broadcast')
plt.show()
print('R2分数'+str(linreg.score(train_E3ToE7, value_E3ToE7)))
print("-"*50)

X2 = sm.add_constant(train_E3ToE7)
est = sm.OLS(value_E3ToE7, X2)
est2 = est.fit()
print(est2.summary())
print("均方误差："+str(mean_squared_error(value_E3ToE7,linreg.predict(train_E3ToE7))))
print("平均绝对值误差："+str(mean_absolute_error(value_E3ToE7,linreg.predict(train_E3ToE7))))
print("r2："+str(r2_score(value_E3ToE7,linreg.predict(train_E3ToE7))))

#---残差图
f, ax = plt.subplots()
f.tight_layout()
ax.hist(value_E3ToE7[1:1000] - linreg.predict(train_E3ToE7)[1:1000],bins=100, label='loss', color='b');
ax.set_title("loss")
ax.legend(loc='best');
plt.show()
