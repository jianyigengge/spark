# -*- coding: utf-8 -*-
#导入需要的模块
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkContext
from pyspark.mllib.classification import SVMWithSGD
from pyspark.mllib.tree import RandomForest

#关闭上一个进程
try:
    sc.stop()
except:
    pass

#创建一个sc
sc=SparkContext('local[1]','lr')

#将数据转为labelpoint格式
def create_label_point(line):
    line=line.strip('\n').split(',')
    return LabeledPoint(float(line[len(line)-1]), [float(x) for x in line[1:len(line)-1]])

#分别对训练集、测试集数据进行转换
data_train=sc.textFile("file:///home/student/Workshop+xiaojianyi/data_train.csv").map(create_label_point)
data_test=sc.textFile("file:///home/student/Workshop+xiaojianyi/data_test.csv").map(create_label_point)

#逻辑回归拟合训练集，并且计算测试集准确率
lrm = LogisticRegressionWithSGD.train(data_train)
pred1=lrm.predict(data_test.map(lambda x:x.features))
label_and_pred1=data_test.map(lambda x: x.label).zip(pred1)
lrm_acc=label_and_pred1.filter(lambda(x,y):x==y).count()/float(data_test.count())
#print("lrm_acc:%f"%lrm_acc)

#SVM拟合训练集，并且计算测试集准确率
svm=SVMWithSGD.train(data_train)
pred2=svm.predict(data_test.map(lambda x:x.features))
label_and_pred2=data_test.map(lambda x: x.label).zip(pred2)
svm_acc=label_and_pred2.filter(lambda(x,y):x==y).count()/float(data_test.count())
#print("svm_acc:%f"%svm_acc)

#随机森林拟合训练集，并且计算测试集准确率
rf = RandomForest.trainClassifier(data_train, numClasses=2,
                            categoricalFeaturesInfo={},
                            numTrees=200,
                            featureSubsetStrategy="auto",
                            impurity="gini",
                            maxDepth=10,
                            maxBins=32,
                            seed=12)
pred3=rf.predict(data_test.map(lambda x:x.features))
label_and_pred3=data_test.map(lambda x: x.label).zip(pred3)
rf_acc=label_and_pred3.filter(lambda(x,y):x==y).count()/float(data_test.count())

#分别打印出三种模型在测试集上的准确率（lr,svm,randomforest）
print("lrm_acc:%f"%lrm_acc)
print("svm_acc:%f"%svm_acc)
print("rf_acc:%f"%rf_acc)
