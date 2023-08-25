

#Baþvurularýn yapay zeka ile deðerlendirilmesi 
#Bu projede Decision Tree kullanarak gelen iþ baþvurularýný deðerlendiren bir yapay zeka kodluyorum. 
#Decision tree classification : Gelen baþvurunun deðerlendirmeye alýnýp alýnmayacaðýna karar veren Decision tree kullanarak bu projeyi yapacaðým.


#Veri seti algoritmaya verilmiþ ve bir model oluþturulmuþtur, model koordinat ekseninde leafler oluþturur. ,
#Bu leaflerin her biri sýnýflandýrmada ayrý bir sýnýftýr.
#Model oluþturma iþlemi bittiðinde yeni bir kiþinin ürünü alýp olmayacaðýna karar verilirken yeni gelen verinin hangi leafte olduðuna karar verir ve sonuç döndürür.


#Ýþe alým sýrasýnda büyük firmalar daha önce iþe aldýklarý kiþilerin veritabanýnda tuttuklarý bilgileri kullanarak sistemi eðitir ve 
#yeni bir cv veritabanina girdiðinde henüz IK'cinin önüne düþmeden decision tree üzerinde ise alým yapýlmasý olumlu mu diye kontrol ederler. 
#Olumlu deðilse cv Ýk cinin önüne düþmez.

import numpy as np
import pandas as pd
from sklearn import tree

df = pd.read_csv("DecisionTreesClassificationDataSet.csv")



df.head()

# scikit-learn kütüphanesi decision tree'lerin düzgün çalýþmasý için herþeyin rakamlsal olmasýný bekliyor bu nedenle veri setimizdeki tüm Y ve N deðerlerini 0 ve 1 olarak düzeltiyoruz. 
# Ayný sebeple eðitim seviyesini de BS:0 MS:1 ve PhD:2 olarak güncelliyoruz. map() kullanarak boþ hücreler veya geçersiz deðer girilen hücreler NaN ile doldurulacaktýr, 
# buna þuandaki veri setimizde ihtiyacýkýz yok ama sizin ilerde yoðun veri ile çalýþtýðýnýz zaman ihtiyacýnýz olacaktýr.



duzetme_mapping = {'Y': 1, 'N': 0}

df['IseAlindi'] = df['IseAlindi'].map(duzetme_mapping)
df['SuanCalisiyor?'] = df['SuanCalisiyor?'].map(duzetme_mapping)
df['Top10 Universite?'] = df['Top10 Universite?'].map(duzetme_mapping)
df['StajBizdeYaptimi?'] = df['StajBizdeYaptimi?'].map(duzetme_mapping)
duzetme_mapping_egitim = {'BS': 0, 'MS': 1, 'PhD': 2}
df['Egitim Seviyesi'] = df['Egitim Seviyesi'].map(duzetme_mapping_egitim)
df.head()


# Sonuc sütununu ayýrýyoruz:


y = df['IseAlindi']
X = df.drop(['IseAlindi'], axis=1)


X.head()


# Decision Tree'mizi oluþturuyoruz:



clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,y)



# Prediction yapalým þimdi
# 5 yýl deneyimli, hazlihazýrda bir yerde çalýþan ve 3 eski þirkette çalýþmýþ olan, eðitim seviyesi Lisans
# top-tier-school mezunu deðil
print (clf.predict([[5, 1, 3, 0, 0, 0]]))




# Toplam 2 yýllýk iþ deneyimi, 7 kez iþ deðiþtirmiþ çok iyi bir okul mezunu þuan çalýþmýyor
print (clf.predict([[2, 0, 7, 0, 1, 0]]))




# Toplam 2 yýllýk iþ deneyimi, 7 kez iþ deðiþtirmiþ çok iyi bir okul mezunu deðil þuan çalýþýyor
print (clf.predict([[2, 1, 7, 0, 0, 0]]))




# Toplam 20 yýllýk iþ deneyimi, 5 kez iþ deðiþtirmiþ iyi bir okul mezunu þuan çalýþmýyor
print (clf.predict([[20, 0, 5, 1, 1, 1]]))


# ## Toplu Öðrenme: Random Forest

# 20 tane decision tree birleþiminden oluþan bir Random Forest kullanarak tahmin yapacaðýz:



from sklearn.ensemble import RandomForestClassifier



rnd_fr_clf = RandomForestClassifier(n_estimators=20)
rnd_fr_clf = rnd_fr_clf.fit(X, y)

#Predict employment of an employed 10-year veteran
print (rnd_fr_clf.predict([[10, 1, 4, 0, 0, 0]]))
#...and an unemployed 10-year veteran
print (rnd_fr_clf.predict([[10, 0, 4, 0, 0, 0]]))



