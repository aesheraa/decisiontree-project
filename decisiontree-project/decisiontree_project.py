

#Ba�vurular�n yapay zeka ile de�erlendirilmesi 
#Bu projede Decision Tree kullanarak gelen i� ba�vurular�n� de�erlendiren bir yapay zeka kodluyorum. 
#Decision tree classification : Gelen ba�vurunun de�erlendirmeye al�n�p al�nmayaca��na karar veren Decision tree kullanarak bu projeyi yapaca��m.


#Veri seti algoritmaya verilmi� ve bir model olu�turulmu�tur, model koordinat ekseninde leafler olu�turur. ,
#Bu leaflerin her biri s�n�fland�rmada ayr� bir s�n�ft�r.
#Model olu�turma i�lemi bitti�inde yeni bir ki�inin �r�n� al�p olmayaca��na karar verilirken yeni gelen verinin hangi leafte oldu�una karar verir ve sonu� d�nd�r�r.


#��e al�m s�ras�nda b�y�k firmalar daha �nce i�e ald�klar� ki�ilerin veritaban�nda tuttuklar� bilgileri kullanarak sistemi e�itir ve 
#yeni bir cv veritabanina girdi�inde hen�z IK'cinin �n�ne d��meden decision tree �zerinde ise al�m yap�lmas� olumlu mu diye kontrol ederler. 
#Olumlu de�ilse cv �k cinin �n�ne d��mez.

import numpy as np
import pandas as pd
from sklearn import tree

df = pd.read_csv("DecisionTreesClassificationDataSet.csv")



df.head()

# scikit-learn k�t�phanesi decision tree'lerin d�zg�n �al��mas� i�in her�eyin rakamlsal olmas�n� bekliyor bu nedenle veri setimizdeki t�m Y ve N de�erlerini 0 ve 1 olarak d�zeltiyoruz. 
# Ayn� sebeple e�itim seviyesini de BS:0 MS:1 ve PhD:2 olarak g�ncelliyoruz. map() kullanarak bo� h�creler veya ge�ersiz de�er girilen h�creler NaN ile doldurulacakt�r, 
# buna �uandaki veri setimizde ihtiyac�k�z yok ama sizin ilerde yo�un veri ile �al��t���n�z zaman ihtiyac�n�z olacakt�r.



duzetme_mapping = {'Y': 1, 'N': 0}

df['IseAlindi'] = df['IseAlindi'].map(duzetme_mapping)
df['SuanCalisiyor?'] = df['SuanCalisiyor?'].map(duzetme_mapping)
df['Top10 Universite?'] = df['Top10 Universite?'].map(duzetme_mapping)
df['StajBizdeYaptimi?'] = df['StajBizdeYaptimi?'].map(duzetme_mapping)
duzetme_mapping_egitim = {'BS': 0, 'MS': 1, 'PhD': 2}
df['Egitim Seviyesi'] = df['Egitim Seviyesi'].map(duzetme_mapping_egitim)
df.head()


# Sonuc s�tununu ay�r�yoruz:


y = df['IseAlindi']
X = df.drop(['IseAlindi'], axis=1)


X.head()


# Decision Tree'mizi olu�turuyoruz:



clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,y)



# Prediction yapal�m �imdi
# 5 y�l deneyimli, hazlihaz�rda bir yerde �al��an ve 3 eski �irkette �al��m�� olan, e�itim seviyesi Lisans
# top-tier-school mezunu de�il
print (clf.predict([[5, 1, 3, 0, 0, 0]]))




# Toplam 2 y�ll�k i� deneyimi, 7 kez i� de�i�tirmi� �ok iyi bir okul mezunu �uan �al��m�yor
print (clf.predict([[2, 0, 7, 0, 1, 0]]))




# Toplam 2 y�ll�k i� deneyimi, 7 kez i� de�i�tirmi� �ok iyi bir okul mezunu de�il �uan �al���yor
print (clf.predict([[2, 1, 7, 0, 0, 0]]))




# Toplam 20 y�ll�k i� deneyimi, 5 kez i� de�i�tirmi� iyi bir okul mezunu �uan �al��m�yor
print (clf.predict([[20, 0, 5, 1, 1, 1]]))


# ## Toplu ��renme: Random Forest

# 20 tane decision tree birle�iminden olu�an bir Random Forest kullanarak tahmin yapaca��z:



from sklearn.ensemble import RandomForestClassifier



rnd_fr_clf = RandomForestClassifier(n_estimators=20)
rnd_fr_clf = rnd_fr_clf.fit(X, y)

#Predict employment of an employed 10-year veteran
print (rnd_fr_clf.predict([[10, 1, 4, 0, 0, 0]]))
#...and an unemployed 10-year veteran
print (rnd_fr_clf.predict([[10, 0, 4, 0, 0, 0]]))



