
import pandas as pd
import numpy as np


data = pd.read_csv("diabetic_data.csv")
#print(data.shape)

#计算出各项缺失的数据的个数，删掉缺失过多数据的项
datacopy = data.copy()
Rep = datacopy.replace('?', np.NaN)
nacheck = Rep.isnull().sum()
#print(nacheck)

datacopy= datacopy.drop(['weight','payer_code','medical_specialty'],axis=1)
#print(datacopy.shape)

##make function to recode readmission, 0 is no readmission or >30 days, <30 days is an early readmission
def recode(x):
    if x == 'NO'or x == '>30':
        return 0
    else:
        return 1
##recode readmission
datacopy['readmitted'] = datacopy['readmitted'].apply(recode)
#print(datacopy.groupby('readmitted').size())
#print(datacopy.shape)
datacopy = datacopy.loc[datacopy.groupby("patient_nbr")["encounter_id"].idxmin()]

datacopy = datacopy.reset_index()
datacopy["patient_nbr"] = datacopy["index"]


datacopy = datacopy.drop(["index"], axis=1)


#删除discharge_disposition_id的不合格项
a = [11, 13, 14, 19, 20, 21]
datacopy = datacopy[~datacopy.discharge_disposition_id.isin(a)]
#删除那些有缺失数据的行
b=["?"]
datacopy = datacopy[~datacopy.race.isin(b)]
datacopy = datacopy[~datacopy.diag_3.isin(b)]
datacopy = datacopy[~datacopy.diag_2.isin(b)]
datacopy = datacopy[~datacopy.diag_1.isin(b)]
#print(datacopy.shape)

#将年龄由范围变为数字
def reage(x):
    if x == "[0-10)":
        return 5
    elif x == "[10-20)":
        return 15
    elif x == "[20-30)":
        return 25
    elif x == "[30-40)":
        return 35
    elif x == "[40-50)":
        return 45
    elif x == "[50-60)":
        return 55
    elif x == "[60-70)":
        return 65
    elif x == "[70-80)":
        return 75
    elif x == "[80-90)":
        return 85
    else:
        return 95
datacopy['age'] = datacopy['age'].apply(reage)
#print(datacopy['age'])
datacopy= datacopy.drop(['max_glu_serum','A1Cresult','number_outpatient','number_emergency','number_inpatient'],axis=1)
#datacopy.to_csv("dataset_cleaned1.csv", sep=",")

#datacopy2 = pd.read_csv("dataset_cleaned1.csv")
#print(datacopy2.shape)
datacopy2=datacopy
def recodeYandN(x):
    if x == 'No':
        return 0
    elif x == 'Yes':
        return 1
datacopy2['diabetesMed'] = datacopy2['diabetesMed'].apply(recodeYandN)
#print(datacopy2["diabetesMed"])
def recodeChange(x):
    if x== 'No':
        return 0
    elif x == 'Ch':
        return 1
datacopy2['change'] = datacopy2['change'].apply(recodeChange)
def recodeGender(x):
    if x== 'Female':
        return 0
    elif x == 'Male':
        return 1
datacopy2['gender'] = datacopy2['gender'].apply(recodeGender)
def recodeInsulin(x):
    if x== 'No':
        return 0
    elif x == 'Steady':
        return 1
    elif x == 'Up':
        return 2
    elif x == 'Down':
        return 3
datacopy2['insulin'] = datacopy2['insulin'].apply(recodeInsulin)
datacopy2['rosiglitazone'] = datacopy2['rosiglitazone'].apply(recodeInsulin)
datacopy2['pioglitazone'] = datacopy2['pioglitazone'].apply(recodeInsulin)
datacopy2['glyburide'] = datacopy2['glyburide'].apply(recodeInsulin)
datacopy2['glimepiride'] = datacopy2['glimepiride'].apply(recodeInsulin)
datacopy2['chlorpropamide'] = datacopy2['chlorpropamide'].apply(recodeInsulin)
datacopy2['repaglinide'] = datacopy2['repaglinide'].apply(recodeInsulin)
datacopy2['metformin'] = datacopy2['metformin'].apply(recodeInsulin)
datacopy2['glipizide'] = datacopy2['glipizide'].apply(recodeInsulin)

def recodeRace(x):
    if x== 'AfricanAmerican':
        return 0
    elif x == 'Asian':
        return 1
    elif x == 'Caucasian':
        return 2
    elif x == 'Hispanic':
        return 3
    else:
        return 4
datacopy2['race'] = datacopy2['race'].apply(recodeRace)

datacopy2= datacopy2.drop(['nateglinide','acetohexamide','tolbutamide','acarbose','miglitol','troglitazone','tolazamide','examide','citoglipton'],axis=1)
datacopy2= datacopy2.drop(['glyburide-metformin','glipizide-metformin','glimepiride-pioglitazone','metformin-rosiglitazone','metformin-pioglitazone'],axis=1)


data=datacopy2
data=data.dropna(axis=0,how='any')

data = data.drop(["encounter_id","patient_nbr"], axis=1)



#print(data.shape)
## Recoding "diag_1" items[18]
# 1: 390–459, 785
# 2: 460–519, 786
# 3: 520–579, 787
# 4: 250.xx
# 5: 800–999
# 6. 710–739
# 7. 580–629, 788
# 8. 140–239, 780, 781, 784, 790–799, 240–279, without 250, 680–709, 782, 001–139
# 9. 290–319, E–V, 280–289, 320–359, 630–679, 360–389, 740–759

def recodeDiag(x):
    diag = x

    if (diag >='390' and diag <= '459') or diag == '785':
        return 1
    elif (diag >= '460' and diag <= '519') or diag == '786':
        return 2
    elif (diag >= '520' and diag <= '579') or diag == '787':
        return 3
    elif diag > '250' and diag < '251':
        return 4
    elif diag >= '800' and diag <= '999':
        return 5
    elif diag >= '710' and diag <= '793':
        return 6
    elif (diag >= '580' and diag <= '629') or diag == '788':
        return 7
    elif (diag >= '140' and diag <= '239')or diag == '780'or diag == '781'or diag == '784'or (diag >= '790' and diag <= '799') or(diag >= '240' and diag <= '279') or (diag >= '680' and diag <= '709') or diag == '782'or(diag >= '1' and diag <= '139'):
        return 8
    else:
        return 9



data['diag_1'] = data['diag_1'].apply(recodeDiag)
data['diag_2'] = data['diag_2'].apply(recodeDiag)
data['diag_3'] = data['diag_3'].apply(recodeDiag)

data.to_csv("dataset_cleanedF.csv")
