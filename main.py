# INSTALL PACKAGES ---------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# MAKE A FUNCTION FOR THE DESCRIPTIONS -------------------------------------------------------------
club_info = pd.read_csv("./data/lending_club_info.csv")

def feat_info(col_name):
    return club_info.loc[club_info["LoanStatNew"]==col_name, "Description"].values[0]

feat_info("loan_amnt")

# READ DATA ----------------------------------------------------------------------------------------
df = pd.read_csv("./data/lending_club_loan_data.csv")
# so i can see whole df
pd.set_option("display.max_columns", None)
df.info()
df.head()
df.describe()

# can see alot of categorical data
# from glance can see that grade and subgrade are similar, purpose and title are similar -> explore later

# EXPLORATORY DATA ANALYSIS & FEATURE ENGINEERING --------------------------------------------------
sns.countplot(data=df, x="loan_status")
plt.show()
# imbalanced dataset
df["loan_status"].value_counts()
len(df[df["loan_status"]=="Fully Paid"])/len(df)
# 80% of dataset is fully paid 

# check missing values
df.isnull().sum() 
# percentage missing
df.isnull().sum()/len(df)
df.columns[df.isnull().sum()>0]
# missing values in: 'emp_title', 'emp_length', 'title', 'revol_util', 'mort_acc', 'pub_rec_bankruptcies'

# make df loan status into numerical where fully paid = 1 and charged off = 0
df["loan_repaid"] = df["loan_status"].map({'Fully Paid': 1, 'Charged Off': 0})
df=df.drop("loan_status",axis=1)
# TARGET VARIABLE IS NOW CALLED LOAN REPAID!!!!!!

# check to see relationship between target and all the numerical features
df.select_dtypes(exclude="object").corr()["loan_repaid"].sort_values()
# check in general
plt.figure(figsize=(10,10))
sns.heatmap(df.select_dtypes(exclude="object").corr(), annot=True)
plt.show()

# LET'S CHECK OUT ALL 26 THE FEATURES
# 1) check loan_amnt variable
sns.histplot(data=df, x="loan_amnt")
plt.show()
# check impact continous variable has on target variable
sns.boxplot(data=df,x='loan_repaid',y='loan_amnt')
plt.show()
# can see spikes at all rounded numbers - typical loan amounts 5k, 10k, 15k etc. 
# can see extremely correlated with installment
df["loan_amnt"].corr(df["installment"])
# 95% correlation!
feat_info("loan_amnt")
# 2) check installment variable
feat_info("installment")
df["installment"].value_counts()
sns.scatterplot(data=df, x="loan_amnt", y="installment")
plt.show()
# not completely perfect predictor but can experiment with removing later to see if it improves accuracy

# 3) check term variable
feat_info("term")
df["term"].value_counts()
# don't want to leave as categorical data because the 36 and 60 can have a numerical relationship
df["term"]=df["term"].apply(lambda x: int(x.split(' ')[1]))
df.info()

# 4) check int_rate
feat_info("int_rate")
df["int_rate"].value_counts()
sns.histplot(data=df, x="int_rate")
plt.show()
# check impact continous variable has on target variable
sns.boxplot(data=df,x='loan_repaid',y='int_rate')
plt.show()
# can see that loans that are repaid have lower IR

# 5, 6) check grade and sub_grade as seen in initial glance they look like perfect predictors
feat_info("grade")
feat_info("sub_grade")
df[["grade","sub_grade"]].head(10)
# we can see that subgrade is grade but has more info so drop grade
df=df.drop("grade", axis=1)
# check if subgrade impacts y value
sorted_grade=sorted(df["sub_grade"].unique())
plt.figure(figsize=(14,5))
sns.countplot(data=df, x="sub_grade",order=sorted_grade, hue="loan_repaid")
plt.show()
# can see for f and g grades that the charge off to fully paid level is almost the same
plt.figure(figsize=(10,5))
fg = ['F1', 'F2', 'F3', 'F4', 'F5', 'G1', 'G2', 'G3', 'G4', 'G5']
sns.countplot(data=df[df["sub_grade"].isin(fg)], x="sub_grade",order=fg, hue="loan_repaid")
plt.show()
# hence we can see the lower the grade the more percentage of people charge off than fully paid -> has impact
# change into dummy variable together with other categorical values at end

# 7) checking emp_title which has alot of missing data
feat_info("emp_title")
df["emp_title"].value_counts()
# too many unique values to make anything meaningful 
# also don't want to feature engineer anything so drop
df=df.drop("emp_title", axis=1)

# 8) checking emp_length which has alot of missing data
feat_info("emp_length")
df["emp_length"].value_counts()
# not too many values so check if it has impact on loan status
df["emp_length"].unique()
ordered_emp_length = ['< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years',
                      '6 years', '7 years', '8 years', '9 years','10+ years']
plt.figure(figsize=(10,5))
sns.countplot(data=df, x="emp_length", order=ordered_emp_length, hue="loan_repaid")
plt.show()
# can't tell from just looking to see if emp_length -> need percentages 
emp_length_fp = df[df["loan_repaid"]==1].groupby(df["emp_length"]).count()["loan_repaid"]
emp_length_co = df[df["loan_repaid"]==0].groupby(df["emp_length"]).count()["loan_repaid"]
percent_co = emp_length_co/(emp_length_fp+emp_length_co)
plt.figure(figsize=(10,5))
sns.barplot(percent_co)
plt.show()
# hence we can see that there is barely any effect from emp_length as the charge off is around 20%
# for all emp_length years so remove
df=df.drop("emp_length", axis=1)

# 9) check homeownership
feat_info("home_ownership")
df["home_ownership"].value_counts()
# can see that there are only a few values so we can one-hot encode it 
# however, can see that there are barely any NONE and ANY values so combine it with OTHER column
df["home_ownership"]=df["home_ownership"].replace(['ANY','NONE'],'OTHER')
# plot to check relationship with target
sns.countplot(data=df, x="home_ownership", hue="loan_repaid")
plt.show()
# change into dummy variable together with other categorical values at end

# 10) check annual_inc 
feat_info("annual_inc")
df["annual_inc"].value_counts()
sns.histplot(data=df, x="annual_inc")
plt.show()
# check impact continous variable has on target variable
sns.boxplot(data=df,x='loan_repaid',y='annual_inc')
plt.show()
df["annual_inc"].describe()
# notice that the data has lots of extremely high outliers -> people with extremely high income
# could remove some high outliers later and see if it impacts

# 11) check verification_status
feat_info("verification_status")
df["verification_status"].value_counts()
sns.countplot(data=df, x="verification_status", hue="loan_repaid")
plt.show()
# change into dummy variable together with other categorical values at end

# 12) check issue_d
feat_info("issue_d")
df["issue_d"].value_counts()
# the month when the loan was issued tells us that the loan was issued
# in our case we are trying to predict if they will pay back or not so that we can issue a loan
# hence, this feature is presenting data leakage so we will remove
df=df.drop("issue_d",axis=1)

# 13, 14) check purpose and title (which has alot of missing values)
# saw before at glance that purpose is similar to title (which has alot of missing values)
feat_info("purpose")
feat_info("title")
df[["purpose","title"]].head(10)
# we can see that title is a subcategory of purpose -> they are perfect predictors/repeat info
# let's remove title since it has so much missing data
df=df.drop("title", axis=1)
df["purpose"].value_counts()
# change into dummy variable together with other categorical values at end

# 15) check dti
feat_info("dti")
df["dti"].value_counts()
sns.histplot(data=df, x="dti")
plt.show()
# check impact continous variable has on target variable
sns.boxplot(data=df,x='loan_repaid',y='dti')
plt.show()
# notice that the data has lots of extremely high outliers -> people with extremely high dti
# could remove some high outliers later and see if it impacts

# 16) check earliest_cr_line
feat_info("earliest_cr_line")
df["earliest_cr_line"].value_counts()
# since it is an object im just going to remove the year and make it an int
df["earliest_cr_line"] = df["earliest_cr_line"].apply(lambda x: int(x.split("-")[1]))
df["earliest_cr_line"].value_counts()
sns.histplot(data=df, x="earliest_cr_line")
plt.show()
# check impact continous variable has on target variable
sns.boxplot(data=df,x='loan_repaid',y='earliest_cr_line')
plt.show()

# 17) check open_acc
feat_info("open_acc")
df["open_acc"].value_counts()
sns.histplot(data=df, x="open_acc")
plt.show()
# check impact continous variable has on target variable
sns.boxplot(data=df,x='loan_repaid',y='open_acc')
plt.show()

# 18) check pub_rec
feat_info("pub_rec")
df["pub_rec"].value_counts()
sns.histplot(data=df, x="pub_rec")
plt.show()
# check impact continous variable has on target variable
sns.boxplot(data=df,x='loan_repaid',y='pub_rec')
plt.show()
df["pub_rec"].describe()
# could feature engineer this and make it yes or no or at least put into bins 

# 19) check revol_bal
feat_info("revol_bal")
df["revol_bal"].value_counts()
sns.histplot(data=df, x="revol_bal")
plt.show()
# check impact continous variable has on target variable
sns.boxplot(data=df,x='loan_repaid',y='revol_bal')
plt.show()
df["revol_bal"].describe()
# notice that the data has lots of extremely high outliers -> people with extremely high credit revolving balance
# could remove some high outliers later and see if it impacts

# 20) check total_acc 
feat_info("total_acc")
df["total_acc"].value_counts()
sns.histplot(data=df, x="total_acc")
plt.show()

# 21) check initial_list_status 
feat_info("initial_list_status")
df["initial_list_status"].value_counts()
sns.countplot(data=df, x="initial_list_status", hue="loan_repaid")
plt.show()
# change into dummy variable together with other categorical values at end

# 22) check application_type 
feat_info("application_type")
df["application_type"].value_counts()
sns.countplot(data=df, x="application_type", hue="loan_repaid")
plt.show()
# change into dummy variable together with other categorical values at end

# 23) check mort_acc which has lots of missing values
feat_info("mort_acc")
df["mort_acc"].value_counts()
df["mort_acc"].isnull().sum()/len(df)
# has 10% missing values so let's replace with imputation
# find highly correlated feature
df.select_dtypes(exclude="object").corr()["mort_acc"].sort_values()
# mort_acc is most correlated with total_acc
# find mean mort_acc for each total_acc value
mean_mort = df.groupby(df["total_acc"])["mort_acc"].mean()
# check: mean_mort[19]
def fillmort(mort_acc, total_acc):
    if np.isnan(mort_acc) == True:
        mort_acc = mean_mort[total_acc]
    return mort_acc
df["mort_acc"] = df.apply(lambda x: fillmort(x["mort_acc"], x["total_acc"]), axis=1)

# 24) check address
feat_info("address")
df["address"].value_counts()
# there's too many unique values - going to feature engineer the zipcode out
df["address"].apply(lambda x: x.split(' ')[-1]).value_counts()
# can see that there are only 10 zipcodes so going to do one-hot encoding
df["zipcode"] = df["address"].apply(lambda x: x.split(' ')[-1])
df=df.drop("address", axis=1)
# change into dummy variable together with other categorical values at end

# 25, 26) check revol_util and pub_rec_bankruptcies which have an extremely small % of missing values
df["revol_util"].isnull().sum()/len(df)
df["pub_rec_bankruptcies"].isnull().sum()/len(df)
# both are have missing values that are less than 1% of the df so we will just drop these rows
df=df.dropna()

# change into dummy variable: sub_grade, home_ownership, verification_status, 
# purpose, initial_list_status, application_type, zipcode
df.select_dtypes(include="object").columns
df = pd.get_dummies(data = df, columns=['sub_grade', 'home_ownership', 'verification_status', 'purpose',
       'initial_list_status', 'application_type', 'zipcode'], drop_first=True)
df = df.drop(['sub_grade', 'home_ownership', 'verification_status', 'purpose',
       'initial_list_status', 'application_type', 'zipcode'], axis=1)

# now check if df has any missing values or categorical values left
df.info()
df.isnull().sum()
df.corr()["loan_repaid"].sort_values()

# save cleaned data to a csv
df.to_csv('./data/cleaned_data.csv', index=False)

"""
Notes from cleaning my dataset:
-------------------------------
1. After initial glance at beginning to find columns that have similar/same values - remove perfect predictors.
2. Take note of the continuous variables with high correlation - might want to try removing later.
3. Take note of the missing values.
4. Check that feature has impact on y value [for data that we are going to apply imputation].
5. For categorical data with no missing value we do one-hot encoding unless too many unique values.

Possible changes to improve my dataset that I can implement later:
- remove installment which has 95% correlation with loan_amnt
- remove high outliers in annual_inc, revol_bal, dti
- could change pub_rec to yes or no or at least put into bins or remove outliers
"""