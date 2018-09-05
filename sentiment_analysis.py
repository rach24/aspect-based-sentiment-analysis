
# coding: utf-8

# Importing all the Libraries

# In[1]:


import numpy as np
import pandas as pd
import re
import string
from textblob import TextBlob
from textblob import Word
from nltk.stem.wordnet import WordNetLemmatizer


# In[8]:


df= pd.read_csv("Data-1_train.csv", skiprows =1, names = ['ID', 'Text', 'Aspect_Term', 'Term_Location', 'Class'], index_col = 'ID')
text = df.Text
aspect = df.Aspect_Term
term_Loc = df.Term_Location


# In[7]:


butt = {'or','nor','so', 'yet','because', 'but', 'that','expect', 'while','after','although','plus'}
from textblob import Word
def get_lexicon_value(sentence, termLoc):
    sentence = sentence.replace("[comma]", ",")
    splitLocations = termLoc.split("--");
 
    beforeAspect, afterAspect = sentence[:int(splitLocations[0])], sentence[int(splitLocations[1]):]
    beforeWords =beforeAspect.replace(",", " XX ")
    afterWords = afterAspect.replace(",", " XX ")
    beforeWords = beforeWords.split()
    afterWords = afterWords.split()

    start = 0
    for i, words in enumerate(beforeWords):
        if words in butt:
            start = max(start, i)
    if start == 0:
        final_sentence = ' '.join(beforeWords[start:]) + ' '
    else:
        final_sentence = ' '.join(beforeWords[start+1:]) + ' '


    pos = len(afterWords)
    i = 0
    for i, words in enumerate(afterWords):
        if words in butt:
            pos = i
            break
    final_sentence += ' '.join(afterWords[:pos])
    return re.sub('\W+', ' ', final_sentence)


# In[4]:


dummy = []
for i in range(len(text)):
    dummy.append(get_lexicon_value(text[i],term_Loc[i]))


# In[5]:


sub = []
pol = []
for i in dummy:
    val = TextBlob(i).sentiment
    pol.append(val[0])
    sub.append(val[1])


# In[6]:


df['Pol'] = np.array(pol).reshape(-1,1)
df['Sub'] = np.array(sub).reshape(-1,1)
Y = np.array(df['Class']).reshape(-1,1)
# X = df['Pol'].reshape(-1,1)
X = df[['Pol', 'Sub']]
X_train = X[:2000]
y_train = Y[:2000]
X_test = X[2000:]
y_test = Y[2000:]


# In[7]:


from sklearn import preprocessing, cross_validation, svm
from sklearn.model_selection import cross_val_score
from textblob.sentiments import NaiveBayesAnalyzer
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(alpha=1)
scores = cross_val_score(clf , X , Y, cv =10 , scoring = 'accuracy')
print scores.mean()


# In[8]:


from sklearn.svm import LinearSVC
clf = LinearSVC(random_state=0, multi_class='ovr')
scores = cross_val_score(clf , X , Y, cv =10 , scoring = 'accuracy')
print scores.mean()


# In[9]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

bdt_real = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=3),
    n_estimators=600,
    learning_rate=1)
scores = cross_val_score(bdt_real , X , Y, cv =10 , scoring = 'accuracy')
print scores.mean()

bdt_discrete = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=3),
    n_estimators=600,
    learning_rate=1.5,
    algorithm="SAMME")
scores = cross_val_score(bdt_discrete , X , Y, cv =10 , scoring = 'accuracy')
print scores.mean()


# Testing using Trained classifier

# In[34]:


df= pd.read_csv("Book1.csv", skiprows =1, names = ['ID', 'Text', 'Aspect_Term', 'Term_Location', 'Class'])


# In[35]:


df.head()


# In[40]:


text = df.Text
aspect = df.Aspect_Term
term_Loc = df.Term_Location
sub = []
pol = []

dummy = []
for i in range(len(text)):
    dummy.append(get_lexicon_value(text[i],term_Loc[i]))

for i in dummy:
    val = TextBlob(i).sentiment
    pol.append(val[0])
    sub.append(val[1])

df['Pol'] = np.array(pol).reshape(-1,1)
df['Sub'] = np.array(sub).reshape(-1,1)
Y1 = np.array(df['Class']).reshape(-1,1)
# X = df['Pol'].reshape(-1,1)
X1 = df[['Pol', 'Sub']]
bdt_discrete.fit(X,Y)
result = bdt_discrete.predict(X1)
# result = accuracy_score(Y1, y_pred)


# In[43]:


#outputting the results to the file
#change the file name.

f = open('output.txt','a')
id = df.ID

for i in range(len(df['ID'])):
    f.write(str(id[i])+";;"+str(result[i])+'\n')
f.close()


# Additional Methods Tried

# In[ ]:


from sklearn import model_selection
seed = 10
num_trees = 100
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print(results.mean())


# In[ ]:


from sklearn import svm
from sklearn.model_selection import GridSearchCV
svc = svm.SVC()
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
clf = GridSearchCV(svc, parameters)
clf.fit(X,Y)


# In[ ]:


scores = clf.cv_results_['mean_test_score']
scores_std = clf.cv_results_['std_test_score']


# In[ ]:


print scores


# In[ ]:


print scores_std


# In[ ]:


clf = svm.SVC()
# if tune_hyper_parameters:
parameters = {
'C': np.arange(1, 5, 1).tolist(),
'kernel': ['rbf', 'poly'],  # precomputed,'poly', 'sigmoid'
'degree': np.arange(0, 3, 1).tolist(),
'gamma': np.arange(0.0, 1.0, 0.1).tolist(),
'coef0': np.arange(0.0, 1.0, 0.1).tolist(),
'shrinking': [True],
'probability': [False],
'tol': np.arange(0.001, 0.01, 0.001).tolist(),
'cache_size': [2000],
'class_weight': [None],
'verbose': [False],
'max_iter': [-1],
'random_state': [None],
}
gs_clf = GridSearchCV(clf, parameters, n_jobs=-1)
gs_clf.fit(X_train, y_train)
print(gs_clf.best_score_)


# In[ ]:


for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))


# In[ ]:


lala =[]
for i in range(len(dummy)):
    if df['Class'][i] == 0:
        lala.append(TextBlob(dummy[i]).sentiment.polarity)


# In[ ]:


print max(lala)
print sum(lala)/len(lala)

