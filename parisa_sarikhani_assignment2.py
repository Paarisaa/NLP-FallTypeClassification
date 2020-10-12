# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 16:01:53 2020

@author: psarikh
"""
import nltk
from collections import defaultdict
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from nltk.stem.porter import *
from nltk.corpus import stopwords
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt



# st = stopwords.words('english')
stemmer = PorterStemmer()


word_clusters = {}

def loadwordclusters():
    infile = open('./50mpaths2.txt', encoding='utf-8')
    for line in infile:
        items = str.strip(line).split()
        class_ = items[0]
        term = items[1]
        word_clusters[term] = class_
    return word_clusters

def getclusterfeatures(sent):
    sent = sent.lower()
    terms = nltk.word_tokenize(sent)
    cluster_string = ''
    for t in terms:
        if t in word_clusters.keys():
                cluster_string += 'clust_' + word_clusters[t] + '_clust '
    return str.strip(cluster_string)



def preprocess_text(raw_text):
    '''
        Preprocessing function
        PROGRAMMING TIP: Always a good idea to have a *master* preprocessing function that reads in a string and returns the
        preprocessed string after applying a series of functions.
    '''
    # Replace/remove username
    # raw_text = re.sub('(@[A-Za-z0-9\_]+)', '@username_', raw_text)
    #stemming and lowercasing (no stopword removal
    words = [stemmer.stem(w) for w in raw_text.lower().split()]
    return (" ".join(words))



if __name__ == '__main__':
    
    # load the data:
    f_path = './pdfalls.csv'
    df = pd.read_csv(f_path)
    df = df.sample(frac = 1) 
    trainig_len = int(0.8*len(df))
    
    trainig_df = df[:trainig_len]
    test_df = df[trainig_len:]
    
    # trainig features:
    feature_age = [str(a) for a in trainig_df['age'].values]
    feature_gender = trainig_df['female'].values
    featute_duration =  [str(a) for a in trainig_df['duration'].values]
    feature_day =  [str(a) for a in trainig_df['fall_study_day'].values]
    feature_location = trainig_df['fall_location'].values
    feature_text =  trainig_df['fall_description'].values
    
    training_classes = trainig_df['fall_class'].values
    
    # testing_features:
    feature_age_test =  [str(a) for a in test_df['age'].values]
    feature_gender_test = test_df['female'].values
    featute_duration_test = [str(a) for a in test_df['duration'].values]
    feature_day_test =  [str(a) for a in test_df['fall_study_day'].values]
    feature_location_test = test_df['fall_location'].values
    feature_text_test = test_df['fall_description'].values    
    
    test_classes = test_df['fall_class'].values


    #PREPROCESS THE text feature:
    training_texts_preprocessed = []
    test_texts_preprocessed = []
    test_clusters = []
    training_clusters = []
    
    # length of text as another feature extracted from text
    training_length = []
    test_length =[]
    
    for tr in feature_text:
        training_texts_preprocessed.append(preprocess_text(tr))
        training_length.append(str(len(tr)))
        training_clusters.append(getclusterfeatures(tr))

    for tt in feature_text_test:
        test_texts_preprocessed.append(preprocess_text(tt))
        test_length.append(str(len(tt)))
        test_clusters.append(getclusterfeatures(tt))

    # VECTORIZE features
    vectorizer_onegram = CountVectorizer(ngram_range=(1, 1), max_features=1000)
    vectorizer_bigram = CountVectorizer(ngram_range=(1, 2), max_features=1000)
    # clustervectorizer = CountVectorizer(ngram_range=(1,1), max_features=1000)
    lenvectorizer = CountVectorizer(ngram_range=(1,1), max_features=1000)
    other_features_vectorizer = CountVectorizer(ngram_range=(1,1), max_features=100) # for age, gender, duration, day, location
    # ------------------
    training_data_vectors = vectorizer_onegram.fit_transform(training_texts_preprocessed).toarray()
    test_data_vectors = vectorizer_onegram.transform(test_texts_preprocessed).toarray()
    # ------------------
    training_data_vectors_bi = vectorizer_bigram.fit_transform(training_texts_preprocessed).toarray()
    test_data_vectors_bi = vectorizer_bigram.transform(test_texts_preprocessed).toarray()
    # ------------------
    # training_cluster_vectors = clustervectorizer.fit_transform(training_clusters).toarray()
    # test_cluster_vectors = clustervectorizer.transform(test_clusters).toarray()
    # ------------------
    training_len = lenvectorizer.fit_transform(training_length).toarray()
    test_len = lenvectorizer.transform(test_length).toarray()
    # ------------------
    training_age = other_features_vectorizer.fit_transform(feature_age).toarray()
    test_age = other_features_vectorizer.transform(feature_age_test).toarray()
    # ------------------
    training_gender = other_features_vectorizer.fit_transform(feature_gender).toarray()
    test_gender = other_features_vectorizer.transform(feature_gender_test).toarray()
    # ------------------
    training_duration = other_features_vectorizer.fit_transform(featute_duration).toarray()
    test_duration = other_features_vectorizer.transform(featute_duration_test).toarray()
    # ------------------
    training_day = other_features_vectorizer.fit_transform(feature_day).toarray()
    test_day = other_features_vectorizer.transform(feature_day_test).toarray()
    # ------------------
    training_loc = other_features_vectorizer.fit_transform(feature_location).toarray()
    test_loc = other_features_vectorizer.transform(feature_location_test).toarray()
    # ------------------
    

    #concatenate all feautures


    training_data_vectors = np.concatenate((training_data_vectors, training_data_vectors_bi, training_len, 
                                            training_age, training_gender, training_duration, training_day, training_loc), axis=1)
    test_data_vectors = np.concatenate((test_data_vectors,test_data_vectors_bi, test_len, test_age,
                                        test_gender, test_duration, test_day, test_loc),axis=1)


    scoring = ['f1_micro', 'f1_macro', 'accuracy']
    #----- Naive Bayes ------------
    gnb_classifier = GaussianNB()
    classifier = gnb_classifier.fit(training_data_vectors, training_classes)
    print('--------------------- Naive Bayes Classifier --------------------- ')
    cv_gnb = cross_validate(classifier, training_data_vectors, training_classes, cv=5, scoring=scoring)
    print('cross-validation results:', cv_gnb)
    # plot
    gnb_pred = gnb_classifier.score(test_data_vectors, test_classes)
    y_pred = gnb_classifier.predict(test_data_vectors)
    gnb_f1_macro = f1_score(test_classes, y_pred, average='macro')
    gnb_f1_micro = f1_score(test_classes, y_pred, average='micro')
    gnb_score = [gnb_pred, gnb_f1_macro, gnb_f1_macro]
    print('prediction score on test set (accuracy, f1-macro score, f1-micro score):', gnb_score)
    #------------- SVC -----------
    print('---------------------SVC Classifier with hyper parameter tunning:---------------------')
    grid_params = {
         'C': [2**-1, 1, 2**1, 2**3, 2**5],
         'kernel': ['rbf', 'linear'],
    }
    svm_classifier = svm.SVC()
    
    for score in scoring:
        svc_clf = GridSearchCV(svm.SVC(),grid_params, scoring=score)
        svc_clf.fit(training_data_vectors, training_classes)
        print("Best parameters set found on development set:")
        print(svc_clf.best_params_)
    
    cv_svm = cross_validate(svc_clf, training_data_vectors, training_classes, cv=5, scoring=scoring)
    print('SVM cross-validation results:', cv_svm)
    svm_pred = svc_clf.score(test_data_vectors, test_classes)
    
    y_pred_svm = svc_clf.predict(test_data_vectors)
    svm_f1_macro = f1_score(test_classes, y_pred_svm, average='macro')
    svm_f1_micro = f1_score(test_classes, y_pred_svm, average='micro')
    svm_score = [svm_pred, svm_f1_macro, svm_f1_macro]
    print('prediction score on test set (accuracy, f1-macro score, f1-micro score):', svm_score)

    #------------ ensemble ------------------
    print('---------------------Ensemle Classifier---------------------')
    clf1 = svm.SVC(C = 8, kernel='rbf', gamma='scale')
    clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
    clf3 = GaussianNB()
    ens_clf1 = VotingClassifier(estimators=[('svm', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
    ens_clf1 = ens_clf1.fit(training_data_vectors, training_classes)
    cv_ens_clf = cross_validate(ens_clf1, training_data_vectors, training_classes, cv=5, scoring=scoring)
    print('Ensemle Classifier Cross-validation results:', cv_ens_clf)
    ens_pred = ens_clf1.score(test_data_vectors, test_classes)
    y_pred_ens = ens_clf1.predict(test_data_vectors)
    ens_f1_macro = f1_score(test_classes, y_pred_ens, average='macro')
    ens_f1_micro = f1_score(test_classes, y_pred_ens, average='micro')
    ens_score = [ens_pred, ens_f1_macro, ens_f1_micro]
    print('prediction score on test set (accuracy, f1-macro score, f1-micro score):', ens_score)
    # -------------- KNN ---------------------
    print('---------------------K-Neighbor Classifier---------------------')
    knn = KNeighborsClassifier(3)
    knn.fit(training_data_vectors, training_classes)
    cv_knn = cross_validate(knn, training_data_vectors, training_classes, cv=5, scoring=scoring)
    print('K-Neighbor Classifier cross-validation results: ', cv_knn)
    knn_pred = knn.score(test_data_vectors, test_classes)
    y_pred_knn = knn.predict(test_data_vectors)
    knn_f1_macro = f1_score(test_classes, y_pred_knn, average='macro')
    knn_f1_micro = f1_score(test_classes, y_pred_knn, average='micro')
    knn_score = [knn_pred, knn_f1_macro, knn_f1_micro]
    print('prediction score on test set (accuracy, f1-macro score, f1-micro score):', knn_score)
    # ----------- Gaussian Process ---------
    print('---------------------Guassian Process Classifier---------------------')
    kernel = 1.0 * RBF(1.0)
    gpc = GaussianProcessClassifier(kernel=kernel, random_state=0).fit(training_data_vectors, training_classes)
    cv_gpc = cross_validate(gpc, training_data_vectors, training_classes, cv=5, scoring=scoring)
    print('GPC Classifier cross-validation results: ', cv_gpc)
    gpc_pred = gpc.score(test_data_vectors, test_classes)
    y_pred_gpc = gpc.predict(test_data_vectors)
    gpc_f1_macro = f1_score(test_classes, y_pred_gpc, average='macro')
    gpc_f1_micro = f1_score(test_classes, y_pred_gpc, average='micro')
    gpc_score = [gpc_pred, gpc_f1_macro, gpc_f1_micro]
    print('prediction score on test set (accuracy, f1-macro score, f1-micro score):', gpc_score)
    # ------------ Decision Tree -------
    print('---------------------decision tree---------------------')
    dtree = DecisionTreeClassifier(random_state=0)
    dtree.fit(training_data_vectors, training_classes)
    cv_dtree = cross_validate(dtree, training_data_vectors, training_classes, cv=5, scoring=scoring)
    print('Decision Tree Classifier cross-validation results: ', cv_dtree)
    dtree_pred = knn.score(test_data_vectors, test_classes)
    y_pred_dtree = gpc.predict(test_data_vectors)
    dtree_f1_macro = f1_score(test_classes, y_pred_dtree, average='macro')
    dtree_f1_micro = f1_score(test_classes, y_pred_dtree, average='micro')
    dtree_score = [dtree_pred, dtree_f1_macro, dtree_f1_micro]
    print('prediction score on test set (accuracy, f1-macro score, f1-micro score):', dtree_score)
    
    
    fig = plt.figure()
    x = np.arange(5)
    plt.subplot(3, 1, 1)
    plt.plot(x, cv_gnb['test_f1_micro'], c ='b', label='Naive Bayes')
    plt.plot(x, cv_svm['test_f1_micro'], c ='g', label=' Optimized SVM')
    plt.plot(x, cv_ens_clf['test_f1_micro'], c ='r', label=' Ensemble')
    plt.plot(x, cv_knn['test_f1_micro'], c ='c', label=' K-Neighbor')
    plt.plot(x, cv_gpc['test_f1_micro'], c ='m', label=' Guassioan Process')
    plt.plot(x, cv_dtree['test_f1_micro'], c ='k', label=' Decision Tree')
    
    plt.title('Cross validation results on training set')
    plt.ylabel('F1 Micro Score')
    plt.xlabel('Cross Validation Folds')
    plt.legend()
    
    
    plt.subplot(3, 1, 2)
    plt.plot(x, cv_gnb['test_f1_macro'], c = 'b')
    plt.plot(x, cv_svm['test_f1_macro'], c = 'g')
    plt.plot(x, cv_ens_clf['test_f1_macro'] ,c = 'r')
    plt.plot(x, cv_knn['test_f1_macro'], c ='c')
    plt.plot(x, cv_gpc['test_f1_macro'], c ='m')
    plt.plot(x, cv_dtree['test_f1_macro'], c ='k')
    # plt.title('F1 Macro Score')
    plt.ylabel('F1 Macro Score')
    plt.xlabel('Cross Validation Folds')

    
    plt.subplot(3, 1, 3)
    plt.plot(x, cv_gnb['test_accuracy'],c ='b')
    plt.plot(x, cv_svm['test_accuracy'],c ='g')
    plt.plot(x, cv_ens_clf['test_accuracy'], c ='r')
    plt.plot(x, cv_knn['test_accuracy'], c ='c')
    plt.plot(x, cv_gpc['test_accuracy'], c ='m')
    plt.plot(x, cv_dtree['test_accuracy'], c ='k')
    # plt.title('Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Cross Validation Folds')
    
    
    plt.show()
    
#   For the best classifier:
    print('Studying the effect of training size set on performance')
    svm_clf = svm.SVC(C=0.5, gamma='auto', kernel='linear')
    train_size = [int(0.2*trainig_len), int(0.4*trainig_len), int(0.6*trainig_len), int(0.8*trainig_len), int(1.*trainig_len)]
    svm_score_sz = []
    svm_f1_macro_sz = []
    svm_f1_micro_sz = []
    for sz in train_size:
        svc_clf.fit(training_data_vectors[:sz], training_classes[:sz])
        y_pred_svm_sz=[]
        svm_score_sz.append(svc_clf.score(test_data_vectors, test_classes))
        y_pred_svm_sz = svc_clf.predict(test_data_vectors)
        svm_f1_macro_sz.append(f1_score(test_classes, y_pred_svm_sz, average='macro'))
        svm_f1_micro_sz.append(f1_score(test_classes, y_pred_svm_sz, average='micro'))

    fig2 = plt.figure()
    plt.plot(train_size, svm_score_sz, label='Accuracy', linewidth = 4)
    plt.plot(train_size, svm_f1_micro_sz, label='F-1 Micro Score')
    plt.plot(train_size, svm_f1_macro_sz, label='F-1 Macro Score')
    plt. title('Effect of training size on the performance of the optimized SVM classifier')
    plt.xlabel('training size')
    plt.ylabel('Performance')
    plt.legend()

    # Ablation study on best classifier, i.e. optimized SVM
    print('leaving the unigram feature out')
    # leaving the unigrams out:
    training_data_vectors1 = np.concatenate(( training_data_vectors_bi, training_len, 
                                            training_age, training_gender, training_duration, training_day, training_loc), axis=1)
    test_data_vectors1 = np.concatenate((test_data_vectors_bi, test_len, test_age,
                                        test_gender, test_duration, test_day, test_loc),axis=1)
    
    svm_clf = svm.SVC(C=0.5, gamma='auto', kernel='linear')
    svc_clf.fit(training_data_vectors1, training_classes)
    acc_svm1= svc_clf.score(test_data_vectors1, test_classes)
    y_pred_svm1 = svc_clf.predict(test_data_vectors1)
    svm_f1_macro1 = f1_score(test_classes, y_pred_svm1, average='macro')
    svm_f1_micro1 = f1_score(test_classes, y_pred_svm1, average='micro')
    scores1 = [acc_svm1, svm_f1_macro1, svm_f1_micro1]
    print('prediction score on test set (accuracy, f1-macro score, f1-micro score):',scores1)
    #  ---------------------------------------------
    print('leaving the bigram feature out')
    # leaving the unigrams out:
    training_data_vectors2 = np.concatenate((training_data_vectors, training_len, 
                                            training_age, training_gender, training_duration, training_day, training_loc), axis=1)
    test_data_vectors2= np.concatenate((test_data_vectors, test_len, test_age,
                                        test_gender, test_duration, test_day, test_loc),axis=1)
    
    svm_clf = svm.SVC(C=0.5, gamma='auto', kernel='linear')
    svc_clf.fit(training_data_vectors2, training_classes)
    acc_svm1= svc_clf.score(test_data_vectors2, test_classes)
    y_pred_svm1 = svc_clf.predict(test_data_vectors2)
    svm_f1_macro1 = f1_score(test_classes, y_pred_svm1, average='macro')
    svm_f1_micro1 = f1_score(test_classes, y_pred_svm1, average='micro')
    scores2 = [acc_svm1, svm_f1_macro1, svm_f1_micro1]
    print('prediction score on test set (accuracy, f1-macro score, f1-micro score):',scores2)
    # ----------------------------------------------------------------------
    print('leaving the length feature out')
    # leaving the unigrams out:
    training_data_vectors3 = np.concatenate((training_data_vectors, training_data_vectors_bi,
                                            training_age, training_gender, training_duration, training_day, training_loc), axis=1)
    test_data_vectors3= np.concatenate((test_data_vectors, test_data_vectors_bi, test_age,
                                        test_gender, test_duration, test_day, test_loc),axis=1)
    
    svm_clf = svm.SVC(C=0.5, gamma='auto', kernel='linear')
    svc_clf.fit(training_data_vectors3, training_classes)
    acc_svm1= svc_clf.score(test_data_vectors3, test_classes)
    y_pred_svm1 = svc_clf.predict(test_data_vectors3)
    svm_f1_macro1 = f1_score(test_classes, y_pred_svm1, average='macro')
    svm_f1_micro1 = f1_score(test_classes, y_pred_svm1, average='micro')
    scores3 = [acc_svm1, svm_f1_macro1, svm_f1_micro1]
    print('prediction score on test set (accuracy, f1-macro score, f1-micro score):',scores3)


   
