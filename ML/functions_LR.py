def RunLinearRegression(trainX, trainy, testX, testy):
    
    # 1. SIMPLE LINEAR REGRESSION ----------------------
    error_test = [[] for _ in range(len(testy))]
    simpleLin = cross_validate(LinearRegression(), trainX, trainy, cv=k,return_estimator = True)
    counter = 0
    for i in simpleLin['estimator']:
        simpleLin_y = i.predict(testX) # pick one of the folds' estimators to predict
        error = (simpleLin_y - testy) / testy
        for j in range(len(testy)):
            error_test[j].append(error.iloc[j])

    # taking the mean for each of the folds' predicted value
    simpleLin_error_test = []
    for i in error_test:
        simpleLin_error_test.append(sum(i)/len(i))

    simpleLin_train = cross_val_predict(LinearRegression(), trainX, trainy, cv=k)
    simpleLin_error_train = (simpleLin_train - trainy) / trainy
    
    # 2. RFE FEATURE SELECTION ---------------------------------

    # https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html
    selector = RFECV(LinearRegression(), step=remove, cv=k).fit(trainX,trainy) # only use training data for selecting features, and fitting

    RFELin_train = selector.predict(trainX)
    RFELin_test = selector.predict(testX)

    RFELin_error_train = (RFELin_train - trainy)/ trainy
    RFELin_error_test = (RFELin_test - testy) / testy
    
    # 3. L1/Lasso FEATURE SELECTION ---------------------------

    regL1 = LassoLarsCV(cv=k).fit(trainX,trainy)
    L1Lin_train = regL1.predict(trainX)
    L1Lin_test = regL1.predict(testX)

    L1Lin_error_train = (L1Lin_train - trainy)/trainy
    L1Lin_error_test = (L1Lin_test - testy) / testy

    # 4. L2/Ridge FEATURE SELECTION ---------------------------

    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html
    regL2 = RidgeCV(cv=k).fit(trainX,trainy)
    L2Lin_train = regL2.predict(trainX)
    L2Lin_test = regL2.predict(testX)

    L2Lin_error_train = (L2Lin_train - trainy)/trainy
    L2Lin_error_test = (L2Lin_test - testy) / testy
    
    return simpleLin_error_train, simpleLin_error_test, RFELin_error_train, RFELin_error_test, L1Lin_error_train, \
        L1Lin_error_test, L2Lin_error_train, L2Lin_error_test
