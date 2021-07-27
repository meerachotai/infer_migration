from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import AdaBoostRegressor

est = 200

# source: https://scipy-lectures.org/packages/scikit-learn/auto_examples/plot_linear_model_cv.html
alphas = np.logspace(-3, 1, 50)

alphas = np.logspace(-3, 0.7, 50)

plt.figure(figsize=(5, 3))

for Model in [KernelRidge, Ridge]:
    scores = [cross_val_score(Model(alpha), trainX, trainy, cv=3).mean()
            for alpha in alphas]
    plt.plot(alphas, scores, label=Model.__name__)

plt.legend(loc='lower left')
# plt.legend(loc='lower left')
plt.xlabel('alpha')
plt.ylabel('cross validation score')
plt.tight_layout()
plt.show()

alpha = 2

trainX, testX, trainy, testy = train_test_split(X, y, test_size=test, random_state=seed)
KR = KernelRidge(alpha=alpha).fit(trainX, trainy)

KR_test = KR.predict(testX)
KR_error_test = (KR_test - testy) / testy

KR_train = KR.predict(trainX)
KR_error_train = (KR_train - trainy) / trainy

# cross_val = "\n" + str(k) + "-fold " + "Cross-Validation with training samples"
test_train = "\nTesting samples:" + str(len(testy)) + ", Training samples:" + str(len(trainy))

fig, ax = plt.subplots(1,1,figsize = (10,10), facecolor = 'w')

ax.scatter(trainy,KR_error_train, facecolors='dodgerblue', label="train", s=10) # 'bo'
ax.scatter(testy,KR_error_test, facecolors='indianred',label="test", s=10) # 'ro'
ax.set_title("Log-Log Linear Regression with Kernel Ridge", size=15)
ax.set_ylabel("error in predicted migration rates", fontsize=15)
ax.set_xlabel("actual migration rate ln(m)",fontsize=15)
ax.axhline(y=0, color='k', linestyle='--')
# ax.set_ylim([miny,maxy])
ax.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=15)
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)


fig.suptitle(test_train + "\n" + file + "\nalpha = " + str(alpha), fontsize=15)

fig.tight_layout()

fig.savefig("KR_asym4.png")


regr = AdaBoostRegressor(random_state=seed, n_estimators=est).fit(trainX, trainy)


Boost_train = regr.predict(trainX)
Boost_error_train = (Boost_train - trainy) / trainy
Boost_test = regr.predict(testX)
Boost_error_test = (Boost_test - testy) / testy

fig, ax = plt.subplots(1,1,figsize = (10,10), facecolor = 'w')

ax.scatter(trainy,Boost_error_train, facecolors='dodgerblue', label="train", s=10) # 'bo'
ax.scatter(testy,Boost_error_test, facecolors='indianred',label="test", s=10) # 'ro'
ax.set_title("Log-Log Regression with AdaBoostRegressor", size=15)
ax.set_ylabel("error in predicted migration rates", fontsize=15)
ax.set_xlabel("actual migration rate ln(m)",fontsize=15)
ax.axhline(y=0, color='k', linestyle='--')
# ax.set_ylim([miny,maxy])
ax.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=15)
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)

add = "\nseed:" + str(seed) + ", estimators:" + str(est)
test_train = "\nTesting samples:" + str(len(testy)) + ", Training samples:" + str(len(trainy))

fig.suptitle(test_train + "\n" + file + add, fontsize=15)

fig.tight_layout()

fig.savefig("Boost_asym4.png") # high bias -> high variance