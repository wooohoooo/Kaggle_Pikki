from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import otto_Pikki as ot
clf2 = RandomForestClassifier(n_estimators = 4000, n_jobs=-1, max_depth = 7)
clf = GradientBoostingClassifier(n_estimators = 1000, subsample = .8, max_features = 20,max_depth = 10)
n1 = ot.make_guess_csv(clf,name='turkeytest')
n2 = ot.make_guess_csv(clf2,name='turkeytest')

ot.combine_results(clf_list=[n1,n2],voting='soft')
