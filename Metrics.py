"""
Tasks
Regression
Classification
Ordinal
    order is important
    e..g predicting magnitude of an earthquake
    the most common way is treat it as multiclass problem
          here the prediction will not take into account that the classes have certain order
          u get a feeling that there is problem if u look at the prediction probability for the classes
          u often get a asymtreic distribution whereas u shd get a gaussian distribution around the max distribution probability class
    the other way is to treat it as a regression and post process ur result.
        the order will be taken into account but some soesticated post processing may be needed as this might lead to inaccuracies


Metrics
AUC , Logloss - classification binary
MAP@K - recomender systems
RMSE, RMSLogError , Quadratic weightd kappa - regression

Handling never seen metrics before
    1) Kaggle Discussion forums
    2) Try to experiment with it by coding the evaluation function; how metric reacts to different types of errors 
            Page 107 from kaggle book; sample articles  
            1) https://www.kaggle.com/code/carlolepelaars/understanding-the-metric-spearman-s-rho
            2) https://www.kaggle.com/code/carlolepelaars/understanding-the-metric-quadratic-weighted-kappa
            3) https://www.kaggle.com/code/rohanrao/osic-understanding-laplace-log-likelihood

Metrics for regression    
    MAE 
    MSE & R squarred
      mean of the good old SSE
      MSE = 1/n * SSE
      R2 = SSE ( y_pred - y_act) / SST ( y - ymean ) ( sum of squares total )
            R2 compares the squared error of the model against the squarred errors agains the simplest model ( mean )
            R2 can help u determine whether transforming your target is helping to obtain better predictions

        MSE is seldom used but RMSE is preffered; because the value will resemble origial scale of ur target & its easier to glance to figure out if the model is doing good
        R2 is better becaue it is perfectly correlated with MSE & values range between 0 & 1 making all comparisions better           
    RMSE
        In MSE, Large prediction errors are greatly penalized but in RMSE the effect is lessened because of root effect
        you can get a better fit using MSE as objective function by first applying root to ur target
        functions such as TransformedargetRegressor in scikit learn help you to appropriatly transform target
    RMSLE
        RMSLE = sqrt (  1/n * sum ((log(y_pred+1) - log(Y_act+1)) ** 2)   )
        you dont penealize huge differences; you care most about is the scale of your predictions w.r.t scale of ground truth
        RMSLE is most used metric 
            Why RMSLE Is More Frequently Used (in Certain Contexts):
                Handles Scale Variance: RMSLE is more suitable when the target variable yy spans multiple orders of 
                magnitude or when smaller values are as important as larger ones. Logarithmic transformation reduces the impact of scale differences.
            Relative Performance Matters: RMSLE penalizes relative errors more than absolute errors, making it appropriate for use cases where percentage-based 
                accuracy is more critical than absolute error values (e.g., forecasting, sales predictions).
            Robust to Outliers: Since it operates on the logarithmic scale, RMSLE is less sensitive to large outliers compared to RMSE. 
                This property can be crucial for datasets with skewed distributions or extreme values.
            Favors Under-predictions: RMSLE penalizes under-predictions more than over-predictions. 
                For instance, underestimating demand in inventory predictions can lead to stockouts, making RMSLE desirable in such applications.
    MAE
        MAE is not sesitive to outliers
        results in much slower convergenge; since u are optimizing for predicting the median of the target ( l1 Norm ) instead of the mean ( l2 norm )
            more complex computations for the optimizer ; training time grow exponentially
        used in forecasting tasks

        Downsides of MAE
            Non-Differentiability at Zero: The absolute value function in MAE is not differentiable at zero, making it harder to optimize directly using gradient-based methods. 
                Specialized techniques, such as subgradient methods, are required.
            Not Sensitive to Large Errors: MAE gives equal weight to all errors, regardless of their magnitude. This can be a limitation in cases where large errors should be 
                penalized more (e.g., MSE penalizes larger errors disproportionately).
            Lacks Interpretability for Squared Units: In some applications (e.g., physics or engineering), squared error metrics might be preferred because the units are squared, 
                which can better align with domain-specific interpretations.
            Doesn't Prioritize Variability: MAE might not capture the variance of the errors effectively, which could lead to poorer performance in tasks where consistent 
                small errors are preferable over occasional large ones.
            Sensitivity to Outliers in Non-Symmetric Distributions: While MAE is more robust to outliers than MSE, it can still be affected by skewed data distributions, 
                where outliers disproportionately influence the result.
        In summary, MAE is a solid choice when simplicity and robustness to outliers are desired, but it may not be the best option when dealing with tasks that require 
            sensitivity to larger errors or gradients for optimization


    Metrics for Classification
        Binary Classification            
        Multiclass class
        Multilabel


        Binary Classification
            acuracy = correct answers / total answers
                focussed strongly on the effective performance of the model; can lead to wrong conslusions when the class is imbalanced
            confusion matrix
                Actual vs Predicted
                    Accuracy = TP + TN / ( Tp + TN + FP + FN )
                    precision = TP / TP + FP - accuracy of the positive class
                        the metric tells you how often you are correct when u predict a positive class
                    recall =  TP / TP + FN 
                    remember precision/recall trade-off

            F1 - Score
                the harmnic mean of precision & recall 
                    F1 = 2 * ( precision * recall ) / precision + recall )
                    high F1 score - model has improved in precision or recall or both
                F-Beta score - weighted harmonic mean between precision & recall
                    F1 =   ( (1 + Beta2) * precision * recall ) / (beta2) + precision + recall )

            Log Loss & ROC-AUC
                Log Loss , also knnown as cross-entropy in deep learning
                log loss = -1/n*sum( (y_act*log(y_pred)) + (1-y_act*log(1-y_pred)) )
                    here the object is estimate as correctly as possible  the prob of the example of being a positive class

                    Explanation
                        Predicted Probabilities (y^): Logistic regression outputs probabilities, which must be between 0 and 1. 
                        Penalty for Incorrect Predictions: Log loss heavily penalizes confident but incorrect predictions. For example:
                        If y=1 and y^ is close to 0, the loss becomes very large due to the log⁡(y^) term.
                        Similarly, if y=0 and y_pred is close to 1, the log(1-y_pred) term dominates and leads to a large loss.

                    Properties
                            Range: Log loss is always non-negative. A lower log loss indicates better model performance.
                            Perfect Prediction: If the model predicts probabilities perfectly (i.e., y_pred=1 when y=1 and y_pred=0 when y=0), the log loss is zero.
                            Probabilistic Interpretation: Log loss is based on the likelihood of the true labels under the predicted probability distribution, 
                                    making it suitable for probabilistic models.
                    Practical Use
                        Model Training: Log loss is often used as the objective function in logistic regression to optimize model parameters via gradient descent.
                        Evaluation Metric: It provides a continuous measure of model performance and is preferred over accuracy in cases where probabilistic outputs are important 
                        (e.g., imbalanced datasets).

                ROC
                    TP rate vs False positive Rate
                    ideally a well performing classifier shd quickly climb up the true positive rate at low values of False positive rate
                    0.9 to 1.0 is very good model
                    if the classes are balanced or not too imbalanced, increases in the AUC are proportional to the effectiveness of the trained model
                    if the classes are imbalanced or rare, AUC starts high & its increments may mean very little in terms of predicting the rare class better;
                        Average precision is more helpfull

                Mathews correlation coefficient ( MCC )
                        MCC = (TP*TN) - (FP * FN) / sqrt ( (TP+FP) * (TP+FN) * (TN+FP) * (TN+FN) )
                        behaving as a correlation coeff ranginf from +1 ( perfect prediction) to -1 ( inverse prediction); this metric can be considered a measure of the quality 
                            of the prediction even when the classes are quite imbalanced

                        Neuron Engineer Understanding of hte ratio 
                                MCC = ( Pos_precision + Neg_precision -1 ) * PosNegRatio

                                Pos_pre = TP / TP + FP
                                Neg_pre = TN / TN + FN
                                PosPredCount = TP + FP
                                NegPredCount = TN + FN
                                PosNegRatio = sqrt ( PosPredCount * NegPredCount /  POSLabelCount * NegLabelCount )

                            You can ge higher performance from imprving both positive & negative class precision but thats not enough; you also have to have pos * neg pred 
                                in proportional to the ground truth or your submission will be greatly penalized

        
        Metric for Multi Class classification
            when moving to multi class classification you simply use binary classification metrics applied to each class and then you summarize them 
            using some of hte averageing strategies 

            e.g. if you want to evaluate your solution based on F1 score you have 3 avg choices

            Macro averaging
                Calculate F1 score for each class and then average -  multiple F1 scores averaged
                    each class will count as much as others no matter how frequent / how imp they are. therefore in equal penalizations when the model doesnt perform well
                Macro = F1Class1 + F2class2 +.......F1classn / N0

            Micro averaging
                wil sum all the contributions from each class to compute agg F1 score; it results in no particular fevor to or penalization of any class, 
                    it can more accuratly account for imbalance class
                Micro = F1class1+class2+class3+.....classn - one single F1 score
            Weighting
                F1 score of each class & then make a weighted avg mean of all of them; this approach clearly favours majority class 
                weighted = F1class*W1 + F1class2 * W2 + .....+ F1classn * Wn
                        W1 + W2 + ...+ Wn = 1.0
            
            The below are just listed and has to be individually researched

                Common Multiclass Metrics
                    Multiclass accuracy
                    Multiclass Logloss
                    Macro F1 & Micro F1
                    Mean  F1
                    Quadratic Weighted Kappa  / Cohen Kappa & 
                    Inter annotation aggrement 

                Metrics for object detection Problems
                    Intersection over union
                    Dice 

                Metrics for multilabel classification & recommendation problems
                    P @ K
                    MAP@K  - mean avg precision @ K
                    AP @ K


        Optimizing Evaluation Metrics
            An objective function is inside your learning algo that measures how well the algo's model is fitting the data. the function provides feedback to the algo 
            in order to for it to fit better in successive iterations; if the evaluation metric perfectly matches the objective function. the best results are obtained.

            often the evaluation metric provided can only be approx by existing objective functions; when ur ojective function does not match your evaluation metric, you have few alternatives
                1) Modify your learning algo and have it incorporate an objective function that matches the evaluation metric
                        LightGBM & XgBoost alow you to set custom objective function
                2) Tune ur hyperparametrers, chosing the ones that make the best results when using  the evaluation metric
                3) post processing ur results so they match the evaluation criteria more closely ( discussed more )

                if you dont have the evaluation function coded then
                    1) Browse thru most common packages - https://scikit-learn.org/stable/modules/model_evaluation.html
                    2) browse github projects - Ben hammer metrics projects
                    3) Kaggle meta Dataset - competitions table ( which competition uses the same evaluation metrics )

                
            Custom Metrics & Custom Objective functions
                You can use custom objective functions when using XGboost, catBoost & lightGBM as well as deep learning model based on Tensor flow & pytorch
                https://petamind.com/advanced-keras-custom-loss-functions/
                https://kevinmusgrave.github.io/pytorch-metric-learning/extend/losses/
                https://towardsdatascience.com/custom-metrics-in-keras-and-how-simple-they-are-to-use-in-tensorflow2-2-6d079c2ca279

                if you need to create a custom loss in lightGBM , XGBOOST; you have to code a function that inputs as prediction & fround truth and return the outputs as graadient & hessian

                    Feature	            Gradient	                        Hessian
                    Definition      	Vector of first derivatives.	    Matrix of second derivatives.
                    Dimension	        Vector (n*1).	                    Square matrix (n*n).
                    Purpose	            Direction of steepest ascent.	    Curvature information.
                    Use Case	        First-order optimization methods.	Second-order optimization methods.

                    Refer to page 137 in Kaggle book for example 
                    xgb = xgb.XBGClassifier(objective=focal_loss(alpha=0.25,gamma=1))
                    Read more about this - maxhallford.github.io/blog/lightgbm-focal-loss/

                    if building your own objective function is too ambitious then code it as a custom evaluation metric, though ur model wont be directly optimized to perform against this function; 
                    you can still improve performance with hyperparam tuuning

            Post Processing your Predictions
                Post processing tuning implies that ur predictions are transformed, by means of a function into something else in order to present a better evaluation. 
                After building ur custom loss or optimizing for ur evaluation metric; you can improve ur results by leveraging the char of ur evaluation metric using a 
                specific function applied to ur predictions
            
            Predicted Probability & its adjustment
                 There are situations where it is paramount to predict correct probabilities
                 The main problems to look for when striving to achieve correct prob pred with you model are
                    1) models that do not return a truly prob estimate
                    2) unbalanced distribution of classes in ur problem
                    3) diff class distribution between your training data & test data

                    even if u use predict_proba methos; this is a very weak assurance that they will return a true probability
                    Be it Decision Trees & other models ; prob estimates that are not truy based on soid prob estimations - this affects many many other models.

                    Aside from the model, the presence of imbalance between classes also result in models that are not reliable. 
                    hence a good approach i the case of unbalanced data is to rebalance by undersampling or oversampling or custom weights for each class to be applied.
                    All these may render your model more performant however they will surly distort the prob estimates. you may have to adjust them

                    Finally, a third point of concern is related to how the test set is distributed. 

                    from a general point of view assuming that you do not have an idea of the test distribution but it is beneficial to gather info from training data
                    if will be much easier to correct your predicted probabilities from the training data; the overall performance will be better

                    sklearn.calibration.CalibratedClassifierCV(base_estimator=None,*,method='sigmoid',cv=None,n_job=None,ensemble=True)

                    "The purpose of the caibration function is to apply a post processing function to ur predicted prob in order to make them adhere to emperical prob seen int he ground truth.

                    Two options
                        1) Sigmoid method(plat's scaling )
                        2) isotonic regresion ( non parametric ; it teds to overfit )

                    you also have to chose how to fit this calibrator. remember that this is a model that is applied to the results of ur model;
                    you have to avoid overfitting by systematically reworking predictions. you could use cross val

                    




















                    

            
                





                        
"""