"""
Validation Strategies
    Having a proper validation strategy can help you decide which of ur models shd be picked. 


Bias & Variance
    A god validation system helps you with metrics that are more reliable than the error measures u get from the training set.
    Metrics obtained on the test set are affected by the capacity & complexity of each model
    Each model has a set of internal parameters that help the model to record the patterns taken from the data
    The capacity or expressiveness of a model as a matter of bias & vaiance
        if the mathematical function of a model is not complex or expressive enough to capture the complexity of the problem then we have a bias
        if the mathematical function of a model is too complex for the problem then we have a variance. the model will record more details & noise int he training data than needed
        and behave erratic.

    The process of learning elements of the training set that have no generalization value is commonly called overfitting
    The core purpose of a validation is to explicitly define a score or loss value that seperates the generalizable part of that value from that due to overfitting the training set char

    Plot Loss vs Iterations 
        you will notice that learning always happens ont he training set but this is not true on the validation data.
            learning more from the training data does not always mean learning to predict 

    Overfitting at various levels
        at the level of training data, when u use a model that is too complex for the problem
        at the level of validation set itself, when u tune ur model too much w.r.t specific validation set

Trying different splitting strategies
    The validation loss is based on the data sample that is not part of the training set
    it is an empirical measure that tell u how good ur model is at predicting
    correctly choosing the data sample u use for validation constitutes ur validation strategy

    The first choice to work with a holdout system; there is risk of not properly choosing a representative of the data / overfitting to you validation holdout
    The second choice is to use a probabilitic approach; 
            Cross validation ( simple random sampling, stratified sampling, time sampling, sampling by groups )
            leave one out ( LOO )
            bootstrap

        Note: since there is always some randomness in sampling; there is always a certain degree of error 
    With validation we want it to be evaluated on its ability to derive patterns & functions that work on unseen samples

    The Basic Train_test split
        80/20 ( hold out )
        hold out is used for evaluation for all the models u train
             when large sample; test data is similar to original distributiion but there is alyas a chance of non represetative sample
             when small sample; comparing the extracted hold out using adversarial validations can help u if u are evaluating correctly

        you can also use stratification , which ensures proportions of certain features are respected int he sample data

        Note: simple train test split is not enough for ensuring correct tracing on ur efforts
              in fact as you keep checking on this test set; you may drive ur choices to some kind of adaptation overfitting ( error picking noise as signals in training ).

        hence although computationally expensive, it is more suited to pick probabilistic evaluation

    Probabilistic evaluation methods - law of large numbers - sample mean close to original mean 
                The idea is you create a smaller set of your original data that is expected to have the same char
        K-fold cross validation ( pg 162 )
            can be used to compare models
            as well as when selecting hyperparameters for ur model
            idea -  5 folds ( each with 80/20 with 1 fold for validation ) - the final score is the avg score across all validation folds and the st dev tells u the uncertainity of the estimate

            Important aspect of K-fold is the estimates on the avg score os a model is trained on same quantity of the data; 
            if after the estimates u train on entire data the estimates are no longer valid ( it wont help u correctly estimating the generalizing power )

            when u reach k=n; you have the LOO but that far too few and not a good estimate of the expected performance on unseen data;
            the LOO metric represents more of the performance of the model on the data itself and not on the unseen data

            Choosing K
                smaller K, smaller each fold, less data more bias
                higher K, the more the data; training is good; validation score is corelated to training score and not the performance on unseen data

                The choice of K depends on the focus
                    1) if your performance is performance estimation, u need low bias; more data hence higher K ( 10 to 20 )
                    2) if your aim is hyperparam tuning, u need a mix of bias & variance. use medium fold. ( 5- 7 )
                    3) if your purpose is just apply variable selection & simplyfy ur dataset; u need low variance dataset; hence a lower no of folds will suffice ( 3- 5 )

                when the size of the data is quite large, u can safely stau on the lower side of K
                
        K-fold variations
            K fold can provide unsuitable splits
                1) when u have to preserve proportion of small classes.  imbalanced target
                2) when u have to preserve distribution of a numeric varibale both at target & features level ( e.g. regression problems )
                3) you case is non i.i.d ( independent identical distributions ) time serie forcasting
            the first 2 scenarios ; stratified K fold. you can obtain the same result with a numeric variable after having discretized by using pandas.cut or
            scikit learns KbinsDiscretizer

            scikitlearns multilearn package for multilabel classfifcation when u have to stratify based on multiple variables or overlapping labels
                http://scikit.ml/api/skmultilearn.model_selection.iterative_stratification.html#

            for regression problems
                use a discrete proxy for ur target instead of continuous target
                pandas cut function; sturges Rule
            
            An Alternate approach is to focus on the distributions of the feaures int he training and aim to reproduce them; cluster analysis
            predicted clusters as strata
                https://www.kaggle.com/code/lucamassaron/are-you-doing-cross-validation-the-best-way

            Time series
                   you cannot validate by random sampling becuase you will mix diff time frames & later timeframes could bear traces of previous ones ( auto correlation )
                you can use training & validation split based on time; your validation capabilities will be limited but it will be anchored to a specific time
                training => validation => test ( public & private )
                for a more complex approach use sklearn.model_selection.timeseriessplit ( page 167 )
                    option1 - growing training set & a moving validation set
                    option2 - fixed lookback; training & validation splits are moving over time
                fixed lookback helps to provide a fairer evaluation of time series as the training set size is same

                Timeseriessplit can be set to to keep a pre defined gap between ur training & test time; usefull when u are told that the test set is a certain time in future

        Nested Cross validation
        producing out of fold predictions
        subsampling
        the bootstrap

    Tuning your model validation system
    Using adversial validation
    Handling different distributions of training & test data
    Handling leakage











"""