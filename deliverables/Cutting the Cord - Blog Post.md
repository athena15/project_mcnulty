# Cutting the Cord: Predicting Customer Churn for a Telecom Company



![img](https://cdn-images-1.medium.com/max/800/1*qjPKnMibxxX98BbQ8FWO2A.png)

You can find the [code in the project repository here](https://github.com/athena15/project_mcnulty), or view the [final presentation slides here](https://docs.google.com/presentation/d/11sF4lvK3YN3HboN2kcaCh0DWFBSYYQvWvHPWNumpTps/edit?usp=sharing).

### Why study customer churn?

Churn is one of the largest problems facing most businesses. [According to Harvard Business Review](https://hbr.org/2014/10/the-value-of-keeping-the-right-customers), **it costs between 5 times and 25 times as much to find a new customer than to retain an existing one**. In other words, your existing customers are worth their weight in gold!

Preventing customer churn is an important business function. It’s also one that has too often been approached with scattershot, back-of-the-envelope methods. By building a model to predict customer churn with machine learning algorithms, ideally we can nip the problem of unsatisfied customers in the bud — and keep the revenue flowing.

### Background



![img](https://cdn-images-1.medium.com/max/800/1*6Qv2w2J6EyQ-gNU6wv6B_g.png)

I used a dataset from Kaggle.com that included 7,033 unique customer records for a telecom company called Telco. Each entry had information about the customer, which included features such as:

> Services — which services the customer subscribed to (internet, phone, cable, etc.)

> Monthly bill total

> Tenure — How long they had been a customer

> Basic demographic info — whether they were elderly, had dependents, etc.

For the purposes of our study, the dependent variable was whether or not the customer had “**churned**” or not within the past month. In fact, a full **27% of our customers were labeled as having left the company within the last month.** With rates of attrition this high, it would only be a matter of months before the company lost most of its customers — if we didn’t intervene.



![img]()

Churn!

### Assumptions

For the purpose of our analysis, we made 2 assumptions:

1. **For each customer that left the company, it would cost Telco $500 to replace that customer.** Marketing, ads, campaigns, and outreach — the cost adds up.
2. **We could retain each customer who was likely to churn by investing $100 in them**. This could be through discounts, improving service (upping their internet speed, for example), or offering perks.

### Setting the Bar



![img](https://cdn-images-1.medium.com/max/600/1*pAJ2XWzHhAUfFfWxkVxPkA.jpeg)

Let’s make our model better.

In order to declare victory, we need to know what success looks like! Our primary measure of success would be how well our model performed against a default, dummy model. Think of this as the “status quo” option. Since we know that spending $100 to retain a customer will save us $500 in the long run — but we don’t know which customers are likely to churn — **our default, “dummy” model is simply spending $100 on allof our customers, to ensure that we capture those who churn.** This is the model we’ll look to beat.

### Methodology

For our model, we first looked at several different machine learning algorithms to see which ones to move forward with. Our first step was to **split our data into training and test sets** using train-test-split, which would allow us to cross-validate our results later. **We also stratified the train-test-split,** to ensure that the same proportion of our target variable was found in both our training and test sets.

```
# Stratify our train-test-split so that we have a balanced split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40, stratify=y)
```

We also used some **minority oversampling** in order to balance our dataset. Since only ~27% of the records in our database were marked as “churned”, feeding our data into our algorithm without oversampling would have led it to underclassify our target variable. We used **imblearn’s SMOTE to bring our minority class up to 50%** of our dataset.

```
sm = SMOTE(random_state=42, ratio=1.0)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train)
```

Now that our data was balanced, we then performed **yet another train-test-split** — this time just on our training data. The reason for doing it this way is so that we don’t violate the cardinal rule of cross-validation — basing decisions off of the results that your test data provide.

```
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_train_res, y_train_res, test_size=0.33, random_state=20)
```

After all this glorious data munging, we plotted a ROC curve to compare how each algorithm did on identifying true positives (sensitivity) vs. false positives (specificity).



![img](https://cdn-images-1.medium.com/max/600/0*q7JiIJkMQAPj3xIK)

The ROC Curve, in all its glory.

Based upon this plot, we determined that we would move forward with 4 main classification models: logistic regression, gradient boosting, random forest, and AdaBoost. We set out to tune up these models to get the most we could out of them.

For our analysis, **recall was our target metric**. We care the most about capturing as many *true positives* (people who are likely to churn) with our model, and we’re less concerned that we may sweep in some *false negatives* (people who did not churn) along with them.

Knowing this, we then used **Sci-Kit Learn’s GridSearchCV** function, which allows us to tune our model. We set recall as the scoring metric to optimize on, and then used combinations of different hyperparameters to find the model with the best fit. Our goal was to squeeze out every last ounce of recall we could out of our models, and nothing less!

```
pipe = Pipeline([('scaler', StandardScaler()),
                 ('clf', RandomForestClassifier())])

param_grid = [{'clf__max_depth': [4, 5, 6],
               'clf__max_features': [5, 10, 15, 20],
               'clf__n_estimators': [50, 100, 150]}]

gs = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1, scoring='recall')
gs.fit(X_train_res, y_train_res)
```

GridSearchCV also includes a handy cross-validation function (that’s what the CV stands for!), so we were performing **Stratified K-Folds cross-validation**on every model’s pass through new parameters. Needless to say, we were being quite thorough in our attempts to avoid overfitting our models.

Finally, a big part of our analysis had to do with creating a **“net dollars saved” function** that would determine how much money we spent on retaining customers, versus how much we saved by not having to replace them. This, along with recall, made up the decision criteria upon which we judged whether we had a successful model.

### Results



![img](https://cdn-images-1.medium.com/max/800/1*M8Vv0nxGxhcHgEm7hSNCAw.jpeg)

All of our models performed well.

After tuning our models, it came down to 3 models that were neck-and-neck. **Our final step was to adjust the probability threshold for each model (between i = .01 and i = 1).** This allowed us to optimize our “net dollars saved” function. Essentially, for each probability threshold *i*, we were asking our model to predict whether the customer would churn or not — even very low and very high i-values. Of course, as *i* approached 0, our models would essentially predict that *everyone* would churn — and conversely, as it approached 0, that no one would churn. By plotting this function, we were able to see the exact probability that optimized our “net dollars saved” function.



![img](https://cdn-images-1.medium.com/max/600/1*7-R61YZpN_H5CPQJOaegUw.jpeg)

Making money by keeping customers.

In the end, **the Logistic Regression model won out.** It showed excellent recall at 81%, and **maximized our “net dollars saved” function. Overall, our model saved us a total $272,200, beating the “status quo” model handily. That model only saved a (paltry) $162,000.**

Thanks for reading! If you enjoyed the post, [find me on LinkedIn](https://www.linkedin.com/in/brennerheintz/), give me a clap on Medium, or [email me here](mailto:brenner.heintz@gmail.com). Onward and upward, friends!