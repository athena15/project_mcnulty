# Project McNulty Proposal

### Brenner Heintz

Proposal: Use data from Kaggle’s “Telco Customer Churn” dataset to model which customers will leave the company within the next month.

1. Background

2. 1. Why is this important?

   2. 1. Customer churn is a huge problem for every business - in fact, customer retention is nearly every business’s number one problem.
      2. Companies who can retain their clients retain the profits that those customers confer.
      3. In today’s world, cable companies are bleeding customers (especially young, tech-savvy customers) like never before. Any improvement in how well those companies can hold onto those customers may pay huge dividends in terms of the companie’s lifetime profits per user.

3. Data

4. 1. Kaggle Dataset - “Telco Customer Churn”

   2. 1. Available at <https://www.kaggle.com/blastchar/telco-customer-churn/home>
      2. Data contains records of over 7,000 customers, and whether those customers left within the last month. That column, “Churn”, is the dependent variable for the analysis.
      3. Features:

- Customer ID
- Gender (female, male)
- Whether the customer is a senior citizen or not
- Whether the customer has a partner or not
- \# of Dependents
- Business Tenure with the company
- Whether customer subscribes to:
- ​	Phone service
- Multiple Lines
- Internet Service
- Online security services offered by the company
- Cloud backup services offered by the company
- Device protection services
- Premium tech support
- Streaming TV service
- Streaming movie service
- Contract length with the customer
- Paperless Billing/”Snail Mail” Billing
- Method of Payment (electronic, check, credit card, etc.)
- Total monthly charges
- Total customer charges paid over customer lifetime

1. 1. Potential Additional Features

   2. 1. Tech savviness - based upon services subscribed to, whether or not each client owns their own modem, etc.
      2. Business users vs. home users - business users are more likely to have multiple lines, high bills, and fast internet. They are also more likely to stay with their current provider as switching ISP’s could be disruptive to their business.
      3. Household composition - could segment the market based upon the age of the primary subscriber, whether they have a partner, and whether they have dependents in the household.

2. Methodology & Approach:

3. 1. Hypothesis:

   2. 1. I hypothesize that older clients who pay primarily by electronic check and/or autodraft, who have dependents in the household, are the least likely to churn. The most likey to churn are young individuals with no partner, and no dependents, who appear to have tech savviness based upon their use of (or lack of use of) various services. For example, a young “tech savvy” person would likely have fast DSL or Fiber Optic Internet, no phone subscription, no device protection, and no need for tech support.

   3. Tools:

   4. 1. Python (Pandas, NumPy, Scikit-Learn, Statsmodels, requests, Matplotlib, Seaborn, Google Maps API)
      2. Jupyter Notebook

   5. Use logistic regression and/or Support Vector Machines, and/or any other classification techniques I deem necessary to up the accuracy of the model within reason and model extensibility.

4. Minimum Viable Product:

5. 1. A logistic regression model predicting the amount of customer churn, which is predictive enough to inform company policy and point to potential ways to retain customers.