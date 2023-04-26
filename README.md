# Project 1 Telco

## Project Description

Telco is a communications company. Like most companies, one of the priorities is to keep customers and to prefer their services over that of competitors. The Telco Project focuses on customer's churn rate and how to predict when a customer is more likely to churn. The project will allow the company to intervene with alternative options based on the final findings.

## Project Goals

* Find potential drivers for customers churning
* Identify those customers at higher risk for churning
* Propose actionable options to business managers to retain at risk customers of churning

## Initial Thoughts

There are some key indicators in the data that lead to churn and that those indicators will be evident by the conclusion of the project.

## The Plan

* Acquire the telco_churn dataset using SQL through a coded acquire .py file.

* Prepare the data using the following columns:
    * target: churn
    * features:
        * payment_type
        * internet_service_type
        * contract_type
        * tech_support
        * monthly_charges

* Explore dataset for indicators of churn
    * Answer the following questions:
        * Does payment_type affect churn?
        * Does internet_service_type affect churn?
        * Does contract_type affect churn?
        * Does whether or not a customer has tech_support affect churn?
        * Does monthly_charges affect churn?

* Develop a model
    * Using the selected data features develop appropriate predictive models
    * Evaluate the models in action using train and validate splits
    * Choose the most accurate model 
    * Evaluate the most accurate model using the final test data set

* Draw conclusions

## Data Dictionary

| **Feature**           | **Definition**                                                                       | **Units** |
|-----------------------|--------------------------------------------------------------------------------------|-----------|
| churn (target)        | indicates if a customer is still using services or discontinued services - yes or no |           |
| payment_type          | indicates the type of payment - electronic or by mail                                |           |
| internet_service_type | indicates the type of internet - DSL or fiber optic                                  |           |
| contract_type         | indicates the type of contract - one year or month-to-month                          |           |
| tech_support          | indicates if a customer uses tech_support - yes or no                                |           |
| monthly_charges       | indicates how much a customer pays per month for service                             | USD       |


## Steps to Reproduce

1) Clone the repo git@github.com:Brian-ONeil/project1_telco.git in terminal
2) Use personal env.py to connect to download SQL telco dataset
3) Run notebook

## Takeaways and Conclusions
* The statistical modeling showed strong relationships between the target churn and the features payment_type, internet_service_type, contract_type, tech_support, and monthly_charges.
* Electronic check payment type is 3 to 4 times higher than any other payment type that churns
* Fiber optic internet_service_type is about 3 times higher likely to churn than DSL and 2 times higher than DSL/none combined that churns
* Month-to-Month contract_type is more than 8 times higher likely to churn than both annual contract types combined
* Customers not having tech_support is approximately 4 times more likely to churn than customers who have tech_support
* Customers that churn have about a $15 higher mean than those who have not churned
* All features were worthy of modeling.
* The final Logistic Regression Model on the test data set made a small improvement over the baseline of just over 5%
* It is possible that adding or dropping more features could add to the small improvement
* The Logistics Regression Model maintained over a 77% accuracy with only a 0.4 differential in both train and validate. Although, a small difference I feel the Logistics Regression Model is the highest accuracy average with a good consistency. 
* In the end the Regression Model actually improved over the 77% in train and validate to a 78.6% accuracy in the test data. Based on the previous results, I feel 78.6% when compared to the 73.4% baseline is a small gain toward a viable prediction model.

## Recommendations
* Consider offering rebates for using automatic banking
* Consider a root cause of such a high churn for the flagship internet_service using fiber optic i.e. cost or maintenance downtime
* Consider better marketing the benefits of annual subscriptions
* Consider offering initial free trial service of tech support
* Consider rebates for higher tenure customers on monthly_charges