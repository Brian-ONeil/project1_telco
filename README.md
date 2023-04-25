# Project 1 Telco

## Project Description

Telco is a communications company. Like most companies, one of the priorities is to keep customers and to prefer their services over that of competitors. The Telco Project focuses on customer's churn rate and how to predict when a customer is more likely to churn. The project will allow the company to intervene with alternative options based on the final findings.

## Project Goal

* Find potential drivers for customers churning
* Identify those customers at higher risk for churning
* Propose actionable options to business managers to retain at risk customers of churning.

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
        * Does tech_support affect churn?
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

* 
*  
*  
*  
*  

## Recommendations

*
*
*