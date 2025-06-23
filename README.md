# Product Return Predictor
**Disclaimer: This is not an official Google product.**

[Overview](#problem-statement) •
[Objective](#objective) •
[Solution Overview](#solution-overview) •
[Related Work](#related-work) •
[Infrastructure Design on Google Cloud Platform](#infrastructure-design-on-google-cloud-platform) •
[Detailed Design](#detailed-design) •
[Data Requirements](#data-requirements) •
[Feature Engineering](#feature-engineering) •
[Model Training and Refresh](#model-training-and-refresh) •
[Prediction and Activation](#prediction-and-activation)

## Problem Statement
We want to help retailers reduce the rate of customers returning their products,
to increase sustainability and profit.
A product can be returned for various reasons, including quality and size
issues. There are also challenges such as “wardrobing” which is where customers
purchase an item with the intention of returning it once it has been used for a
specific purpose.

## Objective
To address the return issue for our retail advertisers, the product return
solution leverages machine learning to predict return probability at transaction
level. The solution aims to predict product refund amounts to adjust the
conversion value for a given transaction/order to make sure our advertisers are
spending the “right” amount of investment in targeting the “right” customers.

## Solution Overview
The product return predictor solution is an end-to-end model training and
prediction pipeline, leveraging GA4 (Google Analytics) data export on BigQuery
and deployed on Google Cloud Platform (GCP).

**The pipeline includes two parts: model training and prediction.**

The **model training** part provides functions leveraging SQL and Python code
for data wrangling, feature engineering, model training and model evaluation.
The **prediction** part provides functions for feature engineering and
prediction.

**Below are the main features of the solution:**

- **Privacy-safe solution leveraging 1st Party data:** Our product return
predictor model leverages 1st Party - mainly GA4 data export on BigQuery (and
other data sources/tables available on BigQuery if needed).

- **Machine Learning model for prediction**: To predict product refund value
for each transaction, we are using a machine learning model on GCP (BigQueryML)
to predict the future refund value of a transaction before the end of the return
policy window (before the transaction is no longer eligible for return).

- **Full automation for model training & prediction process**: The whole system
is orchestrated by a single cloud function that listens to entries in our logs
table and effectively coordinates a status machine, triggering the appropriate
query in the sequence.

- **Simple deployment that can scale across brands and markets**: The entire
solution is designed and developed to enable our advertisers to deploy their
product return model on GCP in a scalable and efficient manner. This means
advertisers/users are expected to install our solution and provide a few input
& adjustments to fully leverage the solution for their own business.

## Related work
The predictive modeling framework is very similar to those used by
[crystalvalue] (https://github.com/google/crystalvalue) used for predicting
customer lifetime value and purchase propensity.

**The biggest difference between crystalValue and Product Return Predictor is:**

- crystalvalue is designed and used to predict customer lifetime value (LTV)
while Product Return Predictor is used to predict return value.
- crystalvalue uses AutoML while Product Return Predictor uses BigQuery ML for
model training.
- Compared with crystalvalue which uses simple features, Product Return
Predictor provides SQL queries that create more comprehensive features for the
predictive model.

## Solution Workflow Details
Two components of the overall process of product return predictor on Google
Cloud Platform (GCP):

- **[Model Training]** Use historical data to train predictive product return
models using BQML & Vertex AI workbench. The model training process is done
using a user-managed notebook instance on VertexAI workbench. The process is
done once every 3 - 12 months depending on how often the user is planning to
refresh their predictive model.

  **The process entails the following steps:**

  - Data Wrangling & Feature Engineering on training dataset
  - Model Training
  - Model Performance Evaluation using training and validation datasets
  - Trained model will be saved on BigQuery

- **[Model Prediction]** Use saved models trained during the model training
phase for prediction. The predicted transaction level refund value is produced
during this process and saved in the BigQuery table for activation on a daily
basis. Cloud function is used to complete the following steps. Cloud Scheduler
is used to schedule daily predictions.

  **The process entails the following steps:**
  - Data Wrangling & Feature Engineering on prediction dataset (using cloud
  function)
  - Model Prediction (using ML.predict on BigQuery ML)

## Infrastructure Design on Google Cloud Platform
<img align="center" width="600" src="./images/product_return_predictor_architecture_design.png" alt="infrastructure_design" /><br>

## Detailed design
The main components of the projects are model training pipeline and prediction
pipeline:

- **[Model Training: every 3-12 months Depending how often the user wants
refresh their predictive model]**
  - **Data Engineering**: Creating model training dataset (stored on BigQuery)
  - **Data Sanity check + Feature Selection**: Data cleanup and selection of
  features via Python Cloud Function (stored on BigQuery).
  - **Model Training Validation and Hyper Parameter tuning**: Use BQML Train and
  validate models on the validation set (20%). Determine which model is the
  best. (Keep log of best performing model for each run of the Model Training
  pipeline).
  - **Model Evaluation**: Predict 10% of latest transactions and produce
  estimated output. Report on Cloud Storage that users can do their own check.
  - **3 target variables considered:**

      1. **refund value** = amount returned to the customer
      2. **refund value proportion** = refund value / transaction value (%
      value)
      3. **refund value flag + refund value**: Create binary classification and
      for transactions with refund value, train a regression model to predict
      refund value

- **[Model Prediction: Daily]**
  - **Data Engineering**: Creating model prediction dataset (stored on
  BigQuery).
  - **Prediction Data Transformation**: Retrieve best performing model from Logs
  table and which are the features being used.
  - **Prediction**: Use Best Performing model to produce prediction.
  - **Data Activation**: Upload to Google Ads the data from BigQuery for offline
  conversion value export ([check OCA - Offline Conversions Adjustment](https://developers.google.com/google-ads/api/docs/conversions/upload-adjustments#python)).

## Data Requirements
**The solution uses GA4 data.**

**Below are the components of the main sources used for the solution from GA4:**

- Transaction id, item id, user/customer id level transaction date & value,
return date & value.
- Item metadata: Item id level product information including pricing, material,
color, brand, category, etc.
- Customer/user metadata: device type, country, gender, etc.
- Transaction id, session id level web activity data (e.g. sessions, time spent
on site, pageviews, etc.)

## Feature Engineering

**Below are the features for the first-time users**:

| Feature Name                      | Feature Type                                      | Feature Description                                                                                                                                              |
| :-------------------------------- | :------------------------------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `transaction_id`                  | Primary Key                                       | Unique identifier for each transaction.                                                                                                                  |
| `transaction_date`                | Primary Key                                       | Date and time when the transaction occurred.                                                                                                             |
| `refund_value`                    | Current Transaction Info (Shopping Cart Info)     | The monetary amount of any refund associated with the transaction.                                                                                       |
| `refund_flag`                     | Current Transaction Info (Shopping Cart Info)     | Indicates if the transaction involved a refund (True/False).                                                                                             |
| `refund_proportion`               | Current Transaction Info (Shopping Cart Info)     | The proportion of the original transaction value that was refunded.                                                                                      |
| `transaction_value`               | Current Transaction Info (Shopping Cart Info)     | The total monetary value of the transaction.                                                                                                             |
| `shipping_value`                  | Current Transaction Info (Shopping Cart Info)     | The monetary value of shipping costs for the transaction.                                                                                                |
| `item_count`                      | Current Transaction Info (Shopping Cart Info)     | The total number of distinct items purchased in the transaction.                                                                                         |
| `unique_item_id_count`            | Current Transaction Info (Shopping Cart Info)     | The count of unique item IDs in the transaction.                                                                                                         |
| `unique_item_name_count`          | Current Transaction Info (Shopping Cart Info)     | The count of unique item names in the transaction.                                                                                                       |
| `unique_item_category_count`      | Current Transaction Info (Shopping Cart Info)     | The count of unique item categories in the transaction.                                                                                                  |
| `unique_item_brand_count`         | Current Transaction Info (Shopping Cart Info)     | The count of unique item brands in the transaction.                                                                                                      |
| `min_price`                       | Current Transaction Info (Shopping Cart Info)     | The minimum price of an item within the transaction.                                                                                                     |
| `max_price`                       | Current Transaction Info (Shopping Cart Info)     | The maximum price of an item within the transaction.                                                                                                     |
| `avg_price`                       | Current Transaction Info (Shopping Cart Info)     | The average price of items within the transaction.                                                                                                       |
| `std_price`                       | Current Transaction Info (Shopping Cart Info)     | The standard deviation of item prices within the transaction.                                                                                            |
| `min_item_revenue`                | Current Transaction Info (Shopping Cart Info)     | The minimum revenue generated by a single item in the transaction.                                                                                       |
| `max_item_revenue`                | Current Transaction Info (Shopping Cart Info)     | The maximum revenue generated by a single item in the transaction.                                                                                       |
| `avg_item_revenue`                | Current Transaction Info (Shopping Cart Info)     | The average revenue generated by items in the transaction.                                                                                               |
| `std_item_revenue`                | Current Transaction Info (Shopping Cart Info)     | The standard deviation of item revenues within the transaction.                                                                                          |
| `min_item_quantity`               | Current Transaction Info (Shopping Cart Info)     | The minimum quantity of a single item purchased in the transaction.                                                                                      |
| `max_item_quantity`               | Current Transaction Info (Shopping Cart Info)     | The maximum quantity of a single item purchased in the transaction.                                                                                      |
| `avg_item_quantity`               | Current Transaction Info (Shopping Cart Info)     | The average quantity of items purchased within the transaction.                                                                                          |
| `std_item_quantity`               | Current Transaction Info (Shopping Cart Info)     | The standard deviation of item quantities within the transaction.                                                                                        |
| `discounted_product_count`        | Current Transaction Info (Shopping Cart Info)     | The number of products in the transaction that were discounted.                                                                                          |
| `max_same_item_item_quantity`     | Current Transaction Info (Shopping Cart Info)     | Maximum quantity of a specific item (same item ID) within the transaction.                                                                               |
| `min_same_item_item_quantity`     | Current Transaction Info (Shopping Cart Info)     | Minimum quantity of a specific item (same item ID) within the transaction.                                                                               |
| `avg_same_item_item_quantity`     | Current Transaction Info (Shopping Cart Info)     | Average quantity of a specific item (same item ID) within the transaction.                                                                               |
| `same_item_in_transaction_flag`   | Current Transaction Info (Shopping Cart Info)     | Indicates if multiple instances of the same item were in the transaction.                                                                                |
| `max_same_item_variant_quantity`  | Current Transaction Info (Shopping Cart Info)     | Maximum quantity of a specific item variant within the transaction.                                                                                      |
| `min_same_item_variant_quantity`  | Current Transaction Info (Shopping Cart Info)     | Minimum quantity of a specific item variant within the transaction.                                                                                      |
| `avg_same_item_variant_quantity`  | Current Transaction Info (Shopping Cart Info)     | Average quantity of a specific item variant within the transaction.                                                                                      |
| `same_item_variant_in_transaction_flag` | Current Transaction Info (Shopping Cart Info) | Indicates if multiple instances of the same item variant were in the transaction.                                                                        |
| `calculated_past_product_refund_rate` | Past Refund Stats                         | Historical refund rate for the product involved in the transaction. Calculation used: SAFE_DIVIDE((SUM(same item past refund count) + AVG(overall past refund rate) * 0.2), (SUM(same item past transaction count) + 0.2)) |
| `calculated_past_product_refund_amt_proportion` | Past Refund Stats                 | Historical proportion of refund amount for the product. Calculation used: SAFE_DIVIDE((SUM(same item past refund value) + AVG(overall past refund value proportion) * 0.2), (SUM(same item past transaction value) + 0.2)) |
| `calculated_past_brand_refund_amt_proportion` | Past Refund Stats                 | Historical proportion of refund amount for the brand of the product. Calculation used: SAFE_DIVIDE((SUM(same brand past refund amount) + AVG(overall past refund proportion) * 0.2), (SUM(same brand past transaction amount) + 0.2)) |
| `calculated_past_brand_refund_rate` | Past Refund Stats                         | Historical refund rate for the brand of the product. Calculation used: SAFE_DIVIDE((SUM(same brand past refund count) + AVG(overall past refund rate) * 0.2), (SUM(same brand past transaction count) + 0.2)) |
| `calculated_past_category_refund_amt_proportion` | Past Refund Stats                 | Historical proportion of refund amount for the product's category. Calculation used: SAFE_DIVIDE((SUM(same brand past refund amount) + AVG(overall past refund proportion) * 0.2), (SUM(same brand past transaction amount) + 0.2)) |
| `calculated_past_category_refund_rate` | Past Refund Stats                       | Historical refund rate for the product's category. Calculation used: SAFE_DIVIDE((SUM(same category past refund count) + AVG(overall past refund rate) * 0.2), (SUM(same category past transaction count) + 0.2)) |
| `country`                         | Current Web Session Info                  | The country from which the transaction or event originated.                                                                                              |
| `city`                            | Current Web Session Info                  | The city from which the transaction or event originated.                                                                                                 |
| `device_category`                 | Current Web Session Info                  | The type of device used (e.g., mobile, desktop, tablet).                                                                                                 |
| `operating_system`                | Current Web Session Info                  | The operating system of the device used.                                                                                                                 |
| `languages`                       | Current Web Session Info                  | The language setting of the user's device or browser.                                                                                                    |
| `session_length_in_seconds`       | Current Web Session Info                  | The duration of the user's session in seconds.                                                                                                           |
| `website_engaged_sessions`        | Current Web Session Info                  | The number of engaged sessions on the website.                                                                                                           |
| `web_traffic_engagement_time_seconds` | Current Web Session Info              | Total time (in seconds) the user spent engaged with web content.                                                                                         |
| `web_traffic_campaign`            | Current Web Session Info                  | The marketing campaign associated with the web traffic.                                                                                                  |
| `web_traffic_medium`              | Current Web Session Info                  | The medium through which the web traffic arrived (e.g., organic, CPC).                                                                                   |
| `web_traffic_source`              | Current Web Session Info                  | The source of the web traffic (e.g., google, direct).                                                                                                    |
| `web_traffic_channel_grouping_user` | Current Web Session Info              | The default channel grouping for the user's traffic.                                                                                                     |
| `event_length_by_event_name`      | Current Web Session Info                  | The length/duration of a specific event type. For creating fields with this naming convention, we list out all the unique event names and create a pivot table to get the event length for a particular event name. |
| `event_count_by_event_name`       | Current Web Session Info                  | The count of a specific event type. For creating fields with this naming convention, we list out all the unique event names and create a pivot table to get the event count for a particular event name. |
| `past_3_days_transaction_value`   | Recent historical refund & transaction stats (used for capturing seasonality) | Sum of transaction values in the past 3 days for the user/entity.                                                                                        |
| `past_3_days_refund_value`        | Recent historical refund & transaction stats (used for capturing seasonality) | Sum of refund values in the past 3 days for the user/entity.                                                                                             |
| `past_3_days_refund_proportion`   | Recent historical refund & transaction stats (used for capturing seasonality) | Proportion of refund value to transaction value in the past 3 days.                                                                                      |
| `past_7_days_transaction_value`   | Recent historical refund & transaction stats (used for capturing seasonality) | Sum of transaction values in the past 7 days for the user/entity.                                                                                        |
| `past_7_days_refund_value`        | Recent historical refund & transaction stats (used for capturing seasonality) | Sum of refund values in the past 7 days for the user/entity.                                                                                             |
| `past_7_days_refund_proportion`   | Recent historical refund & transaction stats (used for capturing seasonality) | Proportion of refund value to transaction value in the past 7 days.                                                                                      |
| `past_30_days_transaction_value`  | Recent historical refund & transaction stats (used for capturing seasonality) | Sum of transaction values in the past 30 days for the user/entity.                                                                                       |
| `past_30_days_refund_value`       | Recent historical refund & transaction stats (used for capturing seasonality) | Sum of refund values in the past 30 days for the user/entity.                                                                                            |
| `past_30_days_refund_proportion`  | Recent historical refund & transaction stats (used for capturing seasonality) | Proportion of refund value to transaction value in the past 30 days.                                                                                     |

**Below are the features for the existing users (non first-time purchasers)**:

| Feature Name                                  | Feature Type                                      | Feature Description                                                                                                                                              |
| :-------------------------------------------- | :------------------------------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `transaction_id`                              | Primary Key                                       | Unique identifier for each transaction.                                                                                                                  |
| `transaction_date`                            | Primary Key                                       | Date and time when the transaction occurred.                                                                                                             |
| `refund_value`                                | Current Transaction Info (Shopping Cart Info)     | The monetary amount of any refund associated with the transaction.                                                                                       |
| `refund_flag`                                 | Current Transaction Info (Shopping Cart Info)     | Indicates if the transaction involved a refund (True/False).                                                                                             |
| `refund_proportion`                           | Current Transaction Info (Shopping Cart Info)     | The proportion of the original transaction value that was refunded.                                                                                      |
| `transaction_value`                           | Current Transaction Info (Shopping Cart Info)     | The total monetary value of the transaction.                                                                                                             |
| `shipping_value`                              | Current Transaction Info (Shopping Cart Info)     | The monetary value of shipping costs for the transaction.                                                                                                |
| `item_count`                                  | Current Transaction Info (Shopping Cart Info)     | The total number of distinct items purchased in the transaction.                                                                                         |
| `unique_item_id_count`                        | Current Transaction Info (Shopping Cart Info)     | The count of unique item IDs in the transaction.                                                                                                         |
| `unique_item_name_count`                      | Current Transaction Info (Shopping Cart Info)     | The count of unique item names in the transaction.                                                                                                       |
| `unique_item_category_count`                  | Current Transaction Info (Shopping Cart Info)     | The count of unique item categories in the transaction.                                                                                                  |
| `unique_item_brand_count`                     | Current Transaction Info (Shopping Cart Info)     | The count of unique item brands in the transaction.                                                                                                      |
| `min_price`                                   | Current Transaction Info (Shopping Cart Info)     | The minimum price of an item within the transaction.                                                                                                     |
| `max_price`                                   | Current Transaction Info (Shopping Cart Info)     | The maximum price of an item within the transaction.                                                                                                     |
| `avg_price`                                   | Current Transaction Info (Shopping Cart Info)     | The average price of items within the transaction.                                                                                                       |
| `std_price`                                   | Current Transaction Info (Shopping Cart Info)     | The standard deviation of item prices within the transaction.                                                                                            |
| `min_item_revenue`                            | Current Transaction Info (Shopping Cart Info)     | The minimum revenue generated by a single item in the transaction.                                                                                       |
| `max_item_revenue`                            | Current Transaction Info (Shopping Cart Info)     | The maximum revenue generated by a single item in the transaction.                                                                                       |
| `avg_item_revenue`                            | Current Transaction Info (Shopping Cart Info)     | The average revenue generated by items in the transaction.                                                                                               |
| `std_item_revenue`                            | Current Transaction Info (Shopping Cart Info)     | The standard deviation of item revenues within the transaction.                                                                                          |
| `min_item_quantity`                           | Current Transaction Info (Shopping Cart Info)     | The minimum quantity of a single item purchased in the transaction.                                                                                      |
| `max_item_quantity`                           | Current Transaction Info (Shopping Cart Info)     | The maximum quantity of a single item purchased in the transaction.                                                                                      |
| `avg_item_quantity`                           | Current Transaction Info (Shopping Cart Info)     | The average quantity of items purchased within the transaction.                                                                                          |
| `std_item_quantity`                           | Current Transaction Info (Shopping Cart Info)     | The standard deviation of item quantities within the transaction.                                                                                        |
| `discounted_product_count`                    | Current Transaction Info (Shopping Cart Info)     | The number of products in the transaction that were discounted.                                                                                          |
| `max_same_item_item_quantity`                 | Current Transaction Info (Shopping Cart Info)     | Maximum quantity of a specific item (same item ID) within the transaction.                                                                               |
| `min_same_item_item_quantity`                 | Current Transaction Info (Shopping Cart Info)     | Minimum quantity of a specific item (same item ID) within the transaction.                                                                               |
| `avg_same_item_item_quantity`                 | Current Transaction Info (Shopping Cart Info)     | Average quantity of a specific item (same item ID) within the transaction.                                                                               |
| `same_item_in_transaction_flag`               | Current Transaction Info (Shopping Cart Info)     | Indicates if multiple instances of the same item were in the transaction.                                                                                |
| `max_same_item_variant_quantity`              | Current Transaction Info (Shopping Cart Info)     | Maximum quantity of a specific item variant within the transaction.                                                                                      |
| `min_same_item_variant_quantity`              | Current Transaction Info (Shopping Cart Info)     | Minimum quantity of a specific item variant within the transaction.                                                                                      |
| `avg_same_item_variant_quantity`              | Current Transaction Info (Shopping Cart Info)     | Average quantity of a specific item variant within the transaction.                                                                                      |
| `same_item_variant_in_transaction_flag`       | Current Transaction Info (Shopping Cart Info)     | Indicates if multiple instances of the same item variant were in the transaction.                                                                        |
| `calculated_past_product_refund_rate`         | Past Refund Stats                                 | Historical refund rate for the product involved in the transaction. Calculation used: SAFE_DIVIDE((SUM(same item past refund count) + AVG(overall past refund rate) * 0.2), (SUM(same item past transaction count) + 0.2)) |
| `calculated_past_product_refund_amt_proportion` | Past Refund Stats                             | Historical proportion of refund amount for the product. Calculation used: SAFE_DIVIDE((SUM(same item past refund value) + AVG(overall past refund value proportion) * 0.2), (SUM(same item past transaction value) + 0.2)) |
| `past_user_purchase_revenue`                  | Past Refund Stats                                 | Past user level total purchase revenue (this can be either user id or user pseudo id level)                                                              |
| `past_user_refund_value`                      | Past Refund Stats                                 | Past user level total refund value                                                                                                                       |
| `past_user_refund_count`                      | Past Refund Stats                                 | Past user level total refund count                                                                                                                       |
| `past_user_transaction_count`                 | Past Refund Stats                                 | Past user level total transaction count                                                                                                                  |
| `calculated_past_user_refund_rate`            | Past Refund Stats                                 | Calculation used: SAFE_DIVIDE((SUM(same user past refund count) + AVG(overall past refund rate) * 0.2), (SUM(same user past transaction count) + 0.2))    |
| `calculated_past_user_refund_amt_proportion`  | Past Refund Stats                                 | Calculation used: SAFE_DIVIDE((SUM(same user past refund amount) + AVG(overall past refund proportion) * 0.2), (SUM(same user past transaction amount) + 0.2)) |
| `past_user_same_item_refund_amt`              | Past Refund Stats                                 | Past user item level refund amount                                                                                                                       |
| `past_user_same_item_refund_count`            | Past Refund Stats                                 | Past user item level refund count                                                                                                                        |
| `past_user_same_item_transaction_count`       | Past Refund Stats                                 | Past user item level transaction count                                                                                                                   |
| `calculated_past_user_item_refund_rate`       | Past Refund Stats                                 | Calculation used: SAFE_DIVIDE((SUM(same item & user past refund count) + AVG(overall past refund rate) * 0.2), (SUM(same user & item past transaction count) + 0.2)) |
| `calculated_past_user_item_refund_amt_proportion` | Past Refund Stats                             | Calculation used: SAFE_DIVIDE((SUM(same user & item past refund count) + AVG(overall past refund rate) * 0.2), (SUM(same user & item past transaction count) + 0.2)) |
| `calculated_past_brand_refund_amt_proportion` | Past Refund Stats                                 | Historical proportion of refund amount for the brand of the product. Calculation used: SAFE_DIVIDE((SUM(same brand past refund amount) + AVG(overall past refund proportion) * 0.2), (SUM(same brand past transaction amount) + 0.2)) |
| `calculated_past_brand_refund_rate`           | Past Refund Stats                                 | Historical refund rate for the brand of the product. Calculation used: SAFE_DIVIDE((SUM(same brand past refund count) + AVG(overall past refund rate) * 0.2), (SUM(same brand past transaction count) + 0.2)) |
| `calculated_past_category_refund_amt_proportion` | Past Refund Stats                            | Historical proportion of refund amount for the product's category. Calculation used: SAFE_DIVIDE((SUM(same brand past refund amount) + AVG(overall past refund proportion) * 0.2), (SUM(same brand past transaction amount) + 0.2)) |
| `calculated_past_category_refund_rate`        | Past Refund Stats                                 | Historical refund rate for the product's category. Calculation used: SAFE_DIVIDE((SUM(same category past refund count) + AVG(overall past refund rate) * 0.2), (SUM(same category past transaction count) + 0.2)) |
| `country`                                     | Current Web Session Info                          | The country from which the transaction or event originated.                                                                                              |
| `city`                                        | Current Web Session Info                          | The city from which the transaction or event originated.                                                                                                 |
| `device_category`                             | Current Web Session Info                          | The type of device used (e.g., mobile, desktop, tablet).                                                                                                 |
| `operating_system`                            | Current Web Session Info                          | The operating system of the device used.                                                                                                                 |
| `languages`                                   | Current Web Session Info                          | The language setting of the user's device or browser.                                                                                                    |
| `session_length_in_seconds`                   | Current Web Session Info                          | The duration of the user's session in seconds.                                                                                                           |
| `website_engaged_sessions`                    | Current Web Session Info                          | The number of engaged sessions on the website.                                                                                                           |
| `web_traffic_engagement_time_seconds`         | Current Web Session Info                          | Total time (in seconds) the user spent engaged with web content.                                                                                         |
| `web_traffic_campaign`                        | Current Web Session Info                          | The marketing campaign associated with the web traffic.                                                                                                  |
| `web_traffic_medium`                          | Current Web Session Info                          | The medium through which the web traffic arrived (e.g., organic, CPC).                                                                                   |
| `web_traffic_source`                          | Current Web Session Info                          | The source of the web traffic (e.g., google, direct).                                                                                                    |
| `web_traffic_channel_grouping_user`           | Current Web Session Info                          | The default channel grouping for the user's traffic.                                                                                                     |
| `event_length_by_event_name`                  | Current Web Session Info                          | The length/duration of a specific event type. For creating fields with this naming convention, we list out all the unique event names and create a pivot table to get the event length for a particular event name. |
| `event_count_by_event_name`                   | Current Web Session Info                          | The count of a specific event type. For creating fields with this naming convention, we list out all the unique event names and create a pivot table to get the event count for a particular event name. |
| `past_3_days_transaction_value`               | Recent historical refund & transaction stats (used for capturing seasonality) | Sum of transaction values in the past 3 days for the user/entity.                                                                                        |
| `past_3_days_refund_value`                    | Recent historical refund & transaction stats (used for capturing seasonality) | Sum of refund values in the past 3 days for the user/entity.                                                                                             |
| `past_3_days_refund_proportion`               | Recent historical refund & transaction stats (used for capturing seasonality) | Proportion of refund value to transaction value in the past 3 days.                                                                                      |
| `past_7_days_transaction_value`               | Recent historical refund & transaction stats (used for capturing seasonality) | Sum of transaction values in the past 7 days for the user/entity.                                                                                        |
| `past_7_days_refund_value`                    | Recent historical refund & transaction stats (used for capturing seasonality) | Sum of refund values in the past 7 days for the user/entity.                                                                                             |
| `past_7_days_refund_proportion`               | Recent historical refund & transaction stats (used for capturing seasonality) | Proportion of refund value to transaction value in the past 7 days.                                                                                      |
| `past_30_days_transaction_value`              | Recent historical refund & transaction stats (used for capturing seasonality) | Sum of transaction values in the past 30 days for the user/entity.                                                                                       |
| `past_30_days_refund_value`                   | Recent historical refund & transaction stats (used for capturing seasonality) | Sum of refund values in the past 30 days for the user/entity.                                                                                            |
| `past_30_days_refund_proportion`              | Recent historical refund & transaction stats (used for capturing seasonality) | Proportion of refund value to transaction value in the past 30 days.                                                                                     |

## Model Training and Refresh
The solution is designed to build two sets of predictive model:
One for first time purchase where there are fewer features to train on. One for
existing customers where there are more features (past user behaviors) to train
on.

For each set of the models, currently we offer users 3 different variations
(options).The user can try out different variations to decide on which model
type to use.
Here are the options:

- **2 step approach**: Binary Classification Model to predict Refund Flag and
Regression model to predict refund value on transactions that have had
refunds.
- **1 step approach**: Regression model to predict refund value or refund
proportion (refund value/transaction value).

**How often should we refresh the model?**
This depends on how often users want to retrain the model. We recommend
monitoring prediction performance overtime when there’s a big shift on model
performance, then the users should consider refreshing the model.

## Prediction and Activation
The recommended activation strategy is to activate on smart bidding by adjusting
the offline conversation value:

**new offline conversion value = current transaction conversion value -
predicted refund amount**
