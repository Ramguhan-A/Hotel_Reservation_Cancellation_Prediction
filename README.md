# HOTEL RESERVATION CANCELLATION PREDICTION (MLOps)

## Objective:

The online hotel reservation channels have dramatically changed booking possibilities and customers’ behavior. A significant number of hotel reservations are called-off due to cancellations or no-shows. The typical reasons for cancellations include change of plans, scheduling conflicts, etc. This is often made easier by the option to do so free of charge or preferably at a low cost which is beneficial to hotel guests but it is a less desirable and possibly revenue-diminishing factor for hotels to deal with.

Can you predict if the customer is going to honor the reservation or cancel it ?

So to Solve this busines problem,
* I have Collected and preprocessed hotel booking data
* Explored multiple machine learning models
* Evaluated models using F1-score metric to balance precision & recall, since both false positives (predicting cancel when not) and false negatives (predicting not cancel when cancel) are costly.
* Identified the most important features influencing cancellation behavior using feature importance analysis.

### Final Model Selection

* Chosen Model: LightGBM (Light Gradient-Boosting Machine)
* Performance: Achieved an F1-score of 89% on the test dataset.

### Business Value:

* Hotels can use this predictive system to forecast cancellations in advance,

* Apply dynamic pricing,

* Offer targeted retention incentives, or

* Overbook strategically to minimize revenue loss.

## Tools & Technologies Used

Programming & Data Handling:

  * Python, Pandas, Numpy

Visualization & EDA:

  * Matplotlib, Seaborn, SHAP (feature explainability)

Machine Learning & Modeling:

  * Scikit-learn, XGBoost, LightGBM

Deployment & MLOps:

  * Streamlit (web app interface)

  * Docker (containerization)

  * Git & GitHub (version control, collaboration)

  * GitHub Actions (CI/CD automation)

  * MLflow (experiment tracking & model management)

  * Google Cloud Run (cloud deployment & scalability)

## Workflow i follwed for this project
<img width="2996" height="2010" alt="Frame 8 (3)" src="https://github.com/user-attachments/assets/97a0a725-c469-4341-8f5d-e7afe9afb706" />

## Dataset Description [Link](https://www.kaggle.com/datasets/ahsan81/hotel-reservations-classification-datasetO)

The dataset contains hotel booking records collected from an online reservation system. Each record represents a booking along with customer details, stay information, and the final booking outcome (canceled or Not canceled).

Key Characteristics:

  * Labeled dataset → outcome for each booking is known.
  
  * Mix of categorical & numerical features (e.g., room_type_reserved, market_segment_type vs. lead_time, avg_price_per_room).

  * Dynamic behavior captured (e.g., changing prices, lead time, special requests).

  * Customer history included (repeated_guest, no_of_previous_cancellations, etc.), which is valuable for predictive modeling.

  * Imbalanced classes → cancellations of the reservation is 28.7%
    
    <img width="401" height="411" alt="image" src="https://github.com/user-attachments/assets/ebc7c5b9-e062-4621-a8bb-35270eb0be2b" />

## Techinques used

#### Data Balancing

   * Applied Oversampling (SMOTE) to handle the class imbalance in the target variable (booking_status) on the training set.

#### Skewness Test

   * Applied log transform on highly skewed collumns

#### Feature Engineering & Encoding

   * Implemented Label Encoding to convert categorical features (e.g., room_type_reserved, market_segment_type, etc.) into numerical form suitable for machine learning algorithms.

#### Exploratory Data Analysis (EDA)

   * Conducted data visualization (using Matplotlib, Seaborn, SHAP) to uncover patterns and relationships between features such as avg_price_per_room, lead_time, and cancellation likelihood.

## Models Experiments
<img width="1621" height="654" alt="8d4db3c9-dc4e-4012-a665-96c7f8636b27" src="https://github.com/user-attachments/assets/cfd664b9-fa48-4bd8-9737-0018458bd94a" />
<img width="940" height="445" alt="image" src="https://github.com/user-attachments/assets/436037a8-e356-4303-a66e-7c8982cf8b60" />

## Feature importance

<img width="707" height="455" alt="c5249e42-3836-4094-8cec-f0624fbe9fc8" src="https://github.com/user-attachments/assets/45a93570-6d5b-47ee-b73f-83402df9720c" />

### 🔑 Top Influencing Features

1. avg_price_per_room (3751 importance)

    * Customers are highly sensitive to room price.

    * If the price is too high, customers may cancel or switch to cheaper alternatives.

    * If the price is low/discounted, they are more likely to stick with the booking.

    * Business implication: Competitive pricing and targeted discounts can reduce cancellations.

2. lead_time (3735 importance)

    * A long gap between booking and stay increases the chance of cancellation (plans change, emergencies arise, better deals appear).

    * Shorter lead times usually mean higher commitment from customers.

    * Business implication: Introduce flexible pricing or cancellation penalties for long lead-time bookings.

3. Seasonality and booking channel play secondary but still plays important roles.

## Final interface (Streamlit deployed on GCP)
<img width="1907" height="962" alt="hotel reverastion 1 on gcp" src="https://github.com/user-attachments/assets/b8d19432-dfa8-4e87-8fa9-c4f38a7252f5" />

<img width="1912" height="956" alt="hotel reservation 2 on gcp" src="https://github.com/user-attachments/assets/8a79d9e7-b209-496c-8b65-2b2a2c67657a" />

## Software metrics monitoring (GCP)
<img width="1583" height="668" alt="image" src="https://github.com/user-attachments/assets/b15b1595-0713-4b63-8abf-d645a7ced101" />



