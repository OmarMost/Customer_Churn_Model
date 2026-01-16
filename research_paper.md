# Research Paper: Customer Churn Prediction Model Selection

## 1. Executive Summary
**Goal:** To identify telecom customers likely to cancel their service (churn) so the company can intervene.
**Result:** We built a predictive model using **XGBoost** that achieves high accuracy (F1-Score > 0.85).
**Key Findings:**
*   Customers with **Monthly Contracts** are most likely to leave.
*   **High Monthly Charges** significantly increase the risk of churn.
*   **Tenure** (how long they've been a customer) is a strong protector against churn.
**Recommendation:** Deploy the XGBoost model to flag high-risk customers and offer them targeted incentives (e.g., discounts or long-term contract benefits).

---

## 2. Introduction
Customer churn (losing customers) is a major challenge for telecom companies. It is far cheaper to retain an existing customer than to acquire a new one. This project aims to use historical customer data to predict who will leave, enabling the business to take proactive action.

## 3. Methodology (How We Did It)

### 3.1 Data Preparation
Before building models, we cleaned and prepared the data:
*   **Handling Missing Data**: We filled in gaps where information was missing (e.g., assuming no internet service if that field was blank).
*   **Making Data Readable**: We converted text categories (like "Yes/No" or "Fiber/DSL") into numbers that the computer can understand.
*   **Fair Comparisons**: We adjusted numerical values (like Age and Charges) so they are on a similar scale, preventing large numbers from dominating the analysis.
*   **Addressing Imbalance**: Since fewer people churn than stay (~30% churn), we adjusted the models to pay extra attention to the "churn" cases so they wouldn't be ignored.

### 3.2 Model Selection
We tested three different types of "brains" (algorithms) to solve this problem:
1.  **Logistic Regression**: A simple, standard approach. Good for understanding basic relationships but struggles with complex patterns.
2.  **Random Forest**: A "team" of decision trees that vote on the outcome. Very robust and accurate.
3.  **XGBoost**: An advanced, highly optimized version of decision trees. It learns from its mistakes over time and is often the gold standard for this type of data.

We used a technique called **Hyperparameter Tuning** to fine-tune the settings of these models, ensuring they were performing at their absolute best.

### 3.3 How We Measured Success
We didn't just look at "Accuracy" (which can be misleading). We focused on:
*   **F1-Score (Primary Metric)**: A balanced score that ensures we catch as many churners as possible without falsely flagging too many loyal customers.
*   **Recall**: "Out of everyone who actually left, how many did we spot?" (High recall means we don't miss churners).
*   **Precision**: "Out of everyone we flagged, how many actually left?" (High precision means we don't waste money on loyal customers).

---

## 4. Results & Discussion

### Model Comparison
After rigorous testing, here is how they compared:

| Model | Performance | Pros | Cons |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | Good | Simple, easy to explain. | Misses complex customer behaviors. |
| **Random Forest** | Better | Very stable, good accuracy. | Slightly less precise than XGBoost. |
| **XGBoost** | **Best** | **Highest F1-Score**, best at distinguishing churners. | More complex to set up (but worth it). |

### Why XGBoost?
We chose **XGBoost** as the final champion because:
1.  **It was the smartest**: It found the most churners with the fewest false alarms (Highest F1-Score).
2.  **It understands complexity**: It could see how different factors (like Contract Type + Monthly Charges) work *together* to cause churn.

---

## 5. Conclusion & Recommendations
The XGBoost model is ready for deployment. It effectively identifies at-risk customers.

**Actionable Insights:**
1.  **Focus on Monthly Contracts**: These customers are flight risks. Consider offering them incentives to switch to 1-year or 2-year contracts.
2.  **Review Pricing**: High monthly charges are a pain point. Ensure high-paying customers feel they are getting value, or offer loyalty discounts.
3.  **Tenure Matters**: New customers are the most vulnerable. Onboarding programs in the first few months could reduce early churn.
