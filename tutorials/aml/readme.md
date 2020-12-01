# Azure Machine Learning tutorials for Customer Insights


## Churn Analysis

Churn analysis applies to different business areas. In this example, we’re going to look at service churn, specifically in the context of hotel services as described above. It provides a working example of an end–to–end model pipeline that can be used as a starting point for any other type of churn model.

### Definition of Churn

The definition of churn can differ based on the scenario. In this example, a guest who hasn’t visited the hotel in the past year should be labeled as churned.  

The experiment template can be imported from the gallery. First, ensure that you import the data for **Hotel Stay Activity**, **Customer data**, and **Service Usage Data** from Azure Blob storage.

   ![Import data for churn model](media/import-data-azure-blob-storage.png)

### Featurization

Based on the definition of churn, we first identify the raw features that will influence the label. Then, we process these raw features into numerical features that can be used with machine learning models. Data integration happens in Customer Insights so we can join these tables by using the *Customer ID*.

   ![Join imported data](media/join-imported-data.png)

The featurization for building the model for churn analysis can be a little tricky. The data is a function of time with new hotel activity recorded on daily basis. During featurization, we want to generate static features from the dynamic data. In this case, we generate multiple features from hotel activity with a sliding window of one year. We also expand the categorical features like room type or booking type into separate features using one-hot encoding.  

Final list of features:

| **Number**  | **Original_Column**          | **Derived Features**                                                                                                                      |
|-------------|------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| 1           | Room Type                    | RoomTypeLargeCount, RoomTypeSmallCount                                                                                                    |
| 2           | Booking Type                 | BookingTypeOnlineCount, BookingTypePhoneCallCount                                                                                         |
| 3           | Travel Category              | TravelCategoryBusinessCount, TravelCategoryLeisureCount                                                                                   |
| 4           | Dollars Spent                | TotalDollarSpent                                                                                                                          |
| 5           | Check-in and Checkout dates  | StayDayCount, StayDayCount2016, StayDayCount2015, StayDayCount2014, StayCount, StayCount2016. StayCount2015, StayCount2014                |
| 6           | Service Usage                | UsageTenure, ConciergeUsage, CourierUsage, DryCleaningUsage, GymUsage, PhoneUsage, RestaurantUsage, SpaUsage, TelevisionUsage, WifiUsage  |

### Model selection

Now we need to choose the optimal algorithm to use. In this case, most features are based on categorical features. Typically, decision tree–based models work well. If there are only numerical features, neural networks could be a better choice. Support vector machine (SVM) also is a good candidate in such situations; however, it needs quite a bit of tuning to extract the best performance. We choose **Two-Class Boosted Decision Tree** as the first model of choice followed by **Two-Class SVM** as the second model. Azure Machine Learning lets you do A/B testing, so it’s beneficial to start with two models rather than one.

The following image shows the model training and evaluation pipeline from Azure Machine Learning:

![Churn model in Azure Machine Learning](media/azure-machine-learning-model.png)

We also apply a technique called **Permutation Feature Importance**, an important aspect of model optimization. Built-in models have little to no insight into the impact of any specific feature on the final prediction. The feature importance calculator uses a custom algorithm to compute the influence of individual features on the outcome for a specific model. The feature importance is normalized between +1 to -1. A negative influence means the corresponding feature has counter-intuitive influence on the outcome and should be removed from the model. A positive influence indicates the feature is contributing heavily towards the prediction. These values aren't correlation coefficients as they are different metrics. For more information, see [Permutation Feature Importance](https://docs.microsoft.com/azure/machine-learning/algorithm-module-reference/permutation-feature-importance).
