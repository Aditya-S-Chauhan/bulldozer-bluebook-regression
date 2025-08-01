# Predicting the Sale Price of Bulldozers using Machine Learning 🚜 

In this notebook, we're going to go through a machine learning project to use the characteristics of bulldozers and their past sales prices to predict the sale price of future bulldozers based on their characteristics.

* **Inputs:** Bulldozer characteristics such as make year, base model, model series, state of sale (e.g. which US state was it sold in), drive system and more.
* **Outputs:** Bulldozer sale price (in USD).


And since we're going to predicting results with a time component (predicting future sales based on past sales), this is usng regression for a **time series** or **forecasting** problem.

The data and evaluation metric we'll be using (root mean square log error or RMSLE) is from the [Kaggle Bluebook for Bulldozers competition](https://www.kaggle.com/c/bluebook-for-bulldozers/overview).

The techniques used in here have been inspired and adapted from [the fast.ai machine learning course](https://course18.fast.ai/ml).

## Overview

Since we already have a dataset, we'll approach the problem with the following machine learning modelling framework.

| <img src="https://github.com/mrdbourke/zero-to-mastery-ml/blob/master/images/supervised-projects-6-step-ml-framework-tools-highlight.png?raw=true" width=750/> | 
|:--:| 
| 6 Step Machine Learning Modelling Framework ([read more](https://whimsical.com/9g65jgoRYTxMXxDosndYTB)) |

To work through these topics, we'll use pandas, Matplotlib and NumPy for data analysis, as well as, Scikit-Learn for machine learning and modelling tasks.

| <img src="https://github.com/mrdbourke/zero-to-mastery-ml/blob/master/images/supervised-projects-6-step-ml-framework-tools-highlight.png?raw=true" width=750/> | 
|:--:| 
| Tools that can be used for each step of the machine learning modelling process. |

We'll work through each step and by the end of the notebook, we'll have a trained machine learning model which predicts the sale price of a bulldozer given different characteristics about it.

### Machine Learning Framework

#### 1. Problem Definition

For this dataset, the problem we're trying to solve, or better, the question we're trying to answer is,

> How well can we predict the future sale price of a bulldozer, given its characteristics previous examples of how much similar bulldozers have been sold for?

#### 2. Data

Looking at the [dataset from Kaggle](https://www.kaggle.com/c/bluebook-for-bulldozers/data) we see that it contains historical sales data of bulldozers. Including things like, model type, size, sale date and more.

There are 3 datasets:

1. **Train.csv** - Historical bulldozer sales examples up to 2011 (close to 400,000 examples with 50+ different attributes, including `SalePrice` which is the **target variable**).
2. **Valid.csv** - Historical bulldozer sales examples from January 1 2012 to April 30 2012 (close to 12,000 examples with the same attributes as **Train.csv**).
3. **Test.csv** - Historical bulldozer sales examples from May 1 2012 to November 2012 (close to 12,000 examples but missing the `SalePrice` attribute, as this is what we'll be trying to predict).

> **Note:** You can download the dataset `bluebook-for-bulldozers` dataset directly from Kaggle.

#### 3. Evaluation

For this problem, [Kaggle has set the evaluation metric to being root mean squared log error (RMSLE)](https://www.kaggle.com/c/bluebook-for-bulldozers/overview/evaluation). As with many regression evaluations, the goal will be to get this value as low as possible (a low error value means our model's predictions are close to what the real values are).

To see how well our model is doing, we'll calculate the RMSLE and then compare our results to others on the [Kaggle leaderboard](https://www.kaggle.com/c/bluebook-for-bulldozers/leaderboard).

#### 4. Features

Features are different parts and attributes of the data. 

During this step, you'll want to start finding out what you can about the data.

One of the most common ways to do this is to create a **data dictionary**.

For this dataset, Kaggle provides a data dictionary which contains information about what each attribute of the dataset means. 

For example: 

| Variable Name | Description  | Variable Type |
|------|-----|-----|
| SalesID   | unique identifier of a particular sale of a machine at auction  | Independent  variable |
| MachineID  | identifier for a particular machine; machines may have multiple sales  | Independent  variable |
| ModelID | identifier for a unique machine model (i.e. fiModelDesc) | Independent  variable |
| datasource  | source of the sale record; some sources are more diligent about reporting attributes of the machine than others. Note that a particular datasource may report on multiple auctioneerIDs. | Independent  variable |
| auctioneerID  | identifier of a particular auctioneer, i.e. company that sold the machine at auction. Not the same as datasource.  | Independent  variable |
| YearMade  | year of manufacturer of the Machine  | Independent  variable |
| MachineHoursCurrentMeter | current usage of the machine in hours at time of sale (saledate); null or 0 means no hours have been reported for that sale | Independent  variable |
| UsageBand | value (low, medium, high) calculated comparing this particular Machine-Sale hours to average usage for the fiBaseModel; e.g. 'Low' means this machine has fewer hours given its lifespan relative to the average of fiBaseModel. | Independent  variable |
| Saledate   | time of sale | Independent  variable |
| fiModelDesc  | Description of a unique machine model (see ModelID); concatenation of fiBaseModel & fiSecondaryDesc & fiModelSeries & fiModelDescriptor  | Independent  variable |
| State | US State in which sale occurred | Independent  variable |
| Drive_System | machine configuration; typically describes whether 2 or 4 wheel drive  | Independent  variable |
| Enclosure  | machine configuration - does the machine have an enclosed cab or not | Independent  variable |
| Forks  | machine configuration - attachment used for lifting  | Independent  variable |
| Pad_Type | machine configuration - type of treads a crawler machine uses | Independent  variable |
| Ride_Control  | machine configuration - optional feature on loaders to make the ride smoother | Independent  variable |
| Transmission | machine configuration - describes type of transmission; typically automatic or manual | Independent  variable |
| ... | ... | ... |
| SalePrice | cost of sale in USD | Target/dependent variable | 

You can download the full version of this file directly from the [Kaggle competition page](https://www.kaggle.com/c/bluebook-for-bulldozers/download/Bnl6RAHA0enbg0UfAvGA%2Fversions%2FwBG4f35Q8mAbfkzwCeZn%2Ffiles%2FData%20Dictionary.xlsx) (Kaggle account required) or view it [on Google Sheets](https://docs.google.com/spreadsheets/d/18ly-bLR8sbDJLITkWG7ozKm8l3RyieQ2Fpgix-beSYI/edit?usp=sharing).

With all of this being known, let's get started! 

First, we'll import the dataset and start exploring. 


```python
# Timestamp
import datetime

import datetime
print(f"Notebook last run (end-to-end): {datetime.datetime.now()}")
```

    Notebook last run (end-to-end): 2025-05-02 13:14:39.737361


## 1. Importing the data and preparing it for modelling

First thing is first, let's get the libraries we need imported and the data we'll need for the project.

We'll start by importing pandas, NumPy and matplotlib.


```python
# Import data analysis tools 
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Print the versions we're using (as long as your versions are equal or higher than these, the code should work)
print(f"pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"matplotlib version: {matplotlib.__version__}") 
```

    pandas version: 2.1.4
    NumPy version: 1.26.4
    matplotlib version: 3.10.0


Now we've got our tools for data analysis ready, we can import the data and start to explore it.

We can write some code to check if the files are available locally (on our computer) and if not, we can download them.


```python
# Import training and validation sets
df = pd.read_csv('bluebook-for-bulldozers/TrainAndValid.csv',
                 low_memory= False)
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SalesID</th>
      <th>SalePrice</th>
      <th>MachineID</th>
      <th>ModelID</th>
      <th>datasource</th>
      <th>auctioneerID</th>
      <th>YearMade</th>
      <th>MachineHoursCurrentMeter</th>
      <th>UsageBand</th>
      <th>saledate</th>
      <th>...</th>
      <th>Undercarriage_Pad_Width</th>
      <th>Stick_Length</th>
      <th>Thumb</th>
      <th>Pattern_Changer</th>
      <th>Grouser_Type</th>
      <th>Backhoe_Mounting</th>
      <th>Blade_Type</th>
      <th>Travel_Controls</th>
      <th>Differential_Type</th>
      <th>Steering_Controls</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1139246</td>
      <td>66000.0</td>
      <td>999089</td>
      <td>3157</td>
      <td>121</td>
      <td>3.0</td>
      <td>2004</td>
      <td>68.0</td>
      <td>Low</td>
      <td>11/16/2006 0:00</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Standard</td>
      <td>Conventional</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1139248</td>
      <td>57000.0</td>
      <td>117657</td>
      <td>77</td>
      <td>121</td>
      <td>3.0</td>
      <td>1996</td>
      <td>4640.0</td>
      <td>Low</td>
      <td>3/26/2004 0:00</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Standard</td>
      <td>Conventional</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1139249</td>
      <td>10000.0</td>
      <td>434808</td>
      <td>7009</td>
      <td>121</td>
      <td>3.0</td>
      <td>2001</td>
      <td>2838.0</td>
      <td>High</td>
      <td>2/26/2004 0:00</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1139251</td>
      <td>38500.0</td>
      <td>1026470</td>
      <td>332</td>
      <td>121</td>
      <td>3.0</td>
      <td>2001</td>
      <td>3486.0</td>
      <td>High</td>
      <td>5/19/2011 0:00</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1139253</td>
      <td>11000.0</td>
      <td>1057373</td>
      <td>17311</td>
      <td>121</td>
      <td>3.0</td>
      <td>2007</td>
      <td>722.0</td>
      <td>Medium</td>
      <td>7/23/2009 0:00</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 53 columns</p>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 412698 entries, 0 to 412697
    Data columns (total 53 columns):
     #   Column                    Non-Null Count   Dtype  
    ---  ------                    --------------   -----  
     0   SalesID                   412698 non-null  int64  
     1   SalePrice                 412698 non-null  float64
     2   MachineID                 412698 non-null  int64  
     3   ModelID                   412698 non-null  int64  
     4   datasource                412698 non-null  int64  
     5   auctioneerID              392562 non-null  float64
     6   YearMade                  412698 non-null  int64  
     7   MachineHoursCurrentMeter  147504 non-null  float64
     8   UsageBand                 73670 non-null   object 
     9   saledate                  412698 non-null  object 
     10  fiModelDesc               412698 non-null  object 
     11  fiBaseModel               412698 non-null  object 
     12  fiSecondaryDesc           271971 non-null  object 
     13  fiModelSeries             58667 non-null   object 
     14  fiModelDescriptor         74816 non-null   object 
     15  ProductSize               196093 non-null  object 
     16  fiProductClassDesc        412698 non-null  object 
     17  state                     412698 non-null  object 
     18  ProductGroup              412698 non-null  object 
     19  ProductGroupDesc          412698 non-null  object 
     20  Drive_System              107087 non-null  object 
     21  Enclosure                 412364 non-null  object 
     22  Forks                     197715 non-null  object 
     23  Pad_Type                  81096 non-null   object 
     24  Ride_Control              152728 non-null  object 
     25  Stick                     81096 non-null   object 
     26  Transmission              188007 non-null  object 
     27  Turbocharged              81096 non-null   object 
     28  Blade_Extension           25983 non-null   object 
     29  Blade_Width               25983 non-null   object 
     30  Enclosure_Type            25983 non-null   object 
     31  Engine_Horsepower         25983 non-null   object 
     32  Hydraulics                330133 non-null  object 
     33  Pushblock                 25983 non-null   object 
     34  Ripper                    106945 non-null  object 
     35  Scarifier                 25994 non-null   object 
     36  Tip_Control               25983 non-null   object 
     37  Tire_Size                 97638 non-null   object 
     38  Coupler                   220679 non-null  object 
     39  Coupler_System            44974 non-null   object 
     40  Grouser_Tracks            44875 non-null   object 
     41  Hydraulics_Flow           44875 non-null   object 
     42  Track_Type                102193 non-null  object 
     43  Undercarriage_Pad_Width   102916 non-null  object 
     44  Stick_Length              102261 non-null  object 
     45  Thumb                     102332 non-null  object 
     46  Pattern_Changer           102261 non-null  object 
     47  Grouser_Type              102193 non-null  object 
     48  Backhoe_Mounting          80712 non-null   object 
     49  Blade_Type                81875 non-null   object 
     50  Travel_Controls           81877 non-null   object 
     51  Differential_Type         71564 non-null   object 
     52  Steering_Controls         71522 non-null   object 
    dtypes: float64(3), int64(5), object(45)
    memory usage: 166.9+ MB



```python
df.isna().sum()
```




    SalesID                          0
    SalePrice                        0
    MachineID                        0
    ModelID                          0
    datasource                       0
    auctioneerID                 20136
    YearMade                         0
    MachineHoursCurrentMeter    265194
    UsageBand                   339028
    saledate                         0
    fiModelDesc                      0
    fiBaseModel                      0
    fiSecondaryDesc             140727
    fiModelSeries               354031
    fiModelDescriptor           337882
    ProductSize                 216605
    fiProductClassDesc               0
    state                            0
    ProductGroup                     0
    ProductGroupDesc                 0
    Drive_System                305611
    Enclosure                      334
    Forks                       214983
    Pad_Type                    331602
    Ride_Control                259970
    Stick                       331602
    Transmission                224691
    Turbocharged                331602
    Blade_Extension             386715
    Blade_Width                 386715
    Enclosure_Type              386715
    Engine_Horsepower           386715
    Hydraulics                   82565
    Pushblock                   386715
    Ripper                      305753
    Scarifier                   386704
    Tip_Control                 386715
    Tire_Size                   315060
    Coupler                     192019
    Coupler_System              367724
    Grouser_Tracks              367823
    Hydraulics_Flow             367823
    Track_Type                  310505
    Undercarriage_Pad_Width     309782
    Stick_Length                310437
    Thumb                       310366
    Pattern_Changer             310437
    Grouser_Type                310505
    Backhoe_Mounting            331986
    Blade_Type                  330823
    Travel_Controls             330821
    Differential_Type           341134
    Steering_Controls           341176
    dtype: int64




```python
fig, ax = plt.subplots()
ax.scatter(df.saledate[:1000], df.SalePrice[:1000])
plt.show()
```


    
![png](bluebook-bulldozer-price-regression_files/bluebook-bulldozer-price-regression_14_0.png)
    



```python
df.SalePrice.plot.hist()
```




    <Axes: ylabel='Frequency'>




    
![png](bluebook-bulldozer-price-regression_files/bluebook-bulldozer-price-regression_15_1.png)
    


### 1.1 Parsing dates

Since we are working with time series data, it's a good idea to make sure any date data is the format of a [datetime object](https://docs.python.org/3/library/datetime.html) (a Python data type which encodes specific information about dates).

We can tell pandas which columns to read in as dates by setting the `parse_dates` parameter in [`pd.read_csv`](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html).

Once we've imported our CSV with the `saledate` column parsed, we can view information about our DataFrame again with `df.info()`. 


```python
# Importing data again but this time parse dates

df = pd.read_csv('bluebook-for-bulldozers/TrainAndValid.csv',
                low_memory = False,
                parse_dates = ['saledate'])

```


```python
df.saledate.dtype

```




    dtype('<M8[ns]')




```python
df.saledate[:1000]

```




    0     2006-11-16
    1     2004-03-26
    2     2004-02-26
    3     2011-05-19
    4     2009-07-23
             ...    
    995   2009-07-16
    996   2007-06-14
    997   2005-09-22
    998   2005-07-28
    999   2011-06-16
    Name: saledate, Length: 1000, dtype: datetime64[ns]




```python
fig, ax = plt.subplots()
ax.scatter(df.saledate[:1000] , df.SalePrice[:1000])
plt.show()
```


    
![png](bluebook-bulldozer-price-regression_files/bluebook-bulldozer-price-regression_20_0.png)
    


### 1.2 Sorting our DataFrame by saledate

Now we've formatted our `saledate` column to be NumPy `datetime64[ns]` objects, we can use built-in pandas methods such as `sort_values` to sort our DataFrame by date.

And considering this is a time series problem, sorting our DataFrame by date has the added benefit of making sure our data is sequential.

In other words, we want to use examples from the past (example sale prices from previous dates) to try and predict future bulldozer sale prices. 

Let's use the [`pandas.DataFrame.sort_values`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sort_values.html) method to sort our DataFrame by `saledate` in ascending order.


```python
# 
df.sort_values(by= ['saledate'], inplace= True)
df.saledate[:1000]
```




    205615   1989-01-17
    274835   1989-01-31
    141296   1989-01-31
    212552   1989-01-31
    62755    1989-01-31
                ...    
    54344    1989-03-16
    143206   1989-03-16
    93144    1989-03-16
    86917    1989-03-16
    115938   1989-03-16
    Name: saledate, Length: 1000, dtype: datetime64[ns]



### 1.3 Adding extra features to our DataFrame

One way to potentially increase the predictive power of our data is to enhance it with more features.

This practice is known as [**feature engineering**](https://en.wikipedia.org/wiki/Feature_engineering), taking existing features and using them to create more/different features. 

For now, we'll use our `saledate` column to add extra features such as:

* Year of sale
* Month of sale
* Day of sale
* Day of week sale (e.g. Monday = 1, Tuesday = 2)
* Day of year sale (e.g. January 1st = 1, January 2nd = 2)

Since we're going to be manipulating the data, we'll make a copy of the original DataFrame and perform our changes there.

This will keep the original DataFrame in tact if we need it again.


```python
# Make a copy of the original DataFrame to perform edits on
df_tmp = df.copy()
```


```python
df_tmp.saledate.head(20)
```




    205615   1989-01-17
    274835   1989-01-31
    141296   1989-01-31
    212552   1989-01-31
    62755    1989-01-31
    54653    1989-01-31
    81383    1989-01-31
    204924   1989-01-31
    135376   1989-01-31
    113390   1989-01-31
    113394   1989-01-31
    116419   1989-01-31
    32138    1989-01-31
    127610   1989-01-31
    76171    1989-01-31
    127000   1989-01-31
    128130   1989-01-31
    127626   1989-01-31
    55455    1989-01-31
    55454    1989-01-31
    Name: saledate, dtype: datetime64[ns]



Because we imported the data using `read_csv()` and we asked pandas to parse the dates using `parase_dates=["saledate"]`, we can now access the [different datetime attributes](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DatetimeIndex.html) of the `saledate` column.

Let's use these attributes to add a series of different feature columns to our dataset. 

After we've added these extra columns, we can remove the original `saledate` column as its information will be dispersed across these new columns.


```python
# Add datetime parameters for saledate column
df_tmp['saleYear']= df_tmp.saledate.dt.year
df_tmp['saleMonth']= df_tmp.saledate.dt.month
df_tmp['saleDay']= df_tmp.saledate.dt.day
df_tmp['saleDayOfWeek']= df_tmp.saledate.dt.day_of_week
df_tmp['saleDayOfYear']= df_tmp.saledate.dt.day_of_year

```


```python
df_tmp.T.tail(5)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>205615</th>
      <th>274835</th>
      <th>141296</th>
      <th>212552</th>
      <th>62755</th>
      <th>54653</th>
      <th>81383</th>
      <th>204924</th>
      <th>135376</th>
      <th>113390</th>
      <th>...</th>
      <th>409202</th>
      <th>408976</th>
      <th>411695</th>
      <th>411319</th>
      <th>408889</th>
      <th>410879</th>
      <th>412476</th>
      <th>411927</th>
      <th>407124</th>
      <th>409203</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>saleYear</th>
      <td>1989</td>
      <td>1989</td>
      <td>1989</td>
      <td>1989</td>
      <td>1989</td>
      <td>1989</td>
      <td>1989</td>
      <td>1989</td>
      <td>1989</td>
      <td>1989</td>
      <td>...</td>
      <td>2012</td>
      <td>2012</td>
      <td>2012</td>
      <td>2012</td>
      <td>2012</td>
      <td>2012</td>
      <td>2012</td>
      <td>2012</td>
      <td>2012</td>
      <td>2012</td>
    </tr>
    <tr>
      <th>saleMonth</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>saleDay</th>
      <td>17</td>
      <td>31</td>
      <td>31</td>
      <td>31</td>
      <td>31</td>
      <td>31</td>
      <td>31</td>
      <td>31</td>
      <td>31</td>
      <td>31</td>
      <td>...</td>
      <td>28</td>
      <td>28</td>
      <td>28</td>
      <td>28</td>
      <td>28</td>
      <td>28</td>
      <td>28</td>
      <td>28</td>
      <td>28</td>
      <td>28</td>
    </tr>
    <tr>
      <th>saleDayOfWeek</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>saleDayOfYear</th>
      <td>17</td>
      <td>31</td>
      <td>31</td>
      <td>31</td>
      <td>31</td>
      <td>31</td>
      <td>31</td>
      <td>31</td>
      <td>31</td>
      <td>31</td>
      <td>...</td>
      <td>119</td>
      <td>119</td>
      <td>119</td>
      <td>119</td>
      <td>119</td>
      <td>119</td>
      <td>119</td>
      <td>119</td>
      <td>119</td>
      <td>119</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 412698 columns</p>
</div>




```python
# Drop original saledate column
df_tmp.drop("saledate", axis=1, inplace=True)
```


Now we've broken our `saledate` column into columns/features, we can perform further exploratory analysis such as visualizing the `SalePrice` against the `saleMonth`.

We view the first 10,000 samples (we could also randomly select 10,000 samples too) to see if reveals anything about which month has the highest sales?


```python
# View 10,000 samples SalePrice against saleMonth
fig, ax = plt.subplots()
ax.scatter(x=df_tmp["saleMonth"][:10000], # visualize the first 10000 values
           y=df_tmp["SalePrice"][:10000])
ax.set_xlabel("Sale Month")
ax.set_ylabel("Sale Price ($)");
plt.show()
```


    
![png](bluebook-bulldozer-price-regression_files/bluebook-bulldozer-price-regression_31_0.png)
    


Look like there's not too much conclusive evidence here about which month has the highest sales value.

So, we should plot the median sale price of each month

We can do so by grouping on the `saleMonth` column with [`pandas.DataFrame.groupby`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html) and then getting the median of the `SalePrice` column.


```python
# Group DataFrame by saleMonth and then find the median SalePrice
df_tmp.groupby(["saleMonth"])["SalePrice"].median().plot()
plt.xlabel("Month")
plt.ylabel("Median Sale Price ($)")
plt.show();
```


    
![png](bluebook-bulldozer-price-regression_files/bluebook-bulldozer-price-regression_33_0.png)
    


It looks like the median sale prices of January and February (months 1 and 2) are quite a bit higher than the other months of the year.

Could this be because of New Year budget spending?

In the meantime, there are many other values we could look further into.

### 1.4 Inspect values of other columns

When first exploring a new problem, it's often a good idea to become as familiar with the data as you can.

We can use pandas to aggregate thousands of samples into smaller more managable pieces.

And as we'll see later on, we can use machine learning models to model the data and then later inspect which features the model thought were most important.

How about we see which states sell the most bulldozers?


```python
# Check the different values of different columns
df_tmp.state.value_counts()[:10]

```




    state
    Florida        67320
    Texas          53110
    California     29761
    Washington     16222
    Georgia        14633
    Maryland       13322
    Mississippi    13240
    Ohio           12369
    Illinois       11540
    Colorado       11529
    Name: count, dtype: int64



Looks like Flordia sells a fair amount bulldozers.

How about we go even further and group our samples by `state` and then find the median `SalePrice` per state?

We also compare this to the median `SalePrice` for all samples.


```python
#Group DataFrame by saleMonth and then find the median SalePrice per state as well as across 
median_prices_by_state = df_tmp.groupby(['state'])['SalePrice'] .median() # this will return a pandas Series rather than a DataFrame
median_sale_price = df_tmp['SalePrice'].median()

# Create a plot comparing median sale price per state to median sale price overall
fig, ax = plt.subplots(figsize= (10, 7))

ax.bar(x= median_prices_by_state.index, # State names
       height= median_prices_by_state.values) # Median prices per state

ax.set_xlabel('States')
ax.set_ylabel('Median Sale Price ($)')
ax.tick_params(axis= 'x', rotation= 90, labelsize= 7) # Rotate x-axis labels

ax.axhline(y= median_sale_price, # Overall median line
           color= 'red',
           linestyle= '--',
           label= f'Overall Median: ${median_sale_price:,.0f}' # Formatted label
          )

ax.legend()

plt.tight_layout() # Improve layout to prevent labels from overlapping
plt.show()
```


    
![png](bluebook-bulldozer-price-regression_files/bluebook-bulldozer-price-regression_38_0.png)
    



```python
df_tmp.groupby(['state'])['SalePrice'].median().max()
```




    35000.0



Now that's a nice looking figure!

Interestingly Florida has the most sales and the median sale price is above the overall median of all other states.

And if you had a bulldozer and were chasing the highest sale price, the data would reveal that perhaps selling in South Dakota would be your best bet.

Perhaps bulldozers are in higher demand in South Dakota because of a building or mining boom?

Answering this would require a bit more research.

But what we're doing here is slowly building up a mental model of our data. 

So that if we saw an example in the future, we could compare its values to the ones we've already seen.

## 2. Model driven data exploration

We've performed a small Exploratory Data Analysis (EDA) as well as enriched it with some `datetime` attributes, now let's try to model it.

Why model so early?

Well, we know the evaluation metric (root mean squared log error or RMSLE) we're heading towards. 

We could spend more time doing EDA, finding more out about the data ourselves but what we'll do instead is use a machine learning model to help us do EDA whilst simultaneously working towards the best evaluation metric we can get. 

Remember, one of the biggest goals of starting any new machine learning project is reducing the time between experiments.

Following the [Scikit-Learn machine learning map](https://scikit-learn.org/stable/machine_learning_map.html) and taking into account the fact we've got over 100,000 examples, we find a [`sklearn.linear_model.SGDRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html) or a [`sklearn.ensemble.RandomForestRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn-ensemble-randomforestregressor) model might be a good candidate.

Since we're worked with the Random Forest algorithm before (on the [heart disease classification problem](https://dev.mrdbourke.com/zero-to-mastery-ml/end-to-end-heart-disease-classification/)), let's try it out on our regression problem.

> **Note:** We're trying just one model here for now. But you can try many other kinds of models from the Scikit-Learn library, they mostly work with a similar API. There are even libraries such as [`LazyPredict`](https://github.com/shankarpandala/lazypredict) which will try many models simultaneously and return a table with the results.


```python
# Check for missing values and different datatypes 
df_tmp.info
```




    <bound method DataFrame.info of         SalesID  SalePrice  MachineID  ModelID  datasource  auctioneerID  \
    205615  1646770     9500.0    1126363     8434         132          18.0   
    274835  1821514    14000.0    1194089    10150         132          99.0   
    141296  1505138    50000.0    1473654     4139         132          99.0   
    212552  1671174    16000.0    1327630     8591         132          99.0   
    62755   1329056    22000.0    1336053     4089         132          99.0   
    ...         ...        ...        ...      ...         ...           ...   
    410879  6302984    16000.0    1915521     5266         149          99.0   
    412476  6324811     6000.0    1919104    19330         149          99.0   
    411927  6313029    16000.0    1918416    17244         149          99.0   
    407124  6266251    55000.0     509560     3357         149          99.0   
    409203  6283635    34000.0    1869284     4701         149          99.0   
    
            YearMade  MachineHoursCurrentMeter UsageBand fiModelDesc  ...  \
    205615      1974                       NaN       NaN        TD20  ...   
    274835      1980                       NaN       NaN         A66  ...   
    141296      1978                       NaN       NaN         D7G  ...   
    212552      1980                       NaN       NaN         A62  ...   
    62755       1984                       NaN       NaN         D3B  ...   
    ...          ...                       ...       ...         ...  ...   
    410879      2001                       NaN       NaN        D38E  ...   
    412476      2004                       NaN       NaN        2064  ...   
    411927      2004                       NaN       NaN        337G  ...   
    407124      1993                       NaN       NaN         12G  ...   
    409203      1000                       NaN       NaN        544H  ...   
    
               Backhoe_Mounting Blade_Type      Travel_Controls Differential_Type  \
    205615  None or Unspecified   Straight  None or Unspecified               NaN   
    274835                  NaN        NaN                  NaN          Standard   
    141296  None or Unspecified   Straight  None or Unspecified               NaN   
    212552                  NaN        NaN                  NaN          Standard   
    62755   None or Unspecified        PAT                Lever               NaN   
    ...                     ...        ...                  ...               ...   
    410879  None or Unspecified        PAT  None or Unspecified               NaN   
    412476                  NaN        NaN                  NaN               NaN   
    411927                  NaN        NaN                  NaN               NaN   
    407124                  NaN        NaN                  NaN               NaN   
    409203                  NaN        NaN                  NaN          Standard   
    
           Steering_Controls saleYear saleMonth saleDay saleDayOfWeek  \
    205615               NaN     1989         1      17             1   
    274835      Conventional     1989         1      31             1   
    141296               NaN     1989         1      31             1   
    212552      Conventional     1989         1      31             1   
    62755                NaN     1989         1      31             1   
    ...                  ...      ...       ...     ...           ...   
    410879               NaN     2012         4      28             5   
    412476               NaN     2012         4      28             5   
    411927               NaN     2012         4      28             5   
    407124               NaN     2012         4      28             5   
    409203      Conventional     2012         4      28             5   
    
           saleDayOfYear  
    205615            17  
    274835            31  
    141296            31  
    212552            31  
    62755             31  
    ...              ...  
    410879           119  
    412476           119  
    411927           119  
    407124           119  
    409203           119  
    
    [412698 rows x 57 columns]>




```python
# Find missing values in the head of our DataFrame 
df_tmp.head().isna()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SalesID</th>
      <th>SalePrice</th>
      <th>MachineID</th>
      <th>ModelID</th>
      <th>datasource</th>
      <th>auctioneerID</th>
      <th>YearMade</th>
      <th>MachineHoursCurrentMeter</th>
      <th>UsageBand</th>
      <th>fiModelDesc</th>
      <th>...</th>
      <th>Backhoe_Mounting</th>
      <th>Blade_Type</th>
      <th>Travel_Controls</th>
      <th>Differential_Type</th>
      <th>Steering_Controls</th>
      <th>saleYear</th>
      <th>saleMonth</th>
      <th>saleDay</th>
      <th>saleDayOfWeek</th>
      <th>saleDayOfYear</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>205615</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>274835</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>...</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>141296</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>212552</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>...</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>62755</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 57 columns</p>
</div>



Alright it seems as though we've got some missing values in the `MachineHoursCurrentMeter` as well as the `UsageBand` and a few other columns.

But so far we've only viewed the first few rows.

It'll be very time consuming to go through each row one by one so how about we get the total missing values per column?

We can do so by calling `.isna()` on the whole DataFrame and then chaining it together with `.sum()`.

Doing so will give us the total `True`/`False` values in a given column (when summing, `True` = 1, `False` = 0).


```python
# Check for total missing values per column
df_tmp.isna().sum()
```




    SalesID                          0
    SalePrice                        0
    MachineID                        0
    ModelID                          0
    datasource                       0
    auctioneerID                 20136
    YearMade                         0
    MachineHoursCurrentMeter    265194
    UsageBand                   339028
    fiModelDesc                      0
    fiBaseModel                      0
    fiSecondaryDesc             140727
    fiModelSeries               354031
    fiModelDescriptor           337882
    ProductSize                 216605
    fiProductClassDesc               0
    state                            0
    ProductGroup                     0
    ProductGroupDesc                 0
    Drive_System                305611
    Enclosure                      334
    Forks                       214983
    Pad_Type                    331602
    Ride_Control                259970
    Stick                       331602
    Transmission                224691
    Turbocharged                331602
    Blade_Extension             386715
    Blade_Width                 386715
    Enclosure_Type              386715
    Engine_Horsepower           386715
    Hydraulics                   82565
    Pushblock                   386715
    Ripper                      305753
    Scarifier                   386704
    Tip_Control                 386715
    Tire_Size                   315060
    Coupler                     192019
    Coupler_System              367724
    Grouser_Tracks              367823
    Hydraulics_Flow             367823
    Track_Type                  310505
    Undercarriage_Pad_Width     309782
    Stick_Length                310437
    Thumb                       310366
    Pattern_Changer             310437
    Grouser_Type                310505
    Backhoe_Mounting            331986
    Blade_Type                  330823
    Travel_Controls             330821
    Differential_Type           341134
    Steering_Controls           341176
    saleYear                         0
    saleMonth                        0
    saleDay                          0
    saleDayOfWeek                    0
    saleDayOfYear                    0
    dtype: int64



Woah! It looks like our DataFrame has quite a few missing values.

Not to worry, we can work on fixing this later on.

How about we start by tring to turn all of our data in numbers? 

### 2.1 Inspecting the datatypes in our DataFrame 

One way to help turn all of our data into numbers is to convert the columns with the `object` datatype into a `category` datatype using [`pandas.CategoricalDtype`](https://pandas.pydata.org/docs/reference/api/pandas.CategoricalDtype.html).

> **Note:** There are many different ways to convert values into numbers. And often the best way will be specific to the value you're trying to convert. The method we're going to use, converting all objects (that are mostly strings) to categories is one of the faster methods as it makes a quick assumption that each unique value is its own number. 

We can check the datatype of an individual column using the `.dtype` attribute and we can get its full name using `.dtype.name`.


```python
# Get the dtype of a given column
df_tmp["UsageBand"].dtype, df_tmp["UsageBand"].dtype.name
```




    (dtype('O'), 'object')



Beautiful!

Now we've got a way to check a column's datatype individually.

There's also another group of methods to check a column's datatype directly.

For example, using [`pd.api.types.is_object_dtype(arr_or_dtype)`](https://pandas.pydata.org/docs/reference/api/pandas.api.types.is_object_dtype.html) we can get a boolean response as to whether the input is an object or not.

> **Note:** There are many more of these checks you can perform for other datatypes such as strings under a similar name space `pd.api.types.is_XYZ_dtype`. See the [pandas documentation](https://pandas.pydata.org/docs/reference/arrays.html) for more.

Let's see how it works on our `df_tmp["UsageBand"]` column.


```python
# Check whether a column is an object
pd.api.types.is_object_dtype(df_tmp["UsageBand"])
```




    True




```python
# Check whether a column is a string
pd.api.types.is_string_dtype(df_tmp["state"])
```




    True



Nice!

We can even loop through the items (columns and their labels) in our DataFrame using [`pandas.DataFrame.items()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.items.html) (in Python dictionary terms, calling `.items()` on a DataFrame will treat the column names as the keys and the column values as the values) and print out samples of columns which have the `string` datatype.

As an extra check, passing the sample to [`pd.api.types.infer_dtype()`](https://pandas.pydata.org/docs/reference/api/pandas.api.types.infer_dtype.html) will return the datatype of the sample.

This will be a good way to keep exploring our data.


```python
# Quick exampke of calling .items() on a dictionary
random_dict = {"key1": "hello",
               "key2": "world!"}

for key, value in random_dict.items():
    print(f"This is a key: {key}")
    print(f"This is a value: {value}")
```

    This is a key: key1
    This is a value: hello
    This is a key: key2
    This is a value: world!



```python
for label, content in df_tmp.items():
    if pd.api.types.is_string_dtype(content):
        # Check datatype of target column
        column_dtype= df_tmp[label].dtype.name

        # Get random sample from column values
        example_value = content.sample(1).values

        # Infer random sample datatype
        example_value_dtype = pd.api.types.infer_dtype(example_value)
        print(f"Column name: {label} | Column dtype: {column_dtype} | Example value: {example_value} | Example value dtype: {example_value_dtype}")
        
```

    Column name: fiModelDesc | Column dtype: object | Example value: ['EX200'] | Example value dtype: string
    Column name: fiBaseModel | Column dtype: object | Example value: ['MC70'] | Example value dtype: string
    Column name: fiProductClassDesc | Column dtype: object | Example value: ['Wheel Loader - 120.0 to 135.0 Horsepower'] | Example value dtype: string
    Column name: state | Column dtype: object | Example value: ['Florida'] | Example value dtype: string
    Column name: ProductGroup | Column dtype: object | Example value: ['TTT'] | Example value dtype: string
    Column name: ProductGroupDesc | Column dtype: object | Example value: ['Wheel Loader'] | Example value dtype: string


Hmm... it seems that there are many more columns in the `df_tmp` with the `object` type that didn't display when checking for the string datatype (we know there are many `object` datatype columns in our DataFrame from using `df_tmp.info()`).

How about we try the same as above, except this time instead of `pd.api.types.is_string_dtype`, we use `pd.api.types.is_object_dtype`?

Let's try it.


```python
# Start a count of how many object type columns there are
number_of_object_type_columns = 0

for label, content in df_tmp.items():
    # Check to see if column is of object type (this will include the string columns)
    if pd.api.types.is_object_dtype(content): 
        # Check datatype of target column
        column_datatype = df_tmp[label].dtype.name

        # Get random sample from column values
        example_value = content.sample(1).values

        # Infer random sample datatype
        example_value_dtype = pd.api.types.infer_dtype(example_value)
        print(f"Column name: {label} | Column dtype: {column_datatype} | Example value: {example_value} | Example value dtype: {example_value_dtype}")

        number_of_object_type_columns += 1

print(f"\n[INFO] Total number of object type columns: {number_of_object_type_columns}")
```

    Column name: UsageBand | Column dtype: object | Example value: [nan] | Example value dtype: empty
    Column name: fiModelDesc | Column dtype: object | Example value: ['214E'] | Example value dtype: string
    Column name: fiBaseModel | Column dtype: object | Example value: ['PC160'] | Example value dtype: string
    Column name: fiSecondaryDesc | Column dtype: object | Example value: ['B'] | Example value dtype: string
    Column name: fiModelSeries | Column dtype: object | Example value: [nan] | Example value dtype: empty
    Column name: fiModelDescriptor | Column dtype: object | Example value: [nan] | Example value dtype: empty
    Column name: ProductSize | Column dtype: object | Example value: [nan] | Example value dtype: empty
    Column name: fiProductClassDesc | Column dtype: object | Example value: ['Hydraulic Excavator, Track - 33.0 to 40.0 Metric Tons'] | Example value dtype: string
    Column name: state | Column dtype: object | Example value: ['Texas'] | Example value dtype: string
    Column name: ProductGroup | Column dtype: object | Example value: ['SSL'] | Example value dtype: string
    Column name: ProductGroupDesc | Column dtype: object | Example value: ['Track Excavators'] | Example value dtype: string
    Column name: Drive_System | Column dtype: object | Example value: [nan] | Example value dtype: empty
    Column name: Enclosure | Column dtype: object | Example value: ['EROPS'] | Example value dtype: string
    Column name: Forks | Column dtype: object | Example value: [nan] | Example value dtype: empty
    Column name: Pad_Type | Column dtype: object | Example value: [nan] | Example value dtype: empty
    Column name: Ride_Control | Column dtype: object | Example value: [nan] | Example value dtype: empty
    Column name: Stick | Column dtype: object | Example value: [nan] | Example value dtype: empty
    Column name: Transmission | Column dtype: object | Example value: [nan] | Example value dtype: empty
    Column name: Turbocharged | Column dtype: object | Example value: ['None or Unspecified'] | Example value dtype: string
    Column name: Blade_Extension | Column dtype: object | Example value: [nan] | Example value dtype: empty
    Column name: Blade_Width | Column dtype: object | Example value: [nan] | Example value dtype: empty
    Column name: Enclosure_Type | Column dtype: object | Example value: [nan] | Example value dtype: empty
    Column name: Engine_Horsepower | Column dtype: object | Example value: [nan] | Example value dtype: empty
    Column name: Hydraulics | Column dtype: object | Example value: ['2 Valve'] | Example value dtype: string
    Column name: Pushblock | Column dtype: object | Example value: [nan] | Example value dtype: empty
    Column name: Ripper | Column dtype: object | Example value: [nan] | Example value dtype: empty
    Column name: Scarifier | Column dtype: object | Example value: [nan] | Example value dtype: empty
    Column name: Tip_Control | Column dtype: object | Example value: [nan] | Example value dtype: empty
    Column name: Tire_Size | Column dtype: object | Example value: [nan] | Example value dtype: empty
    Column name: Coupler | Column dtype: object | Example value: [nan] | Example value dtype: empty
    Column name: Coupler_System | Column dtype: object | Example value: ['None or Unspecified'] | Example value dtype: string
    Column name: Grouser_Tracks | Column dtype: object | Example value: [nan] | Example value dtype: empty
    Column name: Hydraulics_Flow | Column dtype: object | Example value: [nan] | Example value dtype: empty
    Column name: Track_Type | Column dtype: object | Example value: [nan] | Example value dtype: empty
    Column name: Undercarriage_Pad_Width | Column dtype: object | Example value: [nan] | Example value dtype: empty
    Column name: Stick_Length | Column dtype: object | Example value: [nan] | Example value dtype: empty
    Column name: Thumb | Column dtype: object | Example value: ['None or Unspecified'] | Example value dtype: string
    Column name: Pattern_Changer | Column dtype: object | Example value: [nan] | Example value dtype: empty
    Column name: Grouser_Type | Column dtype: object | Example value: ['Double'] | Example value dtype: string
    Column name: Backhoe_Mounting | Column dtype: object | Example value: [nan] | Example value dtype: empty
    Column name: Blade_Type | Column dtype: object | Example value: ['PAT'] | Example value dtype: string
    Column name: Travel_Controls | Column dtype: object | Example value: [nan] | Example value dtype: empty
    Column name: Differential_Type | Column dtype: object | Example value: [nan] | Example value dtype: empty
    Column name: Steering_Controls | Column dtype: object | Example value: [nan] | Example value dtype: empty
    
    [INFO] Total number of object type columns: 44


Wonderful, looks like we've got sample outputs from all of the columns with the `object` datatype.

It also looks like that many of random samples are missing values.

### 2.2 Converting strings to categories with pandas 

In pandas, one way to convert object/string values to numerical values is to convert them to categories or more specifically, the `pd.CategoricalDtype` datatype.

This datatype keeps the underlying data the same (e.g. doesn't change the string) but enables easy conversion to a numeric code using [`.cat.codes`](https://pandas.pydata.org/docs/reference/api/pandas.Series.cat.codes.html).

For example, the column `state` might have the values `'Alabama', 'Alaska', 'Arizona'...` and these could be mapped to numeric values `1, 2, 3...` respectively.

To see this in action, let's first convert the object datatype columns to `"category"` datatype.

We can do so by looping through the `.items()` of our DataFrame and reassigning each object datatype column using [`pandas.Series.astype(dtype="category")`](https://pandas.pydata.org/docs/reference/api/pandas.Series.astype.html).


```python
for label, content in df_tmp.items(): 
    if pd.api.types.is_object_dtype(content):
        df_tmp[label] = df_tmp[label].astype("category")
```

Wonderful!

Now let's check if it worked by calling `.info()` on our DataFrame.


```python
df_tmp.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 412698 entries, 205615 to 409203
    Data columns (total 57 columns):
     #   Column                    Non-Null Count   Dtype   
    ---  ------                    --------------   -----   
     0   SalesID                   412698 non-null  int64   
     1   SalePrice                 412698 non-null  float64 
     2   MachineID                 412698 non-null  int64   
     3   ModelID                   412698 non-null  int64   
     4   datasource                412698 non-null  int64   
     5   auctioneerID              392562 non-null  float64 
     6   YearMade                  412698 non-null  int64   
     7   MachineHoursCurrentMeter  147504 non-null  float64 
     8   UsageBand                 73670 non-null   category
     9   fiModelDesc               412698 non-null  category
     10  fiBaseModel               412698 non-null  category
     11  fiSecondaryDesc           271971 non-null  category
     12  fiModelSeries             58667 non-null   category
     13  fiModelDescriptor         74816 non-null   category
     14  ProductSize               196093 non-null  category
     15  fiProductClassDesc        412698 non-null  category
     16  state                     412698 non-null  category
     17  ProductGroup              412698 non-null  category
     18  ProductGroupDesc          412698 non-null  category
     19  Drive_System              107087 non-null  category
     20  Enclosure                 412364 non-null  category
     21  Forks                     197715 non-null  category
     22  Pad_Type                  81096 non-null   category
     23  Ride_Control              152728 non-null  category
     24  Stick                     81096 non-null   category
     25  Transmission              188007 non-null  category
     26  Turbocharged              81096 non-null   category
     27  Blade_Extension           25983 non-null   category
     28  Blade_Width               25983 non-null   category
     29  Enclosure_Type            25983 non-null   category
     30  Engine_Horsepower         25983 non-null   category
     31  Hydraulics                330133 non-null  category
     32  Pushblock                 25983 non-null   category
     33  Ripper                    106945 non-null  category
     34  Scarifier                 25994 non-null   category
     35  Tip_Control               25983 non-null   category
     36  Tire_Size                 97638 non-null   category
     37  Coupler                   220679 non-null  category
     38  Coupler_System            44974 non-null   category
     39  Grouser_Tracks            44875 non-null   category
     40  Hydraulics_Flow           44875 non-null   category
     41  Track_Type                102193 non-null  category
     42  Undercarriage_Pad_Width   102916 non-null  category
     43  Stick_Length              102261 non-null  category
     44  Thumb                     102332 non-null  category
     45  Pattern_Changer           102261 non-null  category
     46  Grouser_Type              102193 non-null  category
     47  Backhoe_Mounting          80712 non-null   category
     48  Blade_Type                81875 non-null   category
     49  Travel_Controls           81877 non-null   category
     50  Differential_Type         71564 non-null   category
     51  Steering_Controls         71522 non-null   category
     52  saleYear                  412698 non-null  int32   
     53  saleMonth                 412698 non-null  int32   
     54  saleDay                   412698 non-null  int32   
     55  saleDayOfWeek             412698 non-null  int32   
     56  saleDayOfYear             412698 non-null  int32   
    dtypes: category(44), float64(3), int32(5), int64(5)
    memory usage: 71.5 MB


It looks like it worked!

All of the object datatype columns now have the category datatype.



```python
# Check the datatype of a single column
df_tmp.state.dtype
```




    CategoricalDtype(categories=['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California',
                      'Colorado', 'Connecticut', 'Delaware', 'Florida', 'Georgia',
                      'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas',
                      'Kentucky', 'Louisiana', 'Maine', 'Maryland',
                      'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi',
                      'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire',
                      'New Jersey', 'New Mexico', 'New York', 'North Carolina',
                      'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania',
                      'Puerto Rico', 'Rhode Island', 'South Carolina',
                      'South Dakota', 'Tennessee', 'Texas', 'Unspecified', 'Utah',
                      'Vermont', 'Virginia', 'Washington', 'Washington DC',
                      'West Virginia', 'Wisconsin', 'Wyoming'],
    , ordered=False, categories_dtype=object)



Excellent, notice how the column is now of type `pd.CategoricalDtype`.

We can also access these categories using [`pandas.Series.cat.categories`](https://pandas.pydata.org/docs/reference/api/pandas.Series.cat.categories.html).


```python
# Get the category names of a given column
df_tmp.state.cat.categories
```




    Index(['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado',
           'Connecticut', 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho',
           'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana',
           'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota',
           'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada',
           'New Hampshire', 'New Jersey', 'New Mexico', 'New York',
           'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon',
           'Pennsylvania', 'Puerto Rico', 'Rhode Island', 'South Carolina',
           'South Dakota', 'Tennessee', 'Texas', 'Unspecified', 'Utah', 'Vermont',
           'Virginia', 'Washington', 'Washington DC', 'West Virginia', 'Wisconsin',
           'Wyoming'],
          dtype='object')



Finally, we can get the category codes (the numeric values representing the category) using [`pandas.Series.cat.codes`](https://pandas.pydata.org/docs/reference/api/pandas.Series.cat.codes.html).


```python
df_tmp.state.cat.codes
```




    205615    43
    274835     8
    141296     8
    212552     8
    62755      8
              ..
    410879     4
    412476     4
    411927     4
    407124     4
    409203     4
    Length: 412698, dtype: int8



This gives us a numeric representation of our object/string datatype columns.


```python
# Get example string using category number
target_state_cat_number = 43
target_state_cat_value = df_tmp.state.cat.categories[target_state_cat_number] 
print(f"[INFO] Target state category number {target_state_cat_number} maps to: {target_state_cat_value}")
```

    [INFO] Target state category number 43 maps to: Texas


All of our data is categorical and thus we can now turn the categories into numbers, however it's still missing values.

### 2.3 Saving our preprocessed data (part 1)

Before we start doing any further preprocessing steps on our DataFrame, how about we save our current DataFrame to file so we could import it again later if necessary.

Saving and updating your dataset as you go is common practice in machine learning problems. As your problem changes and evolves, the dataset you're working with will likely change too.

Making checkpoints of your dataset is similar to making checkpoints of your code.

There are some limitations of the CSV (`.csv`) file format, it doesn't preserve data types, rather it stores all the values as strings.

So when we read in a CSV, pandas defaults to interpreting strings as `object` datatypes.

Not to worry though, we can easily convert them to the `category` datatype as we did before.

> **Note:** If you'd like to retain the datatypes when saving your data, you can use file formats such as [`parquet`](https://pandas.pydata.org/docs/user_guide/io.html#parquet) (Apache Parquet) and [`feather`](https://pandas.pydata.org/docs/user_guide/io.html#feather). These filetypes have several advantages over CSV in terms of processing speeds and storage size. However, data stored in these formats is not human-readable so you won't be able to open the files and inspect them without specific tools. For more on different file formats in pandas, see the [IO tools documentation page](https://pandas.pydata.org/docs/user_guide/io.html#).

Let's try using `parquet` format.

To do so, we can use the [`pandas.DataFrame.to_parquet()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_parquet.html) method.

Files in the `parquet` format typically have the file extension of `.parquet`.


```python
# To save to parquet format requires pyarrow or fastparquet (or both)
# Can install via `pip install pyarrow fastparquet`
df_tmp.to_parquet(path="TrainAndValid_object_values_as_categories.parquet", 
                  engine="auto") # "auto" will automatically use pyarrow or fastparquet, defaulting to pyarrow first
```


```python
# Read in df_tmp from parquet format
df_tmp = pd.read_parquet(path="TrainAndValid_object_values_as_categories.parquet",
                         engine="auto")

# Using parquet format, datatypes are preserved
df_tmp.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 412698 entries, 205615 to 409203
    Data columns (total 57 columns):
     #   Column                    Non-Null Count   Dtype   
    ---  ------                    --------------   -----   
     0   SalesID                   412698 non-null  int64   
     1   SalePrice                 412698 non-null  float64 
     2   MachineID                 412698 non-null  int64   
     3   ModelID                   412698 non-null  int64   
     4   datasource                412698 non-null  int64   
     5   auctioneerID              392562 non-null  float64 
     6   YearMade                  412698 non-null  int64   
     7   MachineHoursCurrentMeter  147504 non-null  float64 
     8   UsageBand                 73670 non-null   category
     9   fiModelDesc               412698 non-null  category
     10  fiBaseModel               412698 non-null  category
     11  fiSecondaryDesc           271971 non-null  category
     12  fiModelSeries             58667 non-null   category
     13  fiModelDescriptor         74816 non-null   category
     14  ProductSize               196093 non-null  category
     15  fiProductClassDesc        412698 non-null  category
     16  state                     412698 non-null  category
     17  ProductGroup              412698 non-null  category
     18  ProductGroupDesc          412698 non-null  category
     19  Drive_System              107087 non-null  category
     20  Enclosure                 412364 non-null  category
     21  Forks                     197715 non-null  category
     22  Pad_Type                  81096 non-null   category
     23  Ride_Control              152728 non-null  category
     24  Stick                     81096 non-null   category
     25  Transmission              188007 non-null  category
     26  Turbocharged              81096 non-null   category
     27  Blade_Extension           25983 non-null   category
     28  Blade_Width               25983 non-null   category
     29  Enclosure_Type            25983 non-null   category
     30  Engine_Horsepower         25983 non-null   category
     31  Hydraulics                330133 non-null  category
     32  Pushblock                 25983 non-null   category
     33  Ripper                    106945 non-null  category
     34  Scarifier                 25994 non-null   category
     35  Tip_Control               25983 non-null   category
     36  Tire_Size                 97638 non-null   category
     37  Coupler                   220679 non-null  category
     38  Coupler_System            44974 non-null   category
     39  Grouser_Tracks            44875 non-null   category
     40  Hydraulics_Flow           44875 non-null   category
     41  Track_Type                102193 non-null  category
     42  Undercarriage_Pad_Width   102916 non-null  category
     43  Stick_Length              102261 non-null  category
     44  Thumb                     102332 non-null  category
     45  Pattern_Changer           102261 non-null  category
     46  Grouser_Type              102193 non-null  category
     47  Backhoe_Mounting          80712 non-null   category
     48  Blade_Type                81875 non-null   category
     49  Travel_Controls           81877 non-null   category
     50  Differential_Type         71564 non-null   category
     51  Steering_Controls         71522 non-null   category
     52  saleYear                  412698 non-null  int32   
     53  saleMonth                 412698 non-null  int32   
     54  saleDay                   412698 non-null  int32   
     55  saleDayOfWeek             412698 non-null  int32   
     56  saleDayOfYear             412698 non-null  int32   
    dtypes: category(44), float64(3), int32(5), int64(5)
    memory usage: 55.4 MB


### 2.4 Finding and filling missing values

Let's remind ourselves of the missing values by getting the top 20 columns with the most missing values.

We do so by summing the results of [`pandas.DataFrame.isna()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.isna.html) and then using [`sort_values(ascending=False)`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sort_values.html) to showcase the rows with the most missing.


```python
# Check missing values
df_tmp.isna().sum().sort_values(ascending=False)[:20]
```




    Blade_Width          386715
    Engine_Horsepower    386715
    Tip_Control          386715
    Pushblock            386715
    Blade_Extension      386715
    Enclosure_Type       386715
    Scarifier            386704
    Hydraulics_Flow      367823
    Grouser_Tracks       367823
    Coupler_System       367724
    fiModelSeries        354031
    Steering_Controls    341176
    Differential_Type    341134
    UsageBand            339028
    fiModelDescriptor    337882
    Backhoe_Mounting     331986
    Stick                331602
    Turbocharged         331602
    Pad_Type             331602
    Blade_Type           330823
    dtype: int64



Ok, it seems like there are a fair few columns with missing values and there are several datatypes across these columns (numerical, categorical).

How about we break the problem down and work on filling each datatype separately?

### 2.5 Filling missing numerical values

There's no set way to fill missing values in your dataset.

And unless you're filling the missing samples with newly discovered actual data, every way you fill your dataset's missing values will introduce some sort of noise or bias. 

We'll start by filling the missing numerical values in ourdataet.

To do this, we'll first find the numeric datatype columns.

We can do by looping through the columns in our DataFrame and calling [`pd.api.types.is_numeric_dtype(arr_or_dtype)`](https://pandas.pydata.org/docs/reference/api/pandas.api.types.is_numeric_dtype.html) on them.


```python
for label, content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):
        # Check datatype of target column
        column_datatype = df_tmp[label].dtype.name

        # Get random sample from column values
        example_value = content.sample(1).values
        
        # Infer random sample datatype
        example_value_dtype = pd.api.types.infer_dtype(example_value)
        print(f"Column name: {label} | Column dtype: {column_datatype} | Example value: {example_value} | Example value dtype: {example_value_dtype}")

```

    Column name: SalesID | Column dtype: int64 | Example value: [1449989] | Example value dtype: integer
    Column name: SalePrice | Column dtype: float64 | Example value: [27500.] | Example value dtype: floating
    Column name: MachineID | Column dtype: int64 | Example value: [1489765] | Example value dtype: integer
    Column name: ModelID | Column dtype: int64 | Example value: [495] | Example value dtype: integer
    Column name: datasource | Column dtype: int64 | Example value: [132] | Example value dtype: integer
    Column name: auctioneerID | Column dtype: float64 | Example value: [4.] | Example value dtype: floating
    Column name: YearMade | Column dtype: int64 | Example value: [1996] | Example value dtype: integer
    Column name: MachineHoursCurrentMeter | Column dtype: float64 | Example value: [5446.] | Example value dtype: floating
    Column name: saleYear | Column dtype: int32 | Example value: [1999] | Example value dtype: integer
    Column name: saleMonth | Column dtype: int32 | Example value: [10] | Example value dtype: integer
    Column name: saleDay | Column dtype: int32 | Example value: [16] | Example value dtype: integer
    Column name: saleDayOfWeek | Column dtype: int32 | Example value: [5] | Example value dtype: integer
    Column name: saleDayOfYear | Column dtype: int32 | Example value: [272] | Example value dtype: integer


Beautiful! Looks like we've got a mixture of `int64` and `float64` numerical datatypes.

Now how about we find out which numeric columns have missing values?

We can do so by using `pandas.isnull(obj).sum()` to detect and sum the missing values in a given array-like object (in our case, the data in a target column).

Let's loop through our DataFrame columns, find the numeric datatypes and check if they have any missing values.


```python
# Check for which numeric columns have null values
for label, content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            print(f"Column name: {label} | Has missing values: {True}")
        else:
            print(f"Column name: {label} | Has missing values: {False}")
```

    Column name: SalesID | Has missing values: False
    Column name: SalePrice | Has missing values: False
    Column name: MachineID | Has missing values: False
    Column name: ModelID | Has missing values: False
    Column name: datasource | Has missing values: False
    Column name: auctioneerID | Has missing values: True
    Column name: YearMade | Has missing values: False
    Column name: MachineHoursCurrentMeter | Has missing values: True
    Column name: saleYear | Has missing values: False
    Column name: saleMonth | Has missing values: False
    Column name: saleDay | Has missing values: False
    Column name: saleDayOfWeek | Has missing values: False
    Column name: saleDayOfYear | Has missing values: False


Okay, it looks like our `auctioneerID` and `MachineHoursCurrentMeter` columns have missing numeric values.

Let's have a look at how we might handle these.

### 2.6 Discussing possible ways to handle missing values

As previously discussed, there are many ways to fill missing values.

For missing numeric values, some potential options are:

| **Method** | **Pros**  | **Cons** |
|-----|-----|-----|
| **Fill with mean of column** | - Easy to calculate/implement <br> - Retains overall data distribution | - Averages out variation <br> - Affected by outliers (e.g. if one value is much higher/lower than others) |
| **Fill with median of column** | - Easy to calculate/implement <br> - Robust to outliers <br> - Preserves center of data  | - Ignores data distribution shape |
| **Fill with mode of column** | - Easy to calculate/implement <br> - More useful for categorical-like data | - May not make sense for continuous/numerical data  |
| **Fill with 0 (or another constant)** | - Simple to implement <br> - Useful in certain contexts like counts  | - Introduces bias (e.g. if 0 was a value that meant something) <br> - Skews data (e.g. if many missing values, replacing all with 0 makes it look like that's the most common value)  |
| **Forward/Backward fill (use previous/future values to fill future/previous values)**  | - Maintains temporal continuity (for time series) | - Assumes data is continuous, which may not be valid |
| **Use a calculation from other columns** | - Takes existing information and reinterprets it | - Can result in unlikely outputs if calculations are not continuous | 
| **Interpolate  (e.g. like dragging a cell in Excel/Google Sheets)** | - Captures trends <br> - Suitable for ordered data | - Can introduce errors <br> - May assume linearity (data continues in a straight line) |
| **Drop missing values** | - Ensures complete data (only use samples with all information) <br> - Useful for small datasets | - Can result in data loss (e.g. if many missing values are scattered across columns, data size can be dramatically reduced) <br> - Reduces dataset size  |

Which method you choose will be dataset and problem dependant and will likely require several phases of experimentation to see what works and what doesn't.

For now, we'll fill our missing numeric values with the median value of the target column.

We'll also add a binary column (0 or 1) with rows reflecting whether or not a value was missing.

For example, `MachineHoursCurrentMeter_is_missing` will be a column with rows which have a value of `0` if that row's `MachineHoursCurrentMeter` column was *not* missing and `1` if it was.




```python
# Fill missing numeric values with the median of the target column
for label, content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            
            # Add a binary column which tells if the data was missing our not
            df_tmp[label+"_is_missing"] = pd.isnull(content).astype(int) # this will add a 0 or 1 value to rows with missing values (e.g. 0 = not missing, 1 = missing)

            # Fill missing numeric values with median since it's more robust than the mean
            df_tmp[label] = content.fillna(content.median())
```

Why add a binary column indicating whether the data was missing or not?

We can easily fill all of the missing numeric values in our dataset with the median. 

However, a numeric value may be missing for a reason. 

Adding a binary column which indicates whether the value was missing or not helps to retain this information. It also means we can inspect these rows later on.


```python
# Show rows where MachineHoursCurrentMeter_is_missing == 1
df_tmp[df_tmp["MachineHoursCurrentMeter_is_missing"] == 1].sample(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SalesID</th>
      <th>SalePrice</th>
      <th>MachineID</th>
      <th>ModelID</th>
      <th>datasource</th>
      <th>auctioneerID</th>
      <th>YearMade</th>
      <th>MachineHoursCurrentMeter</th>
      <th>UsageBand</th>
      <th>fiModelDesc</th>
      <th>...</th>
      <th>Travel_Controls</th>
      <th>Differential_Type</th>
      <th>Steering_Controls</th>
      <th>saleYear</th>
      <th>saleMonth</th>
      <th>saleDay</th>
      <th>saleDayOfWeek</th>
      <th>saleDayOfYear</th>
      <th>auctioneerID_is_missing</th>
      <th>MachineHoursCurrentMeter_is_missing</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>156734</th>
      <td>1556818</td>
      <td>95500.0</td>
      <td>773685</td>
      <td>1269</td>
      <td>132</td>
      <td>2.0</td>
      <td>2006</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>330CL</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2011</td>
      <td>2</td>
      <td>4</td>
      <td>4</td>
      <td>35</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>153074</th>
      <td>1532600</td>
      <td>24000.0</td>
      <td>1200675</td>
      <td>7052</td>
      <td>132</td>
      <td>2.0</td>
      <td>1994</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>307</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2005</td>
      <td>3</td>
      <td>31</td>
      <td>3</td>
      <td>90</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>140726</th>
      <td>1502957</td>
      <td>54000.0</td>
      <td>1415014</td>
      <td>4155</td>
      <td>132</td>
      <td>6.0</td>
      <td>1980</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>D9H</td>
      <td>...</td>
      <td>None or Unspecified</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1995</td>
      <td>6</td>
      <td>8</td>
      <td>3</td>
      <td>159</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>126516</th>
      <td>1473607</td>
      <td>22500.0</td>
      <td>1469446</td>
      <td>7511</td>
      <td>132</td>
      <td>2.0</td>
      <td>1993</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>E70</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2004</td>
      <td>9</td>
      <td>18</td>
      <td>5</td>
      <td>262</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>203784</th>
      <td>1643471</td>
      <td>23000.0</td>
      <td>1236539</td>
      <td>4605</td>
      <td>132</td>
      <td>1.0</td>
      <td>2005</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>310G</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2009</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>62</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 59 columns</p>
</div>



Missing numeric values filled!

How about we check again whether or not the numeric columns have missing values?


```python
for label, content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            print(f"Column name: {label} | Has missing values: {True}")
        else:
             print(f"Column name: {label} | Has missing values: {False}")
```

    Column name: SalesID | Has missing values: False
    Column name: SalePrice | Has missing values: False
    Column name: MachineID | Has missing values: False
    Column name: ModelID | Has missing values: False
    Column name: datasource | Has missing values: False
    Column name: auctioneerID | Has missing values: False
    Column name: YearMade | Has missing values: False
    Column name: MachineHoursCurrentMeter | Has missing values: False
    Column name: saleYear | Has missing values: False
    Column name: saleMonth | Has missing values: False
    Column name: saleDay | Has missing values: False
    Column name: saleDayOfWeek | Has missing values: False
    Column name: saleDayOfYear | Has missing values: False
    Column name: auctioneerID_is_missing | Has missing values: False
    Column name: MachineHoursCurrentMeter_is_missing | Has missing values: False


And thanks to our binary `_is_missing` columns, we can even check how many were missing.


```python
# Check to see how many examples in the auctioneerID were missing
df_tmp.auctioneerID_is_missing.value_counts()
```




    auctioneerID_is_missing
    0    392562
    1     20136
    Name: count, dtype: int64



### 2.7 Filling missing categorical values with pandas

Now we've filled the numeric values, we'll do the same with the categorical values whilst ensuring that they are all numerical too.

Let's first investigate the columns which *aren't* numeric (we've already worked with these). 


```python
# Check columns which aren't numeric
print(f"[INFO] Columns which are not numeric:")
for label, content in df_tmp.items():
    if not pd.api.types.is_numeric_dtype(content):
        print(f"Column name: {label} | Column dtype: {df_tmp[label].dtype.name}")
```

    [INFO] Columns which are not numeric:
    Column name: UsageBand | Column dtype: category
    Column name: fiModelDesc | Column dtype: category
    Column name: fiBaseModel | Column dtype: category
    Column name: fiSecondaryDesc | Column dtype: category
    Column name: fiModelSeries | Column dtype: category
    Column name: fiModelDescriptor | Column dtype: category
    Column name: ProductSize | Column dtype: category
    Column name: fiProductClassDesc | Column dtype: category
    Column name: state | Column dtype: category
    Column name: ProductGroup | Column dtype: category
    Column name: ProductGroupDesc | Column dtype: category
    Column name: Drive_System | Column dtype: category
    Column name: Enclosure | Column dtype: category
    Column name: Forks | Column dtype: category
    Column name: Pad_Type | Column dtype: category
    Column name: Ride_Control | Column dtype: category
    Column name: Stick | Column dtype: category
    Column name: Transmission | Column dtype: category
    Column name: Turbocharged | Column dtype: category
    Column name: Blade_Extension | Column dtype: category
    Column name: Blade_Width | Column dtype: category
    Column name: Enclosure_Type | Column dtype: category
    Column name: Engine_Horsepower | Column dtype: category
    Column name: Hydraulics | Column dtype: category
    Column name: Pushblock | Column dtype: category
    Column name: Ripper | Column dtype: category
    Column name: Scarifier | Column dtype: category
    Column name: Tip_Control | Column dtype: category
    Column name: Tire_Size | Column dtype: category
    Column name: Coupler | Column dtype: category
    Column name: Coupler_System | Column dtype: category
    Column name: Grouser_Tracks | Column dtype: category
    Column name: Hydraulics_Flow | Column dtype: category
    Column name: Track_Type | Column dtype: category
    Column name: Undercarriage_Pad_Width | Column dtype: category
    Column name: Stick_Length | Column dtype: category
    Column name: Thumb | Column dtype: category
    Column name: Pattern_Changer | Column dtype: category
    Column name: Grouser_Type | Column dtype: category
    Column name: Backhoe_Mounting | Column dtype: category
    Column name: Blade_Type | Column dtype: category
    Column name: Travel_Controls | Column dtype: category
    Column name: Differential_Type | Column dtype: category
    Column name: Steering_Controls | Column dtype: category


Okay, we've got plenty of category type columns.

Let's now write some code to fill the missing categorical values as well as ensure they are numerical (non-string). 

To do so, we'll:

1. Create a blank column to category dictionary, we'll use this to store categorical value names (e.g. their string name) as well as their categorical code. We'll end with a dictionary of dictionaries in the form `{"column_name": {category_code: "category_value"...}...}`.
2. Loop through the items in the DataFrame.
3. Check if the column is numeric or not.
4. Add a binary column in the form `ORIGINAL_COLUMN_NAME_is_missing` with a `0` or `1` value for if the row had a missing value.
5. Ensure the column values are in the `pd.Categorical` datatype and get their category codes with `pd.Series.cat.codes` (we'll add `1` to these values since pandas defaults to assigning `-1` to `NaN` values, we'll use `0` instead).
6. Turn the column categories and column category codes from 5 into a dictionary with Python's [`dict(zip(category_names, category_codes))`](https://docs.python.org/3.3/library/functions.html#zip) and save this to the blank dictionary from 1 with the target column name as key.
7. Set the target column value to the numerical category values from 5.

Phew!

That's a fair few steps but nothing we can't handle.

Let's do it!


```python
# 1. Create a dictionary to store column to category values (e.g. we turn our category types into numbers but we keep a record so we can go back)
column_to_category_dict = {} 

# 2. Turn categorical variables into numbers
for label, content in df_tmp.items():

    # 3. Check columns which *aren't* numeric
    if not pd.api.types.is_numeric_dtype(content):

        # 4. Add binary column to inidicate whether sample had missing value
        df_tmp[label+"_is_missing"] = pd.isnull(content).astype(int)

        # 5. Ensure content is categorical and get its category codes
        content_categories = pd.Categorical(content)
        content_category_codes = content_categories.codes + 1 # prevents -1 (the default for NaN values) from being used for missing values (we'll treat missing values as 0)

        # 6. Add column key to dictionary with code: category mapping per column
        column_to_category_dict[label] = dict(zip(content_category_codes, content_categories))
        
        # 7. Set the column to the numerical values (the category code value) 
        df_tmp[label] = content_category_codes      
```

Let's check out a few random samples of our DataFrame.


```python
df_tmp.sample(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SalesID</th>
      <th>SalePrice</th>
      <th>MachineID</th>
      <th>ModelID</th>
      <th>datasource</th>
      <th>auctioneerID</th>
      <th>YearMade</th>
      <th>MachineHoursCurrentMeter</th>
      <th>UsageBand</th>
      <th>fiModelDesc</th>
      <th>...</th>
      <th>Undercarriage_Pad_Width_is_missing</th>
      <th>Stick_Length_is_missing</th>
      <th>Thumb_is_missing</th>
      <th>Pattern_Changer_is_missing</th>
      <th>Grouser_Type_is_missing</th>
      <th>Backhoe_Mounting_is_missing</th>
      <th>Blade_Type_is_missing</th>
      <th>Travel_Controls_is_missing</th>
      <th>Differential_Type_is_missing</th>
      <th>Steering_Controls_is_missing</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>185198</th>
      <td>1620064</td>
      <td>25000.0</td>
      <td>1353324</td>
      <td>8050</td>
      <td>132</td>
      <td>1.0</td>
      <td>1989</td>
      <td>0.0</td>
      <td>0</td>
      <td>493</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>82599</th>
      <td>1381404</td>
      <td>35000.0</td>
      <td>400151</td>
      <td>3459</td>
      <td>132</td>
      <td>4.0</td>
      <td>1997</td>
      <td>0.0</td>
      <td>0</td>
      <td>478</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>343637</th>
      <td>2371759</td>
      <td>26500.0</td>
      <td>1745626</td>
      <td>2797</td>
      <td>136</td>
      <td>2.0</td>
      <td>1997</td>
      <td>0.0</td>
      <td>0</td>
      <td>2613</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>171006</th>
      <td>1599098</td>
      <td>16000.0</td>
      <td>1501829</td>
      <td>4709</td>
      <td>132</td>
      <td>7.0</td>
      <td>1981</td>
      <td>0.0</td>
      <td>0</td>
      <td>1000</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>246685</th>
      <td>1757795</td>
      <td>9750.0</td>
      <td>1264099</td>
      <td>22728</td>
      <td>132</td>
      <td>2.0</td>
      <td>1000</td>
      <td>0.0</td>
      <td>0</td>
      <td>2009</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 103 columns</p>
</div>



Looks like our data is all in numerical form.

How about we investigate an item from our `column_to_category_dict`?

This will show the mapping from numerical value to category (most likely a string) value.


```python
# Check the UsageBand (measure of bulldozer usage)
for key, value in sorted(column_to_category_dict["UsageBand"].items()): # note: calling sorted() on dictionary.items() sorts the dictionary by keys 
    print(f"{key} -> {value}")
```

    0 -> nan
    1 -> High
    2 -> Low
    3 -> Medium


> **Note:** Categorical values do not necessarily have order. They are strictly a mapping from number to value. In this case, our categorical values are mapped in numerical order. If you feel that the order of a value may influence a model in a negative way (e.g. `1 -> High` is *lower* than `3 -> Medium` but should be *higher*), you may want to look into ordering the values in a particular way or using a different numerical encoding technique such as [one-hot encoding](https://en.wikipedia.org/wiki/One-hot).

And we can do the same for the `state` column values.


```python
# Check the first 10 state column values
for key, value in sorted(column_to_category_dict["state"].items())[:10]:
    print(f"{key} -> {value}")
```

    1 -> Alabama
    2 -> Alaska
    3 -> Arizona
    4 -> Arkansas
    5 -> California
    6 -> Colorado
    7 -> Connecticut
    8 -> Delaware
    9 -> Florida
    10 -> Georgia


How about we check to see all of the missing values have been filled?


```python
# Check total number of missing values
total_missing_values = df_tmp.isna().sum().sum()

if total_missing_values == 0:
    print(f"[INFO] Total missing values: {total_missing_values} - Let's build a model!")
else:
    print(f"[INFO] Uh ohh... total missing values: {total_missing_values} - Perhaps we might have to retrace our steps to fill the values?")
```

    [INFO] Total missing values: 0 - Let's build a model!


### 2.8 Saving our preprocessed data (part 2)

One more step before we train new model!

Let's save our work so far so we could re-import our preprocessed dataset if we wanted to.

We'll save it to the `parquet` format again, this time with a suffix to show we've filled the missing values.


```python
# Save preprocessed data with object values as categories as well as missing values filled
df_tmp.to_parquet(path="TrainAndValid_object_values_as_categories_and_missing_values_filled.parquet",
                  engine="auto")
```


```python
# Read in preprocessed dataset
df_tmp = pd.read_parquet(path="TrainAndValid_object_values_as_categories_and_missing_values_filled.parquet",
                         engine="auto")
```

Does it have any missing values?


```python
# Check total number of missing values
total_missing_values = df_tmp.isna().sum().sum()

if total_missing_values == 0:
    print(f"[INFO] Total missing values: {total_missing_values} - Let's build a model!")
else:
    print(f"[INFO] Uh ohh... total missing values: {total_missing_values} - Perhaps we might have to retrace our steps to fill the values?")
```

    [INFO] Total missing values: 0 - Let's build a model!


### 2.9 Fitting a machine learning model to our preprocessed data

Now all of our data is numeric and there are no missing values, we should be able to fit a machine learning model to it!

Let's reinstantiate our trusty [`sklearn.ensemble.RandomForestRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) model.

Since our dataset has a substantial amount of rows (~400k+), let's first make sure the model will work on a smaller sample of 1000 or so.

> **Note:** It's common practice on machine learning problems to see if your experiments will work on smaller scale problems (e.g. smaller amounts of data) before scaling them up to the full dataset. This practice enables you to try many different kinds of experiments with faster runtimes. The benefit of this is that you can figure out what doesn't work before spending more time on what does.

Our `X` values (features) will be every column except the `SalePrice` column.

And our `y` values (labels) will be the entirety of the `SalePrice` column.


We'll time how long our smaller experiment takes using the [magic function `%%time`](https://jakevdp.github.io/PythonDataScienceHandbook/01.07-timing-and-profiling.html) and placing it at the top of the notebook cell.

> **Note:** You can find out more about the `%%time` magic command by typing `%%time?` (note the question mark on the end) in a notebook cell.



```python
%%time
from sklearn.ensemble import RandomForestRegressor
# Sample 1000 samples with random state 42 for reproducibility
df_tmp_sample_1k = df_tmp.sample(n=1000, random_state=42)

# Instantiate a model
model = RandomForestRegressor(n_jobs=-1) # use -1 to utilise all available processors

# Create features and labels
X_sample_1k = df_tmp_sample_1k.drop("SalePrice", axis=1) # use all columns except SalePrice as X values
y_sample_1k = df_tmp_sample_1k["SalePrice"] # use SalePrice as y values (target variable)

# Fit the model to the sample data
model.fit(X=X_sample_1k, 
          y=y_sample_1k) 
```

    CPU times: user 1.75 s, sys: 164 ms, total: 1.91 s
    Wall time: 4.17 s





<style>#sk-container-id-1 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-1 {
  color: var(--sklearn-color-text);
}

#sk-container-id-1 pre {
  padding: 0;
}

#sk-container-id-1 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-1 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-1 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-1 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-1 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-1 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-1 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-1 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-1 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-1 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-1 label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: start;
  justify-content: space-between;
  gap: 0.5em;
}

#sk-container-id-1 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-1 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-1 div.sk-label label.sk-toggleable__label,
#sk-container-id-1 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-1 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-1 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-1 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-1 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-1 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-1 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-1 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-1 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>RandomForestRegressor(n_jobs=-1)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>RandomForestRegressor</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.ensemble.RandomForestRegressor.html">?<span>Documentation for RandomForestRegressor</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted"><pre>RandomForestRegressor(n_jobs=-1)</pre></div> </div></div></div></div>



How about we score our model?

We can do so using the built-in method `score()`. 

By default, `sklearn.ensemble.RandomForestRegressor` uses [coefficient of determination](https://en.wikipedia.org/wiki/Coefficient_of_determination) ($R^2$ or R-squared) as the evaluation metric (higher is better, with a score of 1.0 being perfect).


```python
# Evaluate the model
model_sample_1k_score = model.score(X=X_sample_1k,
                                    y=y_sample_1k)

print(f"[INFO] Model score on {len(df_tmp_sample_1k)} samples: {model_sample_1k_score}")
```

    [INFO] Model score on 1000 samples: 0.9578537847723126


How about we try our model on the whole dataset?


```python
%%time

# Instantiate model
model = RandomForestRegressor(n_jobs=-1) # note: this could take quite a while depending on your machine (it took ~1.5 minutes on my MacBook Pro M1 Pro with 10 cores)

# Create features and labels with entire dataset
X_all = df_tmp.drop("SalePrice", axis=1)
y_all = df_tmp["SalePrice"]

# Fit the model
model.fit(X=X_all, 
          y=y_all)
```

    CPU times: user 10min 50s, sys: 3.45 s, total: 10min 54s
    Wall time: 1min 28s





<style>#sk-container-id-2 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-2 {
  color: var(--sklearn-color-text);
}

#sk-container-id-2 pre {
  padding: 0;
}

#sk-container-id-2 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-2 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-2 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-2 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-2 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-2 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-2 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-2 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-2 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-2 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-2 label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: start;
  justify-content: space-between;
  gap: 0.5em;
}

#sk-container-id-2 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-2 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-2 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-2 div.sk-label label.sk-toggleable__label,
#sk-container-id-2 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-2 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-2 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-2 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-2 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-2 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-2 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-2 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-2 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-2 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>RandomForestRegressor(n_jobs=-1)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" checked><label for="sk-estimator-id-2" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>RandomForestRegressor</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.ensemble.RandomForestRegressor.html">?<span>Documentation for RandomForestRegressor</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted"><pre>RandomForestRegressor(n_jobs=-1)</pre></div> </div></div></div></div>



Ok, that took a little bit longer than fitting on 1000 samples (but that's too be expected, as many more calculations had to be made).

There's a reason we used `n_jobs=-1` too.

If we stuck with the default of `n_jobs=None` (the same as `n_jobs=1`), it would've taken much longer.

And as we've discussed many times, one of the main goals when starting a machine learning project is to reduce your time between experiments.

How about we score the model trained on all of the data?


```python
# Evaluate the model
model_sample_all_score = model.score(X=X_all,
                                     y=y_all)

print(f"[INFO] Model score on {len(df_tmp)} samples: {model_sample_all_score}")
```

    [INFO] Model score on 412698 samples: 0.987579866067053


### 2.10 A big (but fixable) mistake 

One of the hard things about bugs in machine learning projects is that they are often silent.

For example, our model seems to have fit the data with no issues and then evaluated with a good score.

So what's wrong?

It seems we've stumbled across one of the most common bugs in machine learning and that's **data leakage** (data from the training set leaking into the validation/testing sets).

We've evaluated our model on the same data it was trained on.

This isn't the model's fault either.

It's our fault.

Right back at the start we imported a file called `TrainAndValid.csv`, this file contains both the training and validation data.

And while we preprocessed it to make sure there were no missing values and the samples were all numeric, we never split the data into separate training and validation splits.

The right workflow would've been to train the model on the training split and then evaluate it on the *unseen* and *separate* validation split.

Our evaluation scores above are quite good but they can't necessarily be trusted to be replicated on unseen data (data in the real world) because they've been obtained by evaluating the model on data its already seen during training. 

This would be the equivalent of a final exam at university containing all of the same questions as the practice exam without any changes, you may get a good grade, but does that good grade translate to the real world?

Not to worry, we can fix this!

How?

We can import the training and validation datasets separately via `Train.csv` and `Valid.csv` respectively.

Or we could import `TrainAndValid.csv` and perform the appropriate splits according the original [Kaggle competition page](https://www.kaggle.com/c/bluebook-for-bulldozers/data) (training data includes all samples prior to 2012 and validation data includes samples from January 1 2012 to April 30 2012).

In both methods, we'll have to perform similar preprocessing steps we've done so far.

Except because the validation data is supposed to remain as *unseen* data, we'll only use information from the training set to preprocess the validation set (and not mix the two). 

We'll work on this in the subsequent sections.

The takeaway?

Always (if possible) **create appropriate data splits at the start of a project**.

Because it's one thing to train a machine learning model but if you can't evaluate it properly (on unseen data), how can you know how it'll perform (or may perform) in the real world on new and unseen data?

## 3. Splitting data into the right train/validation sets

The bad news is, we evaluated our model on the same data we trained it on.

The good news is, we get to practice importing and preprocessing our data again. 

This time we'll make sure we've got separate training and validation splits. 

Previously, we used pandas to ensure our data was all numeric and had no missing values. 

And we can still use pandas for things such as creating/altering date-related columns.

But using pandas for all of our data preprocessing can be an issue with larger scale datasets or when new data is introduced. 

How about this time we add Scikit-Learn to the mix and make a reproducible pipeline for our data preprocessing needs?

> **Note:** Scikit-Learn has a fantastic guide on [data transformations](https://scikit-learn.org/1.5/data_transforms.html) and in particular [data preprocessing](https://scikit-learn.org/1.5/modules/preprocessing.html). I'd highly recommend spending an hour or so reading through this documentation, even if it doesn't make a lot of sense to begin with. Rest assured, with practice and experimentation you'll start to get the hang of it.

According to the [Kaggle data page](https://www.kaggle.com/c/bluebook-for-bulldozers/data), the train, validation and test sets are split according to dates.

This makes sense since we're working on a time series problem (using past sale prices to try and predict future sale prices).

Knowing this, randomly splitting our data into train, validation and test sets using something like [`sklearn.model_selection.train_test_split()`](https://scikit-learn.org/1.5/modules/generated/sklearn.model_selection.train_test_split.html) wouldn't work as this would mix samples from different dates in an unintended way.

Instead, we split our data into training, validation and test sets using the date each sample occured.

In our case:

* Training data (`Train.csv`) = all samples up until 2011.
* Validation data (`Valid.csv`) = all samples form January 1, 2012 - April 30, 2012.
* Testing data (`Test.csv`) = all samples from May 1, 2012 - November 2012.

Previously we imported `TrainAndValid.csv` which is a combination of `Train.csv` and `Valid.csv` in one file.

We could split this based on the `saledate` column.

However, we could also import the `Train.csv` and `Valid.csv` files separately (we'll import `Test.csv` later on when we've trained a model).

We'll also import `ValidSolution.csv` which contains the `SalePrice` of `Valid.csv` and make sure we match the columns based on the `SalesID` key.

> **Note:** For more on making good training, validation and test sets, check out the post [How (and why) to create a good validation set](https://www.fast.ai/2017/11/13/validation-sets/) by Rachel Thomas as well as [The importance of a test set](https://www.learnml.io/posts/the-importance-of-a-test-set/) by Daniel Bourke.


```python
# Import train samples (making sure to parse dates and then sort by them)
train_df = pd.read_csv(filepath_or_buffer="bluebook-for-bulldozers/Train.csv",
                       parse_dates=["saledate"],
                       low_memory=False).sort_values(by="saledate", ascending=True)

# Import validation samples (making sure to parse dates and then sort by them)
valid_df = pd.read_csv(filepath_or_buffer="bluebook-for-bulldozers/Valid.csv",
                       parse_dates=["saledate"])

# The ValidSolution.csv contains the SalePrice values for the samples in Valid.csv
valid_solution = pd.read_csv(filepath_or_buffer="bluebook-for-bulldozers/ValidSolution.csv")

# Map valid_solution to valid_df
valid_df["SalePrice"] = valid_df["SalesID"].map(valid_solution.set_index("SalesID")["SalePrice"])

# Make sure valid_df is sorted by saledate still
valid_df = valid_df.sort_values("saledate", ascending=True).reset_index(drop=True)

# How many samples are in each DataFrame?
print(f"[INFO] Number of samples in training DataFrame: {len(train_df)}")
print(f"[INFO] Number of samples in validation DataFrame: {len(valid_df)}")
```

    [INFO] Number of samples in training DataFrame: 401125
    [INFO] Number of samples in validation DataFrame: 11573



```python
# Let's check out the training DataFrame
train_df.sample(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SalesID</th>
      <th>SalePrice</th>
      <th>MachineID</th>
      <th>ModelID</th>
      <th>datasource</th>
      <th>auctioneerID</th>
      <th>YearMade</th>
      <th>MachineHoursCurrentMeter</th>
      <th>UsageBand</th>
      <th>saledate</th>
      <th>...</th>
      <th>Undercarriage_Pad_Width</th>
      <th>Stick_Length</th>
      <th>Thumb</th>
      <th>Pattern_Changer</th>
      <th>Grouser_Type</th>
      <th>Backhoe_Mounting</th>
      <th>Blade_Type</th>
      <th>Travel_Controls</th>
      <th>Differential_Type</th>
      <th>Steering_Controls</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>195642</th>
      <td>1633406</td>
      <td>21000</td>
      <td>1533325</td>
      <td>4604</td>
      <td>132</td>
      <td>2.0</td>
      <td>1999</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2004-08-20</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>221256</th>
      <td>1690941</td>
      <td>28000</td>
      <td>1197871</td>
      <td>11586</td>
      <td>132</td>
      <td>2.0</td>
      <td>1995</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2005-01-30</td>
      <td>...</td>
      <td>20 inch</td>
      <td>None or Unspecified</td>
      <td>None or Unspecified</td>
      <td>None or Unspecified</td>
      <td>Double</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>388935</th>
      <td>4362262</td>
      <td>92500</td>
      <td>1212502</td>
      <td>11989</td>
      <td>172</td>
      <td>1.0</td>
      <td>2006</td>
      <td>6667.0</td>
      <td>High</td>
      <td>2010-09-29</td>
      <td>...</td>
      <td>None or Unspecified</td>
      <td>11' 0"</td>
      <td>None or Unspecified</td>
      <td>None or Unspecified</td>
      <td>Double</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>82613</th>
      <td>1381459</td>
      <td>22000</td>
      <td>1294010</td>
      <td>3459</td>
      <td>132</td>
      <td>2.0</td>
      <td>1999</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2004-03-25</td>
      <td>...</td>
      <td>18 inch</td>
      <td>None or Unspecified</td>
      <td>None or Unspecified</td>
      <td>None or Unspecified</td>
      <td>Double</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>339934</th>
      <td>2355520</td>
      <td>47000</td>
      <td>1643140</td>
      <td>1893</td>
      <td>136</td>
      <td>1.0</td>
      <td>2000</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>2009-05-13</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>None or Unspecified</td>
      <td>PAT</td>
      <td>None or Unspecified</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 53 columns</p>
</div>




```python
# And how about the validation DataFrame?
valid_df.sample(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SalesID</th>
      <th>MachineID</th>
      <th>ModelID</th>
      <th>datasource</th>
      <th>auctioneerID</th>
      <th>YearMade</th>
      <th>MachineHoursCurrentMeter</th>
      <th>UsageBand</th>
      <th>saledate</th>
      <th>fiModelDesc</th>
      <th>...</th>
      <th>Stick_Length</th>
      <th>Thumb</th>
      <th>Pattern_Changer</th>
      <th>Grouser_Type</th>
      <th>Backhoe_Mounting</th>
      <th>Blade_Type</th>
      <th>Travel_Controls</th>
      <th>Differential_Type</th>
      <th>Steering_Controls</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6841</th>
      <td>6285515</td>
      <td>1817678</td>
      <td>7939</td>
      <td>149</td>
      <td>1</td>
      <td>2002</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2012-03-20</td>
      <td>110</td>
      <td>...</td>
      <td>None or Unspecified</td>
      <td>None or Unspecified</td>
      <td>None or Unspecified</td>
      <td>Double</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>14000.0</td>
    </tr>
    <tr>
      <th>933</th>
      <td>6304375</td>
      <td>1828647</td>
      <td>495</td>
      <td>149</td>
      <td>2</td>
      <td>1000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2012-01-28</td>
      <td>PC400LC6</td>
      <td>...</td>
      <td>None or Unspecified</td>
      <td>None or Unspecified</td>
      <td>None or Unspecified</td>
      <td>Double</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>27000.0</td>
    </tr>
    <tr>
      <th>5157</th>
      <td>4318165</td>
      <td>2308308</td>
      <td>22070</td>
      <td>172</td>
      <td>1</td>
      <td>2008</td>
      <td>15.0</td>
      <td>Low</td>
      <td>2012-03-01</td>
      <td>315</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>19500.0</td>
    </tr>
    <tr>
      <th>3697</th>
      <td>6276669</td>
      <td>801310</td>
      <td>1580</td>
      <td>149</td>
      <td>1</td>
      <td>2003</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2012-02-13</td>
      <td>D5GLGP</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>None or Unspecified</td>
      <td>PAT</td>
      <td>None or Unspecified</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>35000.0</td>
    </tr>
    <tr>
      <th>6176</th>
      <td>4275972</td>
      <td>2208545</td>
      <td>3362</td>
      <td>172</td>
      <td>1</td>
      <td>1993</td>
      <td>10263.0</td>
      <td>Medium</td>
      <td>2012-03-13</td>
      <td>140G</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>76000.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 53 columns</p>
</div>



We've now got separate training and validation datasets imported.

In a previous section, we created a function to decompose the `saledate` column into multiple features such as `saleYear`, `saleMonth`, `saleDay` and more.

Let's now replicate that function here and apply it to our `train_df` and `valid_df`.


```python
# Make a function to add date columns
def add_datetime_features_to_df(df, date_column="saledate"):
    # Add datetime parameters for saledate
    df["saleYear"] = df[date_column].dt.year
    df["saleMonth"] = df[date_column].dt.month
    df["saleDay"] = df[date_column].dt.day
    df["saleDayofweek"] = df[date_column].dt.dayofweek
    df["saleDayofyear"] = df[date_column].dt.dayofyear

    # Drop original saledate column
    df.drop("saledate", axis=1, inplace=True)

    return df

train_df = add_datetime_features_to_df(df=train_df)
valid_df = add_datetime_features_to_df(df=valid_df)
```


```python
# Display the last 5 columns (the recently added datetime breakdowns)
train_df.iloc[:, -5:].sample(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>saleYear</th>
      <th>saleMonth</th>
      <th>saleDay</th>
      <th>saleDayofweek</th>
      <th>saleDayofyear</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>108598</th>
      <td>2001</td>
      <td>6</td>
      <td>15</td>
      <td>4</td>
      <td>166</td>
    </tr>
    <tr>
      <th>174277</th>
      <td>1995</td>
      <td>12</td>
      <td>16</td>
      <td>5</td>
      <td>350</td>
    </tr>
    <tr>
      <th>83496</th>
      <td>2002</td>
      <td>12</td>
      <td>11</td>
      <td>2</td>
      <td>345</td>
    </tr>
    <tr>
      <th>395898</th>
      <td>2011</td>
      <td>6</td>
      <td>22</td>
      <td>2</td>
      <td>173</td>
    </tr>
    <tr>
      <th>251275</th>
      <td>2001</td>
      <td>10</td>
      <td>27</td>
      <td>5</td>
      <td>300</td>
    </tr>
  </tbody>
</table>
</div>



### 3.1 Trying to fit a model on our training data

I'm a big fan of trying to fit a model on your dataset as early as possible.

If it works, you'll have to inspect and check its results.

And if it doesn't work, you'll get some insights into what you may have to do to your dataset to prepare it.

Let's turn our DataFrames into features (`X`) by dropping the `SalePrice` column (this is the value we're trying to predict) and labels (`y`) by extracting the `SalePrice` column.

Then we'll create a model using `sklearn.ensemble.RandomForestRegressor` and finally we'll try to fit it to only the training data.


```python
# Split training data into features and labels
X_train = train_df.drop("SalePrice", axis=1)
y_train = train_df["SalePrice"]

# Split validation data into features and labels
X_valid = valid_df.drop("SalePrice", axis=1)
y_valid = valid_df["SalePrice"]

# Create a model
model = RandomForestRegressor(n_jobs=-1)

# Fit a model to the training data only
model.fit(X=X_train,
          y=y_train)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    /var/folders/vd/z23lv5250ld1c15z5n45gcbh0000gn/T/ipykernel_33450/150598518.py in ?()
          9 # Create a model
         10 model = RandomForestRegressor(n_jobs=-1)
         11 
         12 # Fit a model to the training data only
    ---> 13 model.fit(X=X_train,
         14           y=y_train)


    ~/anaconda3/lib/python3.10/site-packages/sklearn/base.py in ?(estimator, *args, **kwargs)
       1385                 skip_parameter_validation=(
       1386                     prefer_skip_nested_validation or global_skip_validation
       1387                 )
       1388             ):
    -> 1389                 return fit_method(estimator, *args, **kwargs)
    

    ~/anaconda3/lib/python3.10/site-packages/sklearn/ensemble/_forest.py in ?(self, X, y, sample_weight)
        356         # Validate or convert input data
        357         if issparse(y):
        358             raise ValueError("sparse multilabel-indicator for y is not supported.")
        359 
    --> 360         X, y = validate_data(
        361             self,
        362             X,
        363             y,


    ~/anaconda3/lib/python3.10/site-packages/sklearn/utils/validation.py in ?(_estimator, X, y, reset, validate_separately, skip_check_array, **check_params)
       2957             if "estimator" not in check_y_params:
       2958                 check_y_params = {**default_check_params, **check_y_params}
       2959             y = check_array(y, input_name="y", **check_y_params)
       2960         else:
    -> 2961             X, y = check_X_y(X, y, **check_params)
       2962         out = X, y
       2963 
       2964     if not no_val_X and check_params.get("ensure_2d", True):


    ~/anaconda3/lib/python3.10/site-packages/sklearn/utils/validation.py in ?(X, y, accept_sparse, accept_large_sparse, dtype, order, copy, force_writeable, force_all_finite, ensure_all_finite, ensure_2d, allow_nd, multi_output, ensure_min_samples, ensure_min_features, y_numeric, estimator)
       1366         )
       1367 
       1368     ensure_all_finite = _deprecate_force_all_finite(force_all_finite, ensure_all_finite)
       1369 
    -> 1370     X = check_array(
       1371         X,
       1372         accept_sparse=accept_sparse,
       1373         accept_large_sparse=accept_large_sparse,


    ~/anaconda3/lib/python3.10/site-packages/sklearn/utils/validation.py in ?(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_writeable, force_all_finite, ensure_all_finite, ensure_non_negative, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator, input_name)
       1052                         )
       1053                     array = xp.astype(array, dtype, copy=False)
       1054                 else:
       1055                     array = _asarray_with_order(array, order=order, dtype=dtype, xp=xp)
    -> 1056             except ComplexWarning as complex_warning:
       1057                 raise ValueError(
       1058                     "Complex data not supported\n{}\n".format(array)
       1059                 ) from complex_warning


    ~/anaconda3/lib/python3.10/site-packages/sklearn/utils/_array_api.py in ?(array, dtype, order, copy, xp, device)
        835         # Use NumPy API to support order
        836         if copy is True:
        837             array = numpy.array(array, order=order, dtype=dtype)
        838         else:
    --> 839             array = numpy.asarray(array, order=order, dtype=dtype)
        840 
        841         # At this point array is a NumPy ndarray. We convert it to an array
        842         # container that is consistent with the input's namespace.


    ~/anaconda3/lib/python3.10/site-packages/pandas/core/generic.py in ?(self, dtype)
       2082     def __array__(self, dtype: npt.DTypeLike | None = None) -> np.ndarray:
       2083         values = self._values
    -> 2084         arr = np.asarray(values, dtype=dtype)
       2085         if (
       2086             astype_is_view(values.dtype, arr.dtype)
       2087             and using_copy_on_write()


    ValueError: could not convert string to float: 'Medium'


Oh no!

We run into the error:

> ValueError: could not convert string to float: 'Medium'

Hmm... 

Where have we seen this error before?

It looks like since we re-imported our training dataset (from `Train.csv`) its no longer all numerical (hence the `ValueError` above).

Not to worry, we can fix this!

### 3.2 Encoding categorical features as numbers using Scikit-Learn

We've preprocessed our data previously with pandas.

And while this is a viable approach, how about we practice using another method?

This time we'll use Scikit-Learn's built-in preprocessing methods. 


Scikit-Learn has many built-in helpful and well tested methods for preparing data. 

You can also string together many of these methods and create a [reusable pipeline](https://scikit-learn.org/1.5/modules/generated/sklearn.pipeline.Pipeline.html) (you can think of this pipeline as plumbing for data).

To preprocess our data with Scikit-Learn, we'll first define the numerical and categorical features of our dataset.


```python
# Define numerical and categorical features
numerical_features = [label for label, content in X_train.items() if pd.api.types.is_numeric_dtype(content)]
categorical_features = [label for label, content in X_train.items() if not pd.api.types.is_numeric_dtype(content)]

print(f"[INFO] Numeric features: {numerical_features}")
print(f"[INFO] Categorical features: {categorical_features[:10]}...")
```

    [INFO] Numeric features: ['SalesID', 'MachineID', 'ModelID', 'datasource', 'auctioneerID', 'YearMade', 'MachineHoursCurrentMeter', 'saleYear', 'saleMonth', 'saleDay', 'saleDayofweek', 'saleDayofyear']
    [INFO] Categorical features: ['UsageBand', 'fiModelDesc', 'fiBaseModel', 'fiSecondaryDesc', 'fiModelSeries', 'fiModelDescriptor', 'ProductSize', 'fiProductClassDesc', 'state', 'ProductGroup']...


Nice!

We define our different feature types so we can use different preprocessing methods on each type.

Scikit-Learn has many built-in methods for preprocessing data under the [`sklearn.preprocessing` module](https://scikit-learn.org/stable/api/sklearn.preprocessing.html#).

And I'd encourage you to spend some time reading the [preprocessing data section of the Scikit-Learn user guide](https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing) for more details.

For now, let's focus on turning our categorical features into numbers (from object/string datatype to numeric datatype).

The practice of turning non-numerical features into numerical features is often referred to as [**encoding**](https://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features).

There are several encoders available for different use cases.

| Encoder | Description | Use case | For use on |
| ----- | ----- | ----- | ----- |
| [`LabelEncoder`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html#sklearn.preprocessing.LabelEncoder) | Encode target labels with values between 0 and n_classes-1. | Useful for turning classification target values into numeric representations. | Target labels. |
| [`OneHotEncoder`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html#onehotencoder) | Encode categorical features as a [one-hot numeric array](https://en.wikipedia.org/wiki/One-hot). | Turns every positive class of a unique category into a 1 and every negative class into a 0. | Categorical variables/features. |
| [`OrdinalEncoder`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#ordinalencoder) | Encode categorical features as an integer array. | Turn unique categorical values into a range of integers, for example, 0 maps to 'cat', 1 maps to 'dog', etc. | Categorical variables/features. |
| [`TargetEncoder`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html#targetencoder) | Encode regression and classification targets into a shrunk estimate of the average target values for observations of the category. | Useful for converting targets into a certain range of values. | Target variables. |

For our case, we're going to start with `OrdinalEncoder`.

When transforming/encoding values with Scikit-Learn, the steps as follows:

1. Instantiate an encoder, for example, `sklearn.preprocessing.OrdinalEncoder`.
2. Use the [`sklearn.preprocessing.OrdinalEncoder.fit`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#sklearn.preprocessing.OrdinalEncoder.fit) method on the **training** data (this helps the encoder learn a mapping of categorical to numeric values).
3. Use the [`sklearn.preprocessing.OrdinalEncoder.transform`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#sklearn.preprocessing.OrdinalEncoder.transform) method on the **training** data to apply the learned mapping from categorical to numeric values.
    * **Note:** The [`sklearn.preprocessing.OrdinalEncoder.fit_transform`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#sklearn.preprocessing.OrdinalEncoder.fit_transform) method combines steps 1 & 2 into a single method.
4. Apply the learned mapping to subsequent datasets such as **validation** and **test** splits using `sklearn.preprocessing.OrdinalEncoder.transform` only.

Notice how the `fit` and `fit_transform` methods were reserved for the **training dataset only**.

This is because in practice the validation and testing datasets are meant to be unseen, meaning only information from the training dataset should be used to preprocess the validation/test datasets.

In short:

1. Instantiate an encoder such as `sklearn.preprocessing.OrdinalEncoder`.
2. Fit the encoder to and transform the training dataset categorical variables/features with `sklearn.preprocessing.OrdinalEncoder.fit_transform`.
3. Transform categorical variables/features from subsequent datasets such as the validation and test datasets with the learned encoding from step 2 using `sklearn.preprocessing.OridinalEncoder.transform`. 
    * **Note:** Notice the use of the `transform` method on validation/test datasets rather than `fit_transform`.

Let's do it!

We'll use the `OrdinalEncoder` class to fill any missing values with `np.nan` (`NaN`).

We'll also make sure to only use the `OrdinalEncoder` on the categorical features of our DataFrame.

Finally, the `OrdinalEncoder` expects all input variables to be of the same type (e.g. either numeric only or string only) so we'll make sure all the input variables are strings only using [`pandas.DataFrame.astype(str)`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.astype.html).


```python
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor

# Assuming X_train and X_valid are your Pandas DataFrames
# Assuming categorical_features and numerical_features lists are defined

# --- Impute Numerical Features ---
numerical_imputer = SimpleImputer(strategy='median')
X_train[numerical_features] = numerical_imputer.fit_transform(X_train[numerical_features])
X_valid[numerical_features] = numerical_imputer.transform(X_valid[numerical_features])

# --- Impute Categorical Features ---
categorical_imputer = SimpleImputer(strategy='most_frequent')
X_train[categorical_features] = categorical_imputer.fit_transform(X_train[categorical_features].astype(str))
X_valid[categorical_features] = categorical_imputer.transform(X_valid[categorical_features].astype(str))

# --- Ordinal Encode Categorical Features ---
ordinal_encoder = OrdinalEncoder(categories="auto",
                                 handle_unknown="use_encoded_value",
                                 unknown_value=np.nan)

X_train_preprocessed = X_train.copy()
X_train_preprocessed[categorical_features] = ordinal_encoder.fit_transform(X_train_preprocessed[categorical_features])

X_valid_preprocessed = X_valid.copy()
X_valid_preprocessed[categorical_features] = ordinal_encoder.transform(X_valid_preprocessed[categorical_features])

# --- Final Handling of Remaining NaN Values (CRUCIAL for RandomForestRegressor) ---
X_train_preprocessed.fillna(-1, inplace=True)
X_valid_preprocessed.fillna(-1, inplace=True)


```


```python
X_train_preprocessed[categorical_features].isna().sum().sort_values(ascending=False)[:10], X_train_preprocessed[numerical_features].isna().sum().sort_values(ascending=False)[:10]
```




    (UsageBand         0
     fiModelDesc       0
     Pushblock         0
     Ripper            0
     Scarifier         0
     Tip_Control       0
     Tire_Size         0
     Coupler           0
     Coupler_System    0
     Grouser_Tracks    0
     dtype: int64,
     SalesID                     0
     MachineID                   0
     ModelID                     0
     datasource                  0
     auctioneerID                0
     YearMade                    0
     MachineHoursCurrentMeter    0
     saleYear                    0
     saleMonth                   0
     saleDay                     0
     dtype: int64)




```python
X_valid_preprocessed[categorical_features].isna().sum().sort_values(ascending=False)[:10], X_valid_preprocessed[numerical_features].isna().sum().sort_values(ascending=False)[:10]
```




    (UsageBand         0
     fiModelDesc       0
     Pushblock         0
     Ripper            0
     Scarifier         0
     Tip_Control       0
     Tire_Size         0
     Coupler           0
     Coupler_System    0
     Grouser_Tracks    0
     dtype: int64,
     SalesID                     0
     MachineID                   0
     ModelID                     0
     datasource                  0
     auctioneerID                0
     YearMade                    0
     MachineHoursCurrentMeter    0
     saleYear                    0
     saleMonth                   0
     saleDay                     0
     dtype: int64)



### 3.3 Fitting a model to our preprocessed training data

We've used Scikit-Learn to convert the categorical data in our training and validation sets into numbers.

We have imputed numerical features to handle missing numerical values.

We can use  `sklearn.ensemble.RandomForestRegressor` 

Let's try it out on our `X_train_preprocessed` DataFrame.



```python
# --- Instantiate and Fit the Random Forest Regressor ---
model = RandomForestRegressor(n_jobs=-1)
model.fit(X=X_train_preprocessed, y=y_train)

```




<style>#sk-container-id-3 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-3 {
  color: var(--sklearn-color-text);
}

#sk-container-id-3 pre {
  padding: 0;
}

#sk-container-id-3 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-3 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-3 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-3 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-3 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-3 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-3 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-3 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-3 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-3 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-3 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-3 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-3 label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: start;
  justify-content: space-between;
  gap: 0.5em;
}

#sk-container-id-3 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-3 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-3 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-3 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-3 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-3 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-3 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-3 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-3 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-3 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-3 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-3 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-3 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-3 div.sk-label label.sk-toggleable__label,
#sk-container-id-3 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-3 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-3 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-3 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-3 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-3 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-3 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-3 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-3 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-3 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-3 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-3 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-3 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-3" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>RandomForestRegressor(n_jobs=-1)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" checked><label for="sk-estimator-id-3" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>RandomForestRegressor</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.ensemble.RandomForestRegressor.html">?<span>Documentation for RandomForestRegressor</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted"><pre>RandomForestRegressor(n_jobs=-1)</pre></div> </div></div></div></div>




```python
%%time

# Check model performance on the validation set
model.score(X=X_valid_preprocessed,
            y=y_valid)
```

    CPU times: user 538 ms, sys: 525 ms, total: 1.06 s
    Wall time: 361 ms





    0.8695671042527193



Excellent!

Now you might be wondering why this score ($R^2$ or R-squared by default) is lower than the previous score of ~0.9875.

That's because this score is based on a model that has only seen the training data and is being evaluated on an unseen dataset (training on `Train.csv`, evaluating on `Valid.csv`).

Our previous score was from a model that had all of the evaluation samples in the training data (training and evaluating on `TrainAndValid.csv`).

So in practice, we would consider the most recent score as a much more reliable metric of how well our model might perform on future unseen data.

Just for fun, let's see how our model scores on the training dataset.


```python

# Now you can use the model to predict on X_valid_preprocessed
predictions = model.predict(X_valid_preprocessed)
```

Now, what if we wanted to retrieve the original categorical values?

We can do using the [`OrdinalEncoder.categories_` attribute](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#sklearn.preprocessing.OrdinalEncoder.fit_transform).

This will return the categories of each feature found during `fit` (or during `fit_transform`), the categories will be in the order of the features seen (same order as the columns of the DataFrame).



```python
# Let's inspect the first three categories
ordinal_encoder.categories_[:3]
```




    [array(['High', 'Low', 'Medium', 'nan'], dtype=object),
     array(['100C', '104', '1066', ..., 'ZX800LC', 'ZX80LCK', 'ZX850H'],
           dtype=object),
     array(['10', '100', '104', ..., 'ZX80', 'ZX800', 'ZX850'], dtype=object)]



Since these come in the order of the features seen, we can create a mapping of these using the categorical column names of our DataFrame.


```python
# Create a dictionary of dictionaries mapping column names and their variables to their numerical encoding
column_to_category_mapping = {}

for column_name, category_values in zip(categorical_features, ordinal_encoder.categories_):
    int_to_category = {i: category for i, category in enumerate(category_values)}
    column_to_category_mapping[column_name] = int_to_category

# Inspect an example column name to category mapping
column_to_category_mapping["UsageBand"]
```




    {0: 'High', 1: 'Low', 2: 'Medium', 3: 'nan'}



We can also reverse our `OrdinalEncoder` values with the [`inverse_transform()`](https://scikit-learn.org/1.5/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#sklearn.preprocessing.OrdinalEncoder.inverse_transform) method.

This is helpful for reversing a preprocessing step or viewing the original data again if necessary.


```python

```

## 4. Building an evaluation function

Evaluating a machine learning model is just as important as training one.

And so because of this, let's create an evaluation function to make evaluation faster and reproducible.

According to Kaggle for the Bluebook for Bulldozers competition, [the evaluation function](https://www.kaggle.com/c/bluebook-for-bulldozers/overview/evaluation) they use is root mean squared log error (RMSLE).

$$ \text{RMSLE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} \left( \log(1 + \hat{y}_i) - \log(1 + y_i) \right)^2} $$

Where:

* $ \hat{y}_i $ is the predicted value,  
* $ y_i $ is the actual value,  
* $ n $ is the number of observations.

Contrast this with mean absolute error (MAE), another common regression metric.

$$ \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} \left| \hat{y}_i - y_i \right| $$

With RMSLE, the relative error is more meaningful than the absolute error. You care more about ratios than absolute errors. For example, being off by $100 on a $1000 prediction (10% error) is more significant than being off by $100 on a $10,000 prediction (1% error). RMSLE is sensitive to large percentage errors.

Where as with MAE, is more about exact differences, a $100 prediction error is weighted the same regardless of the actual value.

In each of case, a lower value (closer to 0) is better.

For any problem, it's important to define the evaluation metric you're going to try and improve on.

In our case, let's create a function that calculates multiple evaluation metrics.

Namely, we'll use:

* MAE (mean absolute error) via [`sklearn.metrics.mean_absolute_error`](https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.mean_absolute_error.html) - lower is better.
* RMSLE (root mean squared log error) via [`sklearn.metrics.root_mean_squared_log_error`](https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.root_mean_squared_log_error.html) - lower is better.
* $R^2$ (R-squared or coefficient of determination) via the [`score` method](https://scikit-learn.org/dev/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor.score) - higher is better.

For MAE and RMSLE we'll be comparing the model's predictions to the truth labels.

We can get an array of predicted values from our model using [`model.predict(X=features_to_predict_on)`](https://scikit-learn.org/dev/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor.predict). 


```python
pip install --upgrade scikit-learn
```

    Requirement already satisfied: scikit-learn in /Users/adityachauhan/anaconda3/lib/python3.10/site-packages (1.2.1)
    Collecting scikit-learn
      Downloading scikit_learn-1.6.1-cp310-cp310-macosx_12_0_arm64.whl (11.1 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m11.1/11.1 MB[0m [31m3.8 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    Requirement already satisfied: threadpoolctl>=3.1.0 in /Users/adityachauhan/anaconda3/lib/python3.10/site-packages (from scikit-learn) (3.5.0)
    Requirement already satisfied: numpy>=1.19.5 in /Users/adityachauhan/anaconda3/lib/python3.10/site-packages (from scikit-learn) (1.26.4)
    Requirement already satisfied: scipy>=1.6.0 in /Users/adityachauhan/anaconda3/lib/python3.10/site-packages (from scikit-learn) (1.10.0)
    Requirement already satisfied: joblib>=1.2.0 in /Users/adityachauhan/anaconda3/lib/python3.10/site-packages (from scikit-learn) (1.4.2)
    Installing collected packages: scikit-learn
      Attempting uninstall: scikit-learn
        Found existing installation: scikit-learn 1.2.1
        Uninstalling scikit-learn-1.2.1:
          Successfully uninstalled scikit-learn-1.2.1
    Successfully installed scikit-learn-1.6.1
    Note: you may need to restart the kernel to use updated packages.



```python

```


```python
# Create evaluation function (the competition uses Root Mean Square Log Error)
from sklearn.metrics import mean_absolute_error, root_mean_squared_log_error

# Create function to evaluate our model
def show_scores(model, 
                train_features=X_train_preprocessed,
                train_labels=y_train,
                valid_features=X_valid_preprocessed,
                valid_labels=y_valid):
    
    # Make predictions on train and validation features
    train_preds = model.predict(X=train_features)
    val_preds = model.predict(X=valid_features)

    # Create a scores dictionary of different evaluation metrics
    scores = {"Training MAE": mean_absolute_error(y_true=train_labels, 
                                                  y_pred=train_preds),
              "Valid MAE": mean_absolute_error(y_true=valid_labels, 
                                               y_pred=val_preds),
              "Training RMSLE": root_mean_squared_log_error(y_true=train_labels, 
                                                            y_pred=train_preds),
              "Valid RMSLE": root_mean_squared_log_error(y_true=valid_labels, 
                                                         y_pred=val_preds),
              "Training R^2": model.score(X=train_features, 
                                          y=train_labels),
              "Valid R^2": model.score(X=valid_features, 
                                       y=valid_labels)}
    return scores
```

How about we test it out?


```python
# Try our model scoring function out
model_scores = show_scores(model=model)
model_scores
```




    {'Training MAE': 1585.9736886257404,
     'Valid MAE': 6188.421701373887,
     'Training RMSLE': 0.0850880948979466,
     'Valid RMSLE': 0.257476337715932,
     'Training R^2': 0.987444742393552,
     'Valid R^2': 0.8695671042527193}



## 5. Tuning our model's hyperparameters

Hyperparameters are the settings we can change on our model.

And tuning hyperparameters on a given model can often alter its performance on a given dataset.

Ideally, changing hyperparameters would lead to better results.

However, it's often hard to know what hyperparameter changes would improve a model ahead of time.

So what we can do is run several experiments across various different hyperparameter settings and record which lead to the best results.

### 5.1 Making our modelling experiments faster (to speed up hyperparameter tuning)

Because of the size of our dataset (~400,000 rows), retraining an entire model (about 1-1.5 minutes on my MacBook Pro M1 Pro) for each new set of hyperparameters would take far too long to continuing experimenting as fast as we want to.

So what we'll do is take a sample of the training set and tune the hyperparameters on that before training a larger model.

> **Note:** If you're experiments are taking longer than 10-seconds (or far longer than what you can interact with), you should be trying to speed things up. You can speed experiments up by sampling less data, using a faster computer or using a smaller model.

We can take a artificial sample of the training set by altering the number of samples seen by each `n_estimator` (an `n_estimator` is a [decision tree](https://en.wikipedia.org/wiki/Decision_tree_learning) a random forest will create during training, more trees generally leads to better performance but sacrifices compute time) in [`sklearn.ensemble.RandomForestRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) using the `max_samples` parameter.

For example, setting `max_samples` to 10,000 means every `n_estimator` (default 100) in our `RandomForestRegressor` will only see 10,000 random samples from our DataFrame instead of the entire ~400,000.

In other words, we'll be looking at 40x less samples which means we should get faster computation speeds but we should also  expect our results to worsen (because the model has less samples to learn patterns from).

Let's see if reducing the number samples speeds up our modelling time.


```python
%%time

# Change max samples in RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, # this is the default
                              n_jobs=-1,
                              max_samples=10000) # each estimator sees max_samples (the default is to see all available samples)

# Cutting down the max number of samples each tree can see improves training time
model.fit(X_train_preprocessed, 
          y_train)
```

    CPU times: user 19 s, sys: 284 ms, total: 19.2 s
    Wall time: 2.92 s





<style>#sk-container-id-4 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-4 {
  color: var(--sklearn-color-text);
}

#sk-container-id-4 pre {
  padding: 0;
}

#sk-container-id-4 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-4 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-4 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-4 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-4 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-4 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-4 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-4 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-4 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-4 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-4 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-4 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-4 label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: start;
  justify-content: space-between;
  gap: 0.5em;
}

#sk-container-id-4 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-4 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-4 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-4 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-4 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-4 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-4 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-4 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-4 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-4 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-4 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-4 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-4 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-4 div.sk-label label.sk-toggleable__label,
#sk-container-id-4 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-4 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-4 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-4 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-4 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-4 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-4 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-4 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-4 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-4 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-4 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-4 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-4 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-4" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>RandomForestRegressor(max_samples=10000, n_jobs=-1)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" checked><label for="sk-estimator-id-4" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>RandomForestRegressor</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.ensemble.RandomForestRegressor.html">?<span>Documentation for RandomForestRegressor</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted"><pre>RandomForestRegressor(max_samples=10000, n_jobs=-1)</pre></div> </div></div></div></div>



Nice! That worked much faster than training on the whole dataset.

Let's evaluate our model with our `show_scores` function.


```python
# Get evaluation metrics from reduced sample model
base_model_scores = show_scores(model=model)
base_model_scores
```




    {'Training MAE': 5586.866608563413,
     'Valid MAE': 7174.828474898471,
     'Training RMSLE': 0.2598200410876938,
     'Valid RMSLE': 0.29619048676582604,
     'Training R^2': 0.8593753062077095,
     'Valid R^2': 0.8307591573506167}



Excellent! Even though our new model saw far less data than the previous model, it still looks to be performing quite well.

With this faster model, we can start to run a series of different hyperparameter experiments.

### 5.2 Hyperparameter tuning with RandomizedSearchCV

The goal of hyperparameter tuning is to values for our model's settings which lead to better results.

We could sit there and do this by hand, adjusting parameters on `sklearn.ensemble.RandomForestRegressor` such as `n_estimators`, `max_depth`, `min_samples_split` and more.

However, this would quite tedious.

Instead, we can define a dictionary of hyperparametmer settings in the form `{"hyperparamter_name": [values_to_test]}` and then use [`sklearn.model_selection.RandomizedSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#randomizedsearchcv) (randomly search for best combination of hyperparameters) or [`sklearn.model_selection.GridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#gridsearchcv) (exhaustively search for best combination of hyperparameters) to go through all of these settings for us on a given model and dataset and then record which perform best.

A general workflow is to start with a large number and wide range of potential settings and use `RandomizedSearchCV` to search across these randomly for a limited number of iterations (e.g. `n_iter=10`).

And then take the best results and narrow the search space down before exhaustively search for the best hyperparameters with `GridSearchCV`.

Let's start trying to find better hyperparameters by:

1. Define a dictionary of hyperparameter values for our `RandomForestRegressor` model. We'll keep `max_samples=10000` so our experiments run faster.
2. Setup an instance of `RandomizedSearchCV` to explore the parameter values defined in step 1. We can adjust how many sets of hyperparameters our model tries using the `n_iter` parameter as well as how many times our model performs cross-validation using the `cv` parameter. For example, setting `n_iter=20` and `cv=3` means there will be 3 cross-validation folds for each of the 20 different combinations of hyperparameters, a total of 60 (3*20) experiments.
3. Fit the instance of `RandomizedSearchCV` to the data. This will automatically go through the defined number of iterations and record the results for each. The best model gets loaded at the end.

> **Note:** You can read more about the [tuning of hyperparameters of an esimator/model in the Scikit-Learn user guide](https://scikit-learn.org/stable/modules/grid_search.html#tuning-the-hyper-parameters-of-an-estimator). 


```python
%%time

from sklearn.model_selection import RandomizedSearchCV

# 1. Define a dictionary with different values for RandomForestRegressor hyperparameters
# See documatation for potential different values - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html 
rf_grid = {"n_estimators": np.arange(10, 200, 10),
           "max_depth": [None, 10, 20],
           "min_samples_split": np.arange(2, 10, 1), # min_samples_split must be an int in the range [2, inf) or a float in the range (0.0, 1.0]
           "min_samples_leaf": np.arange(1, 10, 1),
           "max_features": [0.5, 1.0, "sqrt"], # Note: "max_features='auto'" is equivalent to "max_features=1.0", as of Scikit-Learn version 1.1
           "max_samples": [10000]}

# 2. Setup instance of RandomizedSearchCV to explore different parameters 
rs_model = RandomizedSearchCV(estimator=RandomForestRegressor(), # can pass new model instance directly, all settings will be taken from the rf_grid
                              param_distributions=rf_grid,
                              n_iter=20,
                            #   scoring="neg_root_mean_squared_log_error", # want to optimize for RMSLE, though sometimes optimizing for the default metric (R^2) can lead to just as good results all round
                              cv=3,
                              verbose=3) # control how much output gets produced, higher number = more output

# 3. Fit the model using a series of different hyperparameter values
rs_model.fit(X=X_train_preprocessed, 
             y=y_train)
```

    Fitting 3 folds for each of 20 candidates, totalling 60 fits
    [CV 1/3] END max_depth=10, max_features=sqrt, max_samples=10000, min_samples_leaf=8, min_samples_split=7, n_estimators=140;, score=0.488 total time=   2.5s
    [CV 2/3] END max_depth=10, max_features=sqrt, max_samples=10000, min_samples_leaf=8, min_samples_split=7, n_estimators=140;, score=0.658 total time=   2.4s
    [CV 3/3] END max_depth=10, max_features=sqrt, max_samples=10000, min_samples_leaf=8, min_samples_split=7, n_estimators=140;, score=0.626 total time=   2.4s
    [CV 1/3] END max_depth=20, max_features=1.0, max_samples=10000, min_samples_leaf=3, min_samples_split=5, n_estimators=20;, score=0.535 total time=   2.3s
    [CV 2/3] END max_depth=20, max_features=1.0, max_samples=10000, min_samples_leaf=3, min_samples_split=5, n_estimators=20;, score=0.754 total time=   2.2s
    [CV 3/3] END max_depth=20, max_features=1.0, max_samples=10000, min_samples_leaf=3, min_samples_split=5, n_estimators=20;, score=0.611 total time=   2.1s
    [CV 1/3] END max_depth=10, max_features=1.0, max_samples=10000, min_samples_leaf=9, min_samples_split=6, n_estimators=150;, score=0.535 total time=  11.2s
    [CV 2/3] END max_depth=10, max_features=1.0, max_samples=10000, min_samples_leaf=9, min_samples_split=6, n_estimators=150;, score=0.718 total time=  11.2s
    [CV 3/3] END max_depth=10, max_features=1.0, max_samples=10000, min_samples_leaf=9, min_samples_split=6, n_estimators=150;, score=0.594 total time=  11.0s
    [CV 1/3] END max_depth=20, max_features=0.5, max_samples=10000, min_samples_leaf=2, min_samples_split=5, n_estimators=120;, score=0.547 total time=   7.9s
    [CV 2/3] END max_depth=20, max_features=0.5, max_samples=10000, min_samples_leaf=2, min_samples_split=5, n_estimators=120;, score=0.767 total time=   7.7s
    [CV 3/3] END max_depth=20, max_features=0.5, max_samples=10000, min_samples_leaf=2, min_samples_split=5, n_estimators=120;, score=0.638 total time=   7.3s
    [CV 1/3] END max_depth=20, max_features=0.5, max_samples=10000, min_samples_leaf=1, min_samples_split=4, n_estimators=100;, score=0.551 total time=   6.6s
    [CV 2/3] END max_depth=20, max_features=0.5, max_samples=10000, min_samples_leaf=1, min_samples_split=4, n_estimators=100;, score=0.767 total time=   6.7s
    [CV 3/3] END max_depth=20, max_features=0.5, max_samples=10000, min_samples_leaf=1, min_samples_split=4, n_estimators=100;, score=0.628 total time=   6.5s
    [CV 1/3] END max_depth=None, max_features=1.0, max_samples=10000, min_samples_leaf=9, min_samples_split=8, n_estimators=140;, score=0.540 total time=  12.2s
    [CV 2/3] END max_depth=None, max_features=1.0, max_samples=10000, min_samples_leaf=9, min_samples_split=8, n_estimators=140;, score=0.745 total time=  12.5s
    [CV 3/3] END max_depth=None, max_features=1.0, max_samples=10000, min_samples_leaf=9, min_samples_split=8, n_estimators=140;, score=0.610 total time=  11.9s
    [CV 1/3] END max_depth=10, max_features=1.0, max_samples=10000, min_samples_leaf=2, min_samples_split=8, n_estimators=10;, score=0.527 total time=   0.9s
    [CV 2/3] END max_depth=10, max_features=1.0, max_samples=10000, min_samples_leaf=2, min_samples_split=8, n_estimators=10;, score=0.708 total time=   0.8s
    [CV 3/3] END max_depth=10, max_features=1.0, max_samples=10000, min_samples_leaf=2, min_samples_split=8, n_estimators=10;, score=0.595 total time=   0.8s
    [CV 1/3] END max_depth=20, max_features=sqrt, max_samples=10000, min_samples_leaf=4, min_samples_split=3, n_estimators=120;, score=0.533 total time=   2.9s
    [CV 2/3] END max_depth=20, max_features=sqrt, max_samples=10000, min_samples_leaf=4, min_samples_split=3, n_estimators=120;, score=0.720 total time=   3.0s
    [CV 3/3] END max_depth=20, max_features=sqrt, max_samples=10000, min_samples_leaf=4, min_samples_split=3, n_estimators=120;, score=0.645 total time=   2.8s
    [CV 1/3] END max_depth=10, max_features=sqrt, max_samples=10000, min_samples_leaf=2, min_samples_split=7, n_estimators=140;, score=0.492 total time=   2.4s
    [CV 2/3] END max_depth=10, max_features=sqrt, max_samples=10000, min_samples_leaf=2, min_samples_split=7, n_estimators=140;, score=0.663 total time=   2.5s
    [CV 3/3] END max_depth=10, max_features=sqrt, max_samples=10000, min_samples_leaf=2, min_samples_split=7, n_estimators=140;, score=0.629 total time=   2.6s
    [CV 1/3] END max_depth=20, max_features=0.5, max_samples=10000, min_samples_leaf=6, min_samples_split=4, n_estimators=120;, score=0.541 total time=   6.4s
    [CV 2/3] END max_depth=20, max_features=0.5, max_samples=10000, min_samples_leaf=6, min_samples_split=4, n_estimators=120;, score=0.750 total time=   6.4s
    [CV 3/3] END max_depth=20, max_features=0.5, max_samples=10000, min_samples_leaf=6, min_samples_split=4, n_estimators=120;, score=0.634 total time=   6.2s
    [CV 1/3] END max_depth=20, max_features=sqrt, max_samples=10000, min_samples_leaf=3, min_samples_split=6, n_estimators=60;, score=0.532 total time=   1.6s
    [CV 2/3] END max_depth=20, max_features=sqrt, max_samples=10000, min_samples_leaf=3, min_samples_split=6, n_estimators=60;, score=0.728 total time=   1.6s
    [CV 3/3] END max_depth=20, max_features=sqrt, max_samples=10000, min_samples_leaf=3, min_samples_split=6, n_estimators=60;, score=0.640 total time=   1.5s
    [CV 1/3] END max_depth=20, max_features=1.0, max_samples=10000, min_samples_leaf=3, min_samples_split=4, n_estimators=160;, score=0.540 total time=  16.6s
    [CV 2/3] END max_depth=20, max_features=1.0, max_samples=10000, min_samples_leaf=3, min_samples_split=4, n_estimators=160;, score=0.759 total time=  16.9s
    [CV 3/3] END max_depth=20, max_features=1.0, max_samples=10000, min_samples_leaf=3, min_samples_split=4, n_estimators=160;, score=0.617 total time=  16.2s
    [CV 1/3] END max_depth=None, max_features=1.0, max_samples=10000, min_samples_leaf=9, min_samples_split=6, n_estimators=40;, score=0.539 total time=   3.5s
    [CV 2/3] END max_depth=None, max_features=1.0, max_samples=10000, min_samples_leaf=9, min_samples_split=6, n_estimators=40;, score=0.741 total time=   3.6s
    [CV 3/3] END max_depth=None, max_features=1.0, max_samples=10000, min_samples_leaf=9, min_samples_split=6, n_estimators=40;, score=0.614 total time=   3.4s
    [CV 1/3] END max_depth=20, max_features=1.0, max_samples=10000, min_samples_leaf=3, min_samples_split=9, n_estimators=10;, score=0.531 total time=   1.1s
    [CV 2/3] END max_depth=20, max_features=1.0, max_samples=10000, min_samples_leaf=3, min_samples_split=9, n_estimators=10;, score=0.748 total time=   1.1s
    [CV 3/3] END max_depth=20, max_features=1.0, max_samples=10000, min_samples_leaf=3, min_samples_split=9, n_estimators=10;, score=0.603 total time=   1.1s
    [CV 1/3] END max_depth=10, max_features=0.5, max_samples=10000, min_samples_leaf=1, min_samples_split=6, n_estimators=150;, score=0.531 total time=   6.3s
    [CV 2/3] END max_depth=10, max_features=0.5, max_samples=10000, min_samples_leaf=1, min_samples_split=6, n_estimators=150;, score=0.712 total time=   6.5s
    [CV 3/3] END max_depth=10, max_features=0.5, max_samples=10000, min_samples_leaf=1, min_samples_split=6, n_estimators=150;, score=0.626 total time=   6.3s
    [CV 1/3] END max_depth=None, max_features=sqrt, max_samples=10000, min_samples_leaf=9, min_samples_split=7, n_estimators=70;, score=0.521 total time=   1.6s
    [CV 2/3] END max_depth=None, max_features=sqrt, max_samples=10000, min_samples_leaf=9, min_samples_split=7, n_estimators=70;, score=0.700 total time=   1.6s
    [CV 3/3] END max_depth=None, max_features=sqrt, max_samples=10000, min_samples_leaf=9, min_samples_split=7, n_estimators=70;, score=0.634 total time=   1.5s
    [CV 1/3] END max_depth=10, max_features=sqrt, max_samples=10000, min_samples_leaf=7, min_samples_split=5, n_estimators=100;, score=0.489 total time=   1.7s
    [CV 2/3] END max_depth=10, max_features=sqrt, max_samples=10000, min_samples_leaf=7, min_samples_split=5, n_estimators=100;, score=0.657 total time=   1.7s
    [CV 3/3] END max_depth=10, max_features=sqrt, max_samples=10000, min_samples_leaf=7, min_samples_split=5, n_estimators=100;, score=0.624 total time=   1.8s
    [CV 1/3] END max_depth=None, max_features=sqrt, max_samples=10000, min_samples_leaf=5, min_samples_split=3, n_estimators=50;, score=0.522 total time=   1.2s
    [CV 2/3] END max_depth=None, max_features=sqrt, max_samples=10000, min_samples_leaf=5, min_samples_split=3, n_estimators=50;, score=0.716 total time=   1.2s
    [CV 3/3] END max_depth=None, max_features=sqrt, max_samples=10000, min_samples_leaf=5, min_samples_split=3, n_estimators=50;, score=0.635 total time=   1.3s
    [CV 1/3] END max_depth=None, max_features=sqrt, max_samples=10000, min_samples_leaf=4, min_samples_split=8, n_estimators=190;, score=0.530 total time=   4.6s
    [CV 2/3] END max_depth=None, max_features=sqrt, max_samples=10000, min_samples_leaf=4, min_samples_split=8, n_estimators=190;, score=0.721 total time=   4.7s
    [CV 3/3] END max_depth=None, max_features=sqrt, max_samples=10000, min_samples_leaf=4, min_samples_split=8, n_estimators=190;, score=0.646 total time=   4.4s
    [CV 1/3] END max_depth=None, max_features=1.0, max_samples=10000, min_samples_leaf=9, min_samples_split=5, n_estimators=120;, score=0.538 total time=  10.4s
    [CV 2/3] END max_depth=None, max_features=1.0, max_samples=10000, min_samples_leaf=9, min_samples_split=5, n_estimators=120;, score=0.745 total time=  10.5s
    [CV 3/3] END max_depth=None, max_features=1.0, max_samples=10000, min_samples_leaf=9, min_samples_split=5, n_estimators=120;, score=0.607 total time=  10.1s
    CPU times: user 5min 12s, sys: 3.63 s, total: 5min 16s
    Wall time: 5min 16s





<style>#sk-container-id-5 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-5 {
  color: var(--sklearn-color-text);
}

#sk-container-id-5 pre {
  padding: 0;
}

#sk-container-id-5 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-5 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-5 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-5 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-5 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-5 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-5 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-5 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-5 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-5 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-5 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-5 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-5 label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: start;
  justify-content: space-between;
  gap: 0.5em;
}

#sk-container-id-5 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-5 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-5 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-5 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-5 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-5 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-5 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-5 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-5 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-5 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-5 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-5 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-5 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-5 div.sk-label label.sk-toggleable__label,
#sk-container-id-5 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-5 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-5 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-5 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-5 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-5 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-5 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-5 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-5 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-5 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-5 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-5 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-5 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-5" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>RandomizedSearchCV(cv=3, estimator=RandomForestRegressor(), n_iter=20,
                   param_distributions={&#x27;max_depth&#x27;: [None, 10, 20],
                                        &#x27;max_features&#x27;: [0.5, 1.0, &#x27;sqrt&#x27;],
                                        &#x27;max_samples&#x27;: [10000],
                                        &#x27;min_samples_leaf&#x27;: array([1, 2, 3, 4, 5, 6, 7, 8, 9]),
                                        &#x27;min_samples_split&#x27;: array([2, 3, 4, 5, 6, 7, 8, 9]),
                                        &#x27;n_estimators&#x27;: array([ 10,  20,  30,  40,  50,  60,  70,  80,  90, 100, 110, 120, 130,
       140, 150, 160, 170, 180, 190])},
                   verbose=3)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-5" type="checkbox" ><label for="sk-estimator-id-5" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>RandomizedSearchCV</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.model_selection.RandomizedSearchCV.html">?<span>Documentation for RandomizedSearchCV</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted"><pre>RandomizedSearchCV(cv=3, estimator=RandomForestRegressor(), n_iter=20,
                   param_distributions={&#x27;max_depth&#x27;: [None, 10, 20],
                                        &#x27;max_features&#x27;: [0.5, 1.0, &#x27;sqrt&#x27;],
                                        &#x27;max_samples&#x27;: [10000],
                                        &#x27;min_samples_leaf&#x27;: array([1, 2, 3, 4, 5, 6, 7, 8, 9]),
                                        &#x27;min_samples_split&#x27;: array([2, 3, 4, 5, 6, 7, 8, 9]),
                                        &#x27;n_estimators&#x27;: array([ 10,  20,  30,  40,  50,  60,  70,  80,  90, 100, 110, 120, 130,
       140, 150, 160, 170, 180, 190])},
                   verbose=3)</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-6" type="checkbox" ><label for="sk-estimator-id-6" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>best_estimator_: RandomForestRegressor</div></div></label><div class="sk-toggleable__content fitted"><pre>RandomForestRegressor(max_depth=20, max_features=0.5, max_samples=10000,
                      min_samples_leaf=2, min_samples_split=5,
                      n_estimators=120)</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-7" type="checkbox" ><label for="sk-estimator-id-7" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>RandomForestRegressor</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.ensemble.RandomForestRegressor.html">?<span>Documentation for RandomForestRegressor</span></a></div></label><div class="sk-toggleable__content fitted"><pre>RandomForestRegressor(max_depth=20, max_features=0.5, max_samples=10000,
                      min_samples_leaf=2, min_samples_split=5,
                      n_estimators=120)</pre></div> </div></div></div></div></div></div></div></div></div>




```python
# Find the best parameters from RandomizedSearchCV
rs_model.best_params_
```




    {'n_estimators': 120,
     'min_samples_split': 5,
     'min_samples_leaf': 2,
     'max_samples': 10000,
     'max_features': 0.5,
     'max_depth': 20}




```python
# Evaluate the RandomizedSearch model
rs_model_scores = show_scores(rs_model)
rs_model_scores
```




    {'Training MAE': 5801.452269779251,
     'Valid MAE': 7234.26542138088,
     'Training RMSLE': 0.26698880657777896,
     'Valid RMSLE': 0.2994038157501437,
     'Training R^2': 0.849611289592257,
     'Valid R^2': 0.8293046484487806}



### 5.3 Training a model with the best hyperparameters

Like all good machine learning cooking shows, I prepared a model earlier. 

I tried 100 different combinations of hyperparameters (setting `n_iter=100` in `RandomizedSearchCV`) and found the best results came from the settings below.

* `n_estimators=90`
* `max_depth=None`
* `min_samples_leaf=1`
* `min_samples_split=5`
* `max_features=0.5`
* `n_jobs=-1`
* `max_samples=None`



```python
%%time

# Create a model with best found hyperparameters 
# Note: There may be better values out there with longer searches but these are 
# the best I found with a ~2 hour search. A good challenge would be to see if you 
# can find better values.
ideal_model = RandomForestRegressor(n_estimators=90,
                                    max_depth=None,
                                    min_samples_leaf=1,
                                    min_samples_split=5,
                                    max_features=0.5,
                                    n_jobs=-1,
                                    max_samples=None)

# Fit a model to the preprocessed data
ideal_model.fit(X=X_train_preprocessed, 
                y=y_train)
```

    CPU times: user 3min 54s, sys: 2.01 s, total: 3min 56s
    Wall time: 33.6 s





<style>#sk-container-id-6 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-6 {
  color: var(--sklearn-color-text);
}

#sk-container-id-6 pre {
  padding: 0;
}

#sk-container-id-6 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-6 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-6 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-6 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-6 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-6 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-6 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-6 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-6 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-6 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-6 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-6 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-6 label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: start;
  justify-content: space-between;
  gap: 0.5em;
}

#sk-container-id-6 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-6 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-6 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-6 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-6 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-6 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-6 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-6 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-6 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-6 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-6 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-6 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-6 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-6 div.sk-label label.sk-toggleable__label,
#sk-container-id-6 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-6 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-6 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-6 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-6 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-6 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-6 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-6 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-6 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-6 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-6 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-6 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-6 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-6" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>RandomForestRegressor(max_features=0.5, min_samples_split=5, n_estimators=90,
                      n_jobs=-1)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-8" type="checkbox" checked><label for="sk-estimator-id-8" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>RandomForestRegressor</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.ensemble.RandomForestRegressor.html">?<span>Documentation for RandomForestRegressor</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted"><pre>RandomForestRegressor(max_features=0.5, min_samples_split=5, n_estimators=90,
                      n_jobs=-1)</pre></div> </div></div></div></div>




```python
%%time

# Evaluate ideal model
ideal_model_scores = show_scores(model=ideal_model)
ideal_model_scores
```

    CPU times: user 21.8 s, sys: 604 ms, total: 22.4 s
    Wall time: 3.69 s





    {'Training MAE': 1949.0999880180598,
     'Valid MAE': 5974.550618889153,
     'Training RMSLE': 0.10176958631302005,
     'Valid RMSLE': 0.24811437650216134,
     'Training R^2': 0.9811477729044881,
     'Valid R^2': 0.8806284725794442}



With these new hyperparameters as well as using all the samples, we can see an improvement to our models performance.

One thing to keep in mind is that a larger model isn't always the best for a given problem even if it performs better.

For example, you may require a model that performs inference (makes predictions) very fast with a slight tradeoff to performance.

One way to a faster model is by altering some of the hyperparameters to create a smaller overall model. 

Particularly by lowering `n_estimators` since each increase in `n_estimators` is basically building another small model.

Let's half our `n_estimators` value and see how it goes.


```python
%%time

# Halve the number of estimators
fast_model = RandomForestRegressor(n_estimators=45,
                                   max_depth=None,
                                   min_samples_leaf=1,
                                   min_samples_split=5,
                                   max_features=0.5,
                                   n_jobs=-1,
                                   max_samples=None)

# Fit the faster model to the data
fast_model.fit(X=X_train_preprocessed, 
               y=y_train)
```

    CPU times: user 1min 56s, sys: 1.03 s, total: 1min 57s
    Wall time: 17.2 s





<style>#sk-container-id-7 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-7 {
  color: var(--sklearn-color-text);
}

#sk-container-id-7 pre {
  padding: 0;
}

#sk-container-id-7 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-7 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-7 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-7 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-7 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-7 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-7 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-7 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-7 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-7 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-7 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-7 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-7 label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: start;
  justify-content: space-between;
  gap: 0.5em;
}

#sk-container-id-7 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-7 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-7 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-7 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-7 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-7 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-7 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-7 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-7 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-7 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-7 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-7 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-7 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-7 div.sk-label label.sk-toggleable__label,
#sk-container-id-7 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-7 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-7 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-7 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-7 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-7 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-7 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-7 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-7 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-7 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-7 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-7 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-7 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-7" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>RandomForestRegressor(max_features=0.5, min_samples_split=5, n_estimators=45,
                      n_jobs=-1)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-9" type="checkbox" checked><label for="sk-estimator-id-9" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>RandomForestRegressor</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.ensemble.RandomForestRegressor.html">?<span>Documentation for RandomForestRegressor</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted"><pre>RandomForestRegressor(max_features=0.5, min_samples_split=5, n_estimators=45,
                      n_jobs=-1)</pre></div> </div></div></div></div>




```python
%%time

# Get results from the fast model
fast_model_scores = show_scores(model=fast_model)
fast_model_scores
```

    CPU times: user 10.3 s, sys: 223 ms, total: 10.5 s
    Wall time: 1.69 s





    {'Training MAE': 1978.447921118064,
     'Valid MAE': 5988.779312990989,
     'Training RMSLE': 0.1032925206449546,
     'Valid RMSLE': 0.24863612313883446,
     'Training R^2': 0.9804543572359619,
     'Valid R^2': 0.880183406149544}



### 5.4 Comparing our model's scores

We've built four models so far with varying amounts of data and hyperparameters.

Let's compile the results into a DataFrame and then make a plot to compare them.


```python
# Add names of models to dictionaries
base_model_scores["model_name"] = "default_model"
rs_model_scores["model_name"] = "random_search_model"
ideal_model_scores["model_name"] = "ideal_model" 
fast_model_scores["model_name"] = "fast_model" 

# Turn all model score dictionaries into a list
all_model_scores = [base_model_scores, 
                    rs_model_scores, 
                    ideal_model_scores,
                    fast_model_scores]

# Create DataFrame and sort model scores by validation RMSLE
model_comparison_df = pd.DataFrame(all_model_scores).sort_values(by="Valid RMSLE", ascending=False)
model_comparison_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Training MAE</th>
      <th>Valid MAE</th>
      <th>Training RMSLE</th>
      <th>Valid RMSLE</th>
      <th>Training R^2</th>
      <th>Valid R^2</th>
      <th>model_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>5801.452270</td>
      <td>7234.265421</td>
      <td>0.266989</td>
      <td>0.299404</td>
      <td>0.849611</td>
      <td>0.829305</td>
      <td>random_search_model</td>
    </tr>
    <tr>
      <th>0</th>
      <td>5586.866609</td>
      <td>7174.828475</td>
      <td>0.259820</td>
      <td>0.296190</td>
      <td>0.859375</td>
      <td>0.830759</td>
      <td>default_model</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1978.447921</td>
      <td>5988.779313</td>
      <td>0.103293</td>
      <td>0.248636</td>
      <td>0.980454</td>
      <td>0.880183</td>
      <td>fast_model</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1949.099988</td>
      <td>5974.550619</td>
      <td>0.101770</td>
      <td>0.248114</td>
      <td>0.981148</td>
      <td>0.880628</td>
      <td>ideal_model</td>
    </tr>
  </tbody>
</table>
</div>



Now we've got our model result data in a DataFrame, let's turn it into a bar plot comparing the validation RMSLE of each model.


```python
# Get mean RSMLE score of all models
mean_rsmle_score = model_comparison_df["Valid RMSLE"].mean()

# Plot validation RMSLE against each other 
plt.figure(figsize=(10, 5))
plt.bar(x=model_comparison_df["model_name"],
        height=model_comparison_df["Valid RMSLE"].values)
plt.xlabel("Model")
plt.ylabel("Validation RMSLE (lower is better)")
plt.xticks(rotation=0, fontsize=10);
plt.axhline(y=mean_rsmle_score, 
            color="red", 
            linestyle="--", 
            label=f"Mean RMSLE: {mean_rsmle_score:.4f}")
plt.legend()
plt.show();
```


    
![png](bluebook-bulldozer-price-regression_files/bluebook-bulldozer-price-regression_170_0.png)
    


By the looks of the plot, our `ideal_model` is indeed the ideal model, slightly edging out `fast_model` in terms of validation RMSLE.

## 6. Saving our best model to file

Since we've confirmed our best model as our `ideal_model` object, we can save it to file so we can load it in later and use it without having to retrain it.

> **Note:** For more on model saving options with Scikit-Learn, see the [documentation on model persistence](https://scikit-learn.org/stable/model_persistence.html).

To save our model we can use the [`joblib.dump`](https://joblib.readthedocs.io/en/stable/generated/joblib.dump.html) method.


```python
import joblib

bulldozer_price_prediction_model_name = "randomforest_regressor_best_RMSLE.pkl"

# Save model to file
joblib.dump(value=ideal_model, 
            filename=bulldozer_price_prediction_model_name)
```




    ['randomforest_regressor_best_RMSLE.pkl']




```python
# Load the best model
best_model = joblib.load(filename=bulldozer_price_prediction_model_name)
best_model
```




<style>#sk-container-id-8 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-8 {
  color: var(--sklearn-color-text);
}

#sk-container-id-8 pre {
  padding: 0;
}

#sk-container-id-8 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-8 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-8 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-8 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-8 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-8 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-8 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-8 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-8 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-8 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-8 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-8 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-8 label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: start;
  justify-content: space-between;
  gap: 0.5em;
}

#sk-container-id-8 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-8 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-8 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-8 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-8 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-8 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-8 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-8 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-8 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-8 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-8 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-8 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-8 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-8 div.sk-label label.sk-toggleable__label,
#sk-container-id-8 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-8 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-8 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-8 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-8 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-8 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-8 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-8 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-8 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-8 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-8 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-8 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-8 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-8" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>RandomForestRegressor(max_features=0.5, min_samples_split=5, n_estimators=90,
                      n_jobs=-1)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-10" type="checkbox" checked><label for="sk-estimator-id-10" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>RandomForestRegressor</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.ensemble.RandomForestRegressor.html">?<span>Documentation for RandomForestRegressor</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted"><pre>RandomForestRegressor(max_features=0.5, min_samples_split=5, n_estimators=90,
                      n_jobs=-1)</pre></div> </div></div></div></div>




```python
# Confirm that the model works
best_model_scores = show_scores(model=best_model)
best_model_scores
```




    {'Training MAE': 1949.099988018059,
     'Valid MAE': 5974.550618889153,
     'Training RMSLE': 0.10176958631302005,
     'Valid RMSLE': 0.24811437650216137,
     'Training R^2': 0.9811477729044881,
     'Valid R^2': 0.8806284725794442}




```python
# Is the loaded model as good as the non-loaded model?
if np.isclose(a=best_model_scores["Valid RMSLE"], 
              b=ideal_model_scores["Valid RMSLE"],
              atol=1e-4): # Make sure values are within 0.0001 of each other
    print(f"[INFO] Model results are close!")
else:
    print(f"[INFO] Model results aren't close, did something go wrong?")
```

    [INFO] Model results are close!


## 7. Making predictions on test data

Now we've got a trained model saved and loaded, it's time to make predictions on the test data.

Our model is trained on data prior to 2011, however, the test data is from May 1 2012 to November 2012.

So what we're doing is trying to use the patterns our model has learned from the training data to predict the sale price of a bulldozer with characteristics it's never seen before but are assumed to be similar to that of those in the training data.

Let's load in the test data from `Test.csv`, we'll make sure to parse the dates of the `saledate` column.


```python
# Load the test data
test_df = pd.read_csv(filepath_or_buffer="bluebook-for-bulldozers/Test.csv",
                      parse_dates=["saledate"],
                      low_memory=False)
test_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SalesID</th>
      <th>MachineID</th>
      <th>ModelID</th>
      <th>datasource</th>
      <th>auctioneerID</th>
      <th>YearMade</th>
      <th>MachineHoursCurrentMeter</th>
      <th>UsageBand</th>
      <th>saledate</th>
      <th>fiModelDesc</th>
      <th>...</th>
      <th>Undercarriage_Pad_Width</th>
      <th>Stick_Length</th>
      <th>Thumb</th>
      <th>Pattern_Changer</th>
      <th>Grouser_Type</th>
      <th>Backhoe_Mounting</th>
      <th>Blade_Type</th>
      <th>Travel_Controls</th>
      <th>Differential_Type</th>
      <th>Steering_Controls</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1227829</td>
      <td>1006309</td>
      <td>3168</td>
      <td>121</td>
      <td>3</td>
      <td>1999</td>
      <td>3688.0</td>
      <td>Low</td>
      <td>2012-05-03</td>
      <td>580G</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1227844</td>
      <td>1022817</td>
      <td>7271</td>
      <td>121</td>
      <td>3</td>
      <td>1000</td>
      <td>28555.0</td>
      <td>High</td>
      <td>2012-05-10</td>
      <td>936</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Standard</td>
      <td>Conventional</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1227847</td>
      <td>1031560</td>
      <td>22805</td>
      <td>121</td>
      <td>3</td>
      <td>2004</td>
      <td>6038.0</td>
      <td>Medium</td>
      <td>2012-05-10</td>
      <td>EC210BLC</td>
      <td>...</td>
      <td>None or Unspecified</td>
      <td>9' 6"</td>
      <td>Manual</td>
      <td>None or Unspecified</td>
      <td>Double</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1227848</td>
      <td>56204</td>
      <td>1269</td>
      <td>121</td>
      <td>3</td>
      <td>2006</td>
      <td>8940.0</td>
      <td>High</td>
      <td>2012-05-10</td>
      <td>330CL</td>
      <td>...</td>
      <td>None or Unspecified</td>
      <td>None or Unspecified</td>
      <td>Manual</td>
      <td>Yes</td>
      <td>Triple</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1227863</td>
      <td>1053887</td>
      <td>22312</td>
      <td>121</td>
      <td>3</td>
      <td>2005</td>
      <td>2286.0</td>
      <td>Low</td>
      <td>2012-05-10</td>
      <td>650K</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>None or Unspecified</td>
      <td>PAT</td>
      <td>None or Unspecified</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 52 columns</p>
</div>



You might notice that the `test_df` is missing the `SalePrice` column.

That's because that's the variable we're trying to predict based on all of the other variables.

We can make predictions with our `best_model` using the [`predict` method](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor.predict).

### 7.1 Preprocessing the test data (to be in the same format as the training data)

Our model has been trained on data preprocessed in a certain way. 

This means in order to make predictions on the test data, we need to take the same steps we used to preprocess the training data to preprocess the test data.

Remember, whatever you do to preprocess the training data, you have to do to the test data.

Let's recreate the steps we used for preprocessing the training data except this time we'll do it on the test data.  

First, we'll add the extra date features to breakdown the `saledate` column.


```python
# Make a function to add date columns
def add_datetime_features_to_df(df, date_column="saledate"):
    # Add datetime parameters for saledate
    df["saleYear"] = df[date_column].dt.year
    df["saleMonth"] = df[date_column].dt.month
    df["saleDay"] = df[date_column].dt.day
    df["saleDayofweek"] = df[date_column].dt.dayofweek
    df["saleDayofyear"] = df[date_column].dt.dayofyear

    # Drop original saledate column
    df.drop("saledate", axis=1, inplace=True)

    return df

# Preprocess test_df to have same columns as train_df (add the datetime features)
test_df = add_datetime_features_to_df(df=test_df)
test_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SalesID</th>
      <th>MachineID</th>
      <th>ModelID</th>
      <th>datasource</th>
      <th>auctioneerID</th>
      <th>YearMade</th>
      <th>MachineHoursCurrentMeter</th>
      <th>UsageBand</th>
      <th>fiModelDesc</th>
      <th>fiBaseModel</th>
      <th>...</th>
      <th>Backhoe_Mounting</th>
      <th>Blade_Type</th>
      <th>Travel_Controls</th>
      <th>Differential_Type</th>
      <th>Steering_Controls</th>
      <th>saleYear</th>
      <th>saleMonth</th>
      <th>saleDay</th>
      <th>saleDayofweek</th>
      <th>saleDayofyear</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1227829</td>
      <td>1006309</td>
      <td>3168</td>
      <td>121</td>
      <td>3</td>
      <td>1999</td>
      <td>3688.0</td>
      <td>Low</td>
      <td>580G</td>
      <td>580</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2012</td>
      <td>5</td>
      <td>3</td>
      <td>3</td>
      <td>124</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1227844</td>
      <td>1022817</td>
      <td>7271</td>
      <td>121</td>
      <td>3</td>
      <td>1000</td>
      <td>28555.0</td>
      <td>High</td>
      <td>936</td>
      <td>936</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Standard</td>
      <td>Conventional</td>
      <td>2012</td>
      <td>5</td>
      <td>10</td>
      <td>3</td>
      <td>131</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1227847</td>
      <td>1031560</td>
      <td>22805</td>
      <td>121</td>
      <td>3</td>
      <td>2004</td>
      <td>6038.0</td>
      <td>Medium</td>
      <td>EC210BLC</td>
      <td>EC210</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2012</td>
      <td>5</td>
      <td>10</td>
      <td>3</td>
      <td>131</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1227848</td>
      <td>56204</td>
      <td>1269</td>
      <td>121</td>
      <td>3</td>
      <td>2006</td>
      <td>8940.0</td>
      <td>High</td>
      <td>330CL</td>
      <td>330</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2012</td>
      <td>5</td>
      <td>10</td>
      <td>3</td>
      <td>131</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1227863</td>
      <td>1053887</td>
      <td>22312</td>
      <td>121</td>
      <td>3</td>
      <td>2005</td>
      <td>2286.0</td>
      <td>Low</td>
      <td>650K</td>
      <td>650</td>
      <td>...</td>
      <td>None or Unspecified</td>
      <td>PAT</td>
      <td>None or Unspecified</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2012</td>
      <td>5</td>
      <td>10</td>
      <td>3</td>
      <td>131</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 56 columns</p>
</div>




```python
#import pandas as pd
import numpy as np

# Make a copy to avoid modifying the original DataFrame
test_df_preprocessed = test_df.copy()

# --- Impute Numerical Features (using the fitted imputer) ---
test_df_preprocessed[numerical_features] = numerical_imputer.transform(test_df_preprocessed[numerical_features])

# --- Impute Categorical Features (using the fitted imputer) ---
test_df_preprocessed[categorical_features] = categorical_imputer.transform(test_df_preprocessed[categorical_features].astype(str))

# --- Ordinal Encode Categorical Features (using the fitted encoder) ---
test_df_preprocessed[categorical_features] = ordinal_encoder.transform(test_df_preprocessed[categorical_features].astype(str))

# --- Final Handling of Remaining NaN Values (consistent with train/valid) ---
test_df_preprocessed.fillna(-1, inplace=True)

test_df_preprocessed.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 12457 entries, 0 to 12456
    Data columns (total 56 columns):
     #   Column                    Non-Null Count  Dtype  
    ---  ------                    --------------  -----  
     0   SalesID                   12457 non-null  float64
     1   MachineID                 12457 non-null  float64
     2   ModelID                   12457 non-null  float64
     3   datasource                12457 non-null  float64
     4   auctioneerID              12457 non-null  float64
     5   YearMade                  12457 non-null  float64
     6   MachineHoursCurrentMeter  12457 non-null  float64
     7   UsageBand                 12457 non-null  float64
     8   fiModelDesc               12457 non-null  float64
     9   fiBaseModel               12457 non-null  float64
     10  fiSecondaryDesc           12457 non-null  float64
     11  fiModelSeries             12457 non-null  float64
     12  fiModelDescriptor         12457 non-null  float64
     13  ProductSize               12457 non-null  float64
     14  fiProductClassDesc        12457 non-null  float64
     15  state                     12457 non-null  float64
     16  ProductGroup              12457 non-null  float64
     17  ProductGroupDesc          12457 non-null  float64
     18  Drive_System              12457 non-null  float64
     19  Enclosure                 12457 non-null  float64
     20  Forks                     12457 non-null  float64
     21  Pad_Type                  12457 non-null  float64
     22  Ride_Control              12457 non-null  float64
     23  Stick                     12457 non-null  float64
     24  Transmission              12457 non-null  float64
     25  Turbocharged              12457 non-null  float64
     26  Blade_Extension           12457 non-null  float64
     27  Blade_Width               12457 non-null  float64
     28  Enclosure_Type            12457 non-null  float64
     29  Engine_Horsepower         12457 non-null  float64
     30  Hydraulics                12457 non-null  float64
     31  Pushblock                 12457 non-null  float64
     32  Ripper                    12457 non-null  float64
     33  Scarifier                 12457 non-null  float64
     34  Tip_Control               12457 non-null  float64
     35  Tire_Size                 12457 non-null  float64
     36  Coupler                   12457 non-null  float64
     37  Coupler_System            12457 non-null  float64
     38  Grouser_Tracks            12457 non-null  float64
     39  Hydraulics_Flow           12457 non-null  float64
     40  Track_Type                12457 non-null  float64
     41  Undercarriage_Pad_Width   12457 non-null  float64
     42  Stick_Length              12457 non-null  float64
     43  Thumb                     12457 non-null  float64
     44  Pattern_Changer           12457 non-null  float64
     45  Grouser_Type              12457 non-null  float64
     46  Backhoe_Mounting          12457 non-null  float64
     47  Blade_Type                12457 non-null  float64
     48  Travel_Controls           12457 non-null  float64
     49  Differential_Type         12457 non-null  float64
     50  Steering_Controls         12457 non-null  float64
     51  saleYear                  12457 non-null  float64
     52  saleMonth                 12457 non-null  float64
     53  saleDay                   12457 non-null  float64
     54  saleDayofweek             12457 non-null  float64
     55  saleDayofyear             12457 non-null  float64
    dtypes: float64(56)
    memory usage: 5.3 MB



```python
# Make predictions on the preprocessed test data
test_preds = best_model.predict(test_df_preprocessed)
```


```python
# Check the first 10 test predictions
test_preds[:10]
```




    array([14891.66606542, 34336.35361552, 51545.46957672, 94009.45839654,
           26788.22089947, 29943.14814815, 26507.42724868, 19280.01763668,
           18642.75573192, 31095.65736732])




```python
# Check number of test predictions
test_preds.shape, test_df.shape
```




    ((12457,), (12457, 56))



Perfect, looks like theres one prediction per sample in the test DataFrame.

Now how would we submit our predictions to Kaggle?

Well, when looking at the [Kaggle submission requirements](https://www.kaggle.com/c/bluebook-for-bulldozers/overview/evaluation), we see that if we wanted to make a submission, the data is required to be in a certain format. 

Namely, a DataFrame containing the `SalesID` and the predicted `SalePrice` of the bulldozer.

Let's make it.


```python
pred_df = pd.DataFrame()
pred_df["SalesID"] = test_df["SalesID"]
pred_df["SalePrice"] = test_preds
pred_df.sample(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SalesID</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>544</th>
      <td>1229804</td>
      <td>18115.155022</td>
    </tr>
    <tr>
      <th>2073</th>
      <td>4640558</td>
      <td>43962.493001</td>
    </tr>
    <tr>
      <th>3736</th>
      <td>6267081</td>
      <td>23509.841270</td>
    </tr>
    <tr>
      <th>12214</th>
      <td>6638039</td>
      <td>25621.951659</td>
    </tr>
    <tr>
      <th>6068</th>
      <td>6297917</td>
      <td>68167.978996</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Export test dataset predictions to CSV
pred_df.to_csv("predictions.csv",
               index=False)
```

## 8. Making a prediction on a custom sample

We've made predictions on the test dataset which contains sale data from May to November 2012.

But how does our model go on a more recent bulldozer sale?

If we were to find an advertisement on a bulldozer sale, could we use our model on the information in the advertisement to predict the sale price?

In other words, how could we use our model on a single custom sample?

It's one thing to predict on data that has already been formatted but it's another thing to be able to predict a on a completely new and unseen sample.

> **Note:** For predicting on a custom sample, the same rules apply as making predictions on the test dataset. The data you make predictions on should be in the same format that your model was trained on. For example, it should have all the same features and the numerical encodings should be in the same ballpark (e.g. preprocessed by the `ordinal_encoder` we fit to the training set). It's likely that samples you collect from the wild may not be as well formatted as samples in a pre-existing dataset. So it's the job of the machine learning engineer to be able to format/preprocess new samples in the same way a model was trained on.

If we're going to make a prediction on a custom sample, it'll need to be in the same format as our other datasets.

So let's remind ourselves of the columns/features in our test dataset.


```python
#Get example from test_df
test_df_preprocessed_sample = test_df_preprocessed.sample(n=1, random_state=42)

# Turn back into original format
test_df_unpreprocessed_sample = test_df_preprocessed_sample.copy() 
test_df_unpreprocessed_sample[categorical_features] = ordinal_encoder.inverse_transform(test_df_unpreprocessed_sample[categorical_features])
test_df_unpreprocessed_sample.to_dict(orient="records")
```




    [{'SalesID': 1229148.0,
      'MachineID': 1042578.0,
      'ModelID': 9579.0,
      'datasource': 121.0,
      'auctioneerID': 3.0,
      'YearMade': 2004.0,
      'MachineHoursCurrentMeter': 3290.0,
      'UsageBand': 'Medium',
      'fiModelDesc': 'S250',
      'fiBaseModel': 'S250',
      'fiSecondaryDesc': 'nan',
      'fiModelSeries': 'nan',
      'fiModelDescriptor': 'nan',
      'ProductSize': 'nan',
      'fiProductClassDesc': 'Skid Steer Loader - 2201.0 to 2701.0 Lb Operating Capacity',
      'state': 'Missouri',
      'ProductGroup': 'SSL',
      'ProductGroupDesc': 'Skid Steer Loaders',
      'Drive_System': 'nan',
      'Enclosure': 'EROPS',
      'Forks': 'None or Unspecified',
      'Pad_Type': 'nan',
      'Ride_Control': 'nan',
      'Stick': 'nan',
      'Transmission': 'nan',
      'Turbocharged': 'nan',
      'Blade_Extension': 'nan',
      'Blade_Width': 'nan',
      'Enclosure_Type': 'nan',
      'Engine_Horsepower': 'nan',
      'Hydraulics': 'Auxiliary',
      'Pushblock': 'nan',
      'Ripper': 'nan',
      'Scarifier': 'nan',
      'Tip_Control': 'nan',
      'Tire_Size': 'nan',
      'Coupler': 'Hydraulic',
      'Coupler_System': 'Yes',
      'Grouser_Tracks': 'None or Unspecified',
      'Hydraulics_Flow': 'Standard',
      'Track_Type': 'nan',
      'Undercarriage_Pad_Width': 'nan',
      'Stick_Length': 'nan',
      'Thumb': 'nan',
      'Pattern_Changer': 'nan',
      'Grouser_Type': 'nan',
      'Backhoe_Mounting': 'nan',
      'Blade_Type': 'nan',
      'Travel_Controls': 'nan',
      'Differential_Type': 'nan',
      'Steering_Controls': 'nan',
      'saleYear': 2012.0,
      'saleMonth': 6.0,
      'saleDay': 15.0,
      'saleDayofweek': 4.0,
      'saleDayofyear': 167.0}]




```python
# Make a prediction on the preprocessed test sample
best_model.predict(test_df_preprocessed_sample)
```




    array([13757.83068783])



## 9. Finding the most important predictive features

Since we've built a model which is able to make predictions, the people you share these predictions with (or yourself) might be curious of what parts of the data led to these predictions.

This is where **feature importance** comes in. 

Feature importance seeks to figure out which different attributes of the data were most important when it comes to predicting the **target variable**.

In our case, after our model learned the patterns in the data, which bulldozer sale attributes were most important for predicting its overall sale price?

We can do this for our `sklearn.ensemble.RandomForestRegressor` instance using the [`feature_importances_`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor.feature_importances_) attribute.

Let's check it out.


```python
# Find feature importance of our best model
best_model_feature_importances = best_model.feature_importances_
best_model_feature_importances
```




    array([3.84521802e-02, 2.60824534e-02, 5.15230807e-02, 2.22928710e-03,
           4.69470888e-03, 1.96481846e-01, 4.03566006e-03, 1.48884091e-03,
           4.82557272e-02, 4.93350014e-02, 3.80139343e-02, 4.49544097e-03,
           2.73017711e-02, 1.56907709e-01, 4.66736199e-02, 8.42527436e-03,
           7.84592589e-03, 4.90822629e-03, 1.32703219e-03, 7.19164238e-02,
           9.83256931e-04, 9.87419448e-04, 2.46129689e-03, 1.72543794e-04,
           1.12268507e-03, 1.46059514e-04, 1.08961338e-03, 2.47347486e-03,
           9.42805969e-03, 4.28369087e-03, 3.53963962e-03, 1.32700382e-03,
           4.43644797e-03, 1.60907840e-03, 4.26366838e-03, 9.43605987e-03,
           1.77413716e-03, 2.53737469e-02, 7.10158788e-03, 7.58701406e-03,
           2.81300467e-03, 1.38406719e-03, 1.70449248e-03, 9.40687172e-04,
           6.45897540e-04, 7.35244431e-04, 7.91057345e-04, 2.16770291e-03,
           3.29032069e-03, 3.16664008e-04, 4.88538854e-04, 7.32187515e-02,
           5.50451217e-03, 8.64781678e-03, 4.48849620e-03, 1.28721184e-02])




```python
print(f"[INFO] Number of feature importance values: {best_model_feature_importances.shape[0]}") 
print(f"[INFO] Number of features in training dataset: {X_train_preprocessed.shape[1]}")
```

    [INFO] Number of feature importance values: 56
    [INFO] Number of features in training dataset: 56



```python
# Create feature importance DataFrame
column_names = test_df.columns
feature_importance_df = pd.DataFrame({"feature_names": column_names,
                                      "feature_importance": best_model_feature_importances}).sort_values(by="feature_importance",
                                                                                                         ascending=False)
feature_importance_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature_names</th>
      <th>feature_importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>YearMade</td>
      <td>0.196482</td>
    </tr>
    <tr>
      <th>13</th>
      <td>ProductSize</td>
      <td>0.156908</td>
    </tr>
    <tr>
      <th>51</th>
      <td>saleYear</td>
      <td>0.073219</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Enclosure</td>
      <td>0.071916</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ModelID</td>
      <td>0.051523</td>
    </tr>
  </tbody>
</table>
</div>



`YearMade` may be contributing the most value in the model's eyes.

How about we turn our DataFrame into a plot to compare values?


```python
# Plot the top feature importance values
top_n = 20
plt.figure(figsize=(10, 5))
plt.barh(y=feature_importance_df["feature_names"][:top_n], # Plot the top_n feature importance values
         width=feature_importance_df["feature_importance"][:top_n])
plt.title(f"Top {top_n} Feature Importance Values for Best RandomForestRegressor Model")
plt.xlabel("Feature importance value")
plt.ylabel("Feature name")
plt.gca().invert_yaxis()
plt.show();
```


    
![png](bluebook-bulldozer-price-regression_files/bluebook-bulldozer-price-regression_197_0.png)
    


Ok, looks like the top 4 features contributing to our model's predictions are `YearMade`, `ProductSize`, `Enclosure` and `saleYear`.

Referring to the original [data dictionary](https://docs.google.com/spreadsheets/d/18ly-bLR8sbDJLITkWG7ozKm8l3RyieQ2Fpgix-beSYI/edit?usp=sharing), do these values make sense to be contributing the most to the model?

* `YearMade` - Year of manufacture of the machine.
* `ProductSize` - Size of the bulldozer.
* `Enclosure` - Type of bulldozer enclosure (e.g. OROPS = Open Rollover Protective Structures, EROPS = Enclosed Rollover Protective Structures).
* `saleYear` - The year the bulldozer was sold (this is one of our engineered features from `saledate`).


And it also makes sense that `ProductSize` be an important feature when deciding on the price of a bulldozer.

Let's check out the unique values for `ProductSize` and `Enclosure`.



```python
print(f"[INFO] Unique ProductSize values: {train_df['ProductSize'].unique()}")
print(f"[INFO] Unique Enclosure values: {train_df['Enclosure'].unique()}")
```

    [INFO] Unique ProductSize values: ['Medium' nan 'Compact' 'Small' 'Large' 'Large / Medium' 'Mini']
    [INFO] Unique Enclosure values: ['OROPS' 'EROPS' 'EROPS w AC' nan 'EROPS AC' 'NO ROPS'
     'None or Unspecified']


My guess is that a bulldozer with a `ProductSize` of `'Mini'` would sell for less than a bulldozer with a size of `'Large'`.

We could investigate this further in an extension to model driven data exploratory analysis or we could take this information to a colleague or client to discuss further.

Either way, we've now got a machine learning model capable of predicting the sale price of bulldozers given their features/attributes!




```python

```
