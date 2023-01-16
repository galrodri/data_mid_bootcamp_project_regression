# Case Study: Regression - Real State

## Scenario

You are working as an analyst for a real estate company. Your company wants to build a machine learning model to predict the selling prices of houses based on a variety of features on which the value of the house is evaluated.

## Objective

Your job is to build a model that will predict the price of a house based on features provided in the dataset. Senior management also wants to explore the characteristics of the houses using some business intelligence tools. One of those parameters includes understanding which factors are responsible for higher property value - $650K and above.

## Approach

In order to get a first impression of the dataset I worked on the SQL queries with MySQLWorkbench. This program allows to see the structure and the relation between the different columns making some calculations. Secondly, I imported the original dataset in Tableau to get a more visual understanding of the data. Tableau allowed me to choose the more understandable charts to get an idea of which paramateres or variables affect the price of the houses at most. After these two steps, I uploaded the dataset in Jupyter Notebook to prepare the data with Python with the intention of creating a model based on this sample able to predict the price of the houses in the future. As said, this prediction model is based on this dataset, which is only a sample of the whole population of data about sold houses.

## The data

The dataset provided for this project consists of information on 22,000 properties. The dataset contains a sample of historic data of houses sold between May 2014 to May 2015.

The features to be found are the following:
- id: Unique identification number for the property
- date: The date the house was sold
- bedrooms
- bathrooms
- sqft_living
- sqft_lot
- floors
- waterfront: The house which has a view to a waterfront
- view
- condition: How good the condition is (Overall). 1 indicates worn-out property and 5 excellent.
- grade: Overall grade given to the housing unit, based on the King County grading system. 1 poor, 13 excellent.
- sqft_above: Square footage of house apart from the basement
- sqft_basement
- yr_built
- yr_renovated
- zipcode
- lat
- long
- sqft_living15: Living room area in 2015(implies-- some renovations) This might or might not have affected the lotSize area.
- sqft_lot15: Lot size area in 2015(implies-- some renovations)
- price

## The process

### Data cleaning
After checking the data looking for duplicates in the id column, null values and the unique values, I made a list of the aspects to be cleaned in the dataset:\

##### FORMAT

All columns were written uniformly, so there is no need to change the names.

There were some duplicates for the column id. Therefore it should be renamed as house_id and new column should be created in order to be able to clearly find the transactions. This new column should be named as trans_id.

##### SUPERFLUOUS COLUMNS

I deleted the column date, because we already knew that the houses were sold between May 2014 and May 2015.

Furthermore the features sqft_living and sqft_lot were deleted as well, since I kept the newest values located in sqft_living15 and sqft_lot15.

Lat and long features won't be needed at this stage, because I could group the houses using the zipcode.

##### DATA TYPES

Since some columns did have a lot of 0 values, indicating that the feature didn't exist for those rows and in order to make the data easier to read and to compute for the Machine Learning Model, I used a binary data type for the following columns: waterfront, sqft_basement, yr_renovated and view. That means that the Machine Learning Model can read if the house present those characteristics or not.

Although all the values were numbers, some of them were acting as categories. Therefore the data type were changed also for them.

### Data wrangling

In this section I tried to make it easier for the computer to read the data. For this purpose, some of the values in the columns were simplified using a binary system.

Since there were values that were out of the average range of the rest of the values in some columns (outliers), I tried the models leaving those outliers intact and then removing them to see how the models would improve their performance.

For the prediction model of the housing price the following parameters or independent variables were chosen:
- grade
- view
- number of bedrooms
- waterfront
- floors
- zipcode

The other features were not correlated to the price or they were already correlated with one of the above mentioned parameters, so that they are included indirectly in the model.

### ML Models

For this project two different regression models were used: the linear regression model and the support vector regression model. Whereas the support regression model did not provide with satisfactory results, with the linear regression model an accuracy of almost 60% was reached, using the selected parameters and removing the outliers.

About the question which factors are responsible for prices of $650K and above, these are the same as for the dataframe containing all the houses of the sample. Therefore the machine learning models used above should be adequate for the more expensive houses as well.
