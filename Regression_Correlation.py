import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os

# Define the output directory
output_dir = r'F:\CSULA\CIS 5200\Project\Output'
os.makedirs(output_dir, exist_ok=True)

# Load the datasets
laporp47_data = pd.read_csv(r'F:\CSULA\CIS 5200\Project\laporp47.csv')
anova_data = pd.read_csv(r'F:\CSULA\CIS 5200\Project\anova.csv')

# Function to calculate the crime cases per capita (per 100,000 population)
def calculate_crime_cases_per_capita(data):
    # Filtering Los Angeles data
    la_data = data[data['City'] == 'Los Angeles']
    
    # Converting Population to numeric, filling missing values
    la_data['Population'] = pd.to_numeric(la_data['Population'], errors='coerce').fillna(method='ffill')
    
    # Aggregating total crime cases and average population by year
    annual_data = la_data.groupby('c_year').agg(
        Total_Crime_Cases=('cnt', 'sum'),
        Avg_Population=('Population', 'mean')
    ).reset_index()
    
    # Calculating crime cases per 100,000 population
    annual_data['Crime_Cases_Per_Capita'] = (annual_data['Total_Crime_Cases'] / annual_data['Avg_Population']) * 100000
    
    return annual_data[['c_year', 'Crime_Cases_Per_Capita']]

# Calculate the crime cases per capita for the ANOVA dataset
crime_cases_per_capita_data = calculate_crime_cases_per_capita(anova_data)

# Filtering Los Angeles data for Proposition 47
laporp47_la = laporp47_data[laporp47_data['City'] == 'Los Angeles']
laporp47_annual = laporp47_la.groupby('c_year')['cnt'].count().reset_index()
laporp47_annual.columns = ['Year', 'Number_of_Cases']

# Merging the crime cases per capita data with Proposition 47 data on the year
merged_data = pd.merge(laporp47_annual, crime_cases_per_capita_data, left_on='Year', right_on='c_year').drop(columns=['c_year'])

# Save merged data to a CSV file
merged_data.to_csv(os.path.join(output_dir, 'merged_data.csv'), index=False)

# Performing regression analysis with crime cases per capita
X = sm.add_constant(merged_data['Number_of_Cases'])  # Independent variable
y = merged_data['Crime_Cases_Per_Capita']  # Dependent variable
model = sm.OLS(y, X).fit()  # Fitting the regression model
regression_summary = model.summary()

# Save regression summary to a text file
with open(os.path.join(output_dir, 'regression_summary.txt'), 'w') as f:
    f.write(regression_summary.as_text())

# Extracting p-value and coefficient for hypothesis testing
p_value = model.pvalues['Number_of_Cases']
coefficient = model.params['Number_of_Cases']

# Hypothesis Test Interpretation
if p_value < 0.05:
    hypothesis_result = (
        "Reject the null hypothesis. "
        "There is a statistically significant relationship between Proposition 47 cases and the crime rate, "
        "suggesting that Proposition 47 affects the crime rate."
    )
else:
    hypothesis_result = (
        "Fail to reject the null hypothesis. "
        "There is not enough evidence to suggest that Proposition 47 significantly affects the crime rate. "
        "There is not correlation (relationship) between Proposition 47 crime cases and overall crime per capita."
    )

# Plotting the regression results
plt.figure(figsize=(14, 7))
plt.scatter(merged_data['Number_of_Cases'], merged_data['Crime_Cases_Per_Capita'], color='blue', label='Data Points')
plt.plot(merged_data['Number_of_Cases'], model.predict(X), color='red', label='Regression Line')
plt.title('Regression Analysis: Crime Cases Per 100,000 Population vs. Number of Proposition 47 Cases (Los Angeles)')
plt.xlabel('Number of Proposition 47 Cases')
plt.ylabel('Crime Cases Per 100,000 Population')
plt.legend()
plt.grid()
plt.savefig(os.path.join(output_dir, 'regression_analysis.png'))  # Save plot
plt.show()

# Plotting the scatter plot for Proposition 47 cases and crime cases per capita
plt.figure(figsize=(14, 7))
plt.scatter(merged_data['Number_of_Cases'], merged_data['Crime_Cases_Per_Capita'], color='purple', label='Proposition 47 vs. Crime Cases Per Capita')
plt.title('Scatter Plot: Proposition 47 Cases vs. Crime Cases Per 100,000 Population')
plt.xlabel('Number of Proposition 47 Cases')
plt.ylabel('Crime Cases Per 100,000 Population')
plt.legend()
plt.grid()
plt.savefig(os.path.join(output_dir, 'scatter_plot.png'))  # Save plot
plt.show()

# Plotting the correlation line chart
plt.figure(figsize=(14, 7))
plt.plot(merged_data['Year'], merged_data['Number_of_Cases'], marker='o', label='Proposition 47 Cases', color='b')
plt.plot(merged_data['Year'], merged_data['Crime_Cases_Per_Capita'], marker='x', label='Crime Cases Per 100,000 Population', color='r')
plt.title('Correlation Line Chart: Proposition 47 Cases vs. Crime Cases Per 100,000 Population in Los Angeles')
plt.xlabel('Year')
plt.ylabel('Number of Cases / Crime Cases Per 100,000 Population')
plt.legend()
plt.grid()
plt.savefig(os.path.join(output_dir, 'correlation_line_chart.png'))  # Save plot
plt.show()

# Output the regression summary, hypothesis test result, coefficient, and p-value
print(regression_summary)
print(f"\nCoefficient for Number of Cases: {coefficient:.4f}")
print(f"P-value for Number of Cases: {p_value:.4f}")
print("\nHypothesis Test Result:")
print(hypothesis_result)

# Save the hypothesis test result to a text file
with open(os.path.join(output_dir, 'hypothesis_test_result.txt'), 'w') as f:
    f.write(f"Coefficient for Number of Cases: {coefficient:.4f}\n")
    f.write(f"P-value for Number of Cases: {p_value:.4f}\n")
    f.write("\nHypothesis Test Result:\n")
    f.write(hypothesis_result)
