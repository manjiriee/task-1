import pandas as pd
import matplotlib.pyplot as plt

# Read the dataset
df = pd.read_csv('unemployment in India.csv')

# Clean column names
df.columns = df.columns.str.strip()

# Convert 'Date' column to datetime (day-month-year)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

# Extract Year and Month for analysis
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

# Show columns and first few rows
print("Columns in dataset:", df.columns)
print(df.head())

# Plot Estimated Unemployment Rate over time
plt.figure(figsize=(12,6))
plt.plot(df['Date'], df['Estimated Unemployment Rate (%)'], marker='o')
plt.title('Estimated Unemployment Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

