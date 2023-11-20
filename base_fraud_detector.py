import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


class BaseFraudDetector:

    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
        self.clf = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def eda(self):
        # Displaying distribution of the target variable
        sns.countplot(x='isFraud', data=self.df)
        plt.title('Distribution of Fraudulent Transactions')
        plt.show()

        # Transaction types distribution
        sns.countplot(x='type', data=self.df)
        plt.title('Transaction Types Distribution')
        plt.show()

        # Distribution of transaction amounts
        sns.distplot(self.df['amount'], bins=50)
        plt.title('Transaction Amount Distribution')
        plt.xlabel('Transaction Amount')
        plt.ylabel('Frequency')
        plt.show()

        # Distribution of transaction amounts for fraudulent transactions
        sns.distplot(self.df[self.df['isFraud'] == 1]['amount'], bins=50, color='red')
        plt.title('Fraudulent Transaction Amount Distribution')
        plt.xlabel('Transaction Amount (Fraudulent)')
        plt.ylabel('Frequency')
        plt.show()

        # Box plot for transaction amount by transaction type
        sns.boxplot(x='type', y='amount', data=self.df)
        plt.title('Transaction Amount by Transaction Type')
        plt.show()

        # Correlation heatmap
        correlation_matrix = self.df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Heatmap')
        plt.show()


    def preprocess_data(self, method="IQR"):
    # Handle missing values
      self.df.fillna(0, inplace=True)

    # Feature engineering: Dropping unnecessary columns
      self.df.drop(columns=['nameOrig','oldbalanceOrg','newbalanceOrig','nameDest', 'oldbalanceDest', 'newbalanceDest','isFlaggedFraud'], inplace=True)

    # Encode categorical data
      le = LabelEncoder()
      self.df['type'] = le.fit_transform(self.df['type'])

    # Feature Scaling
      scaler = StandardScaler()
      columns_to_scale = ['step', 'amount']
      self.df[columns_to_scale] = scaler.fit_transform(self.df[columns_to_scale])


    # Splitting data
      X = self.df.drop(columns=['isFraud'])
      y = self.df['isFraud']
      self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
