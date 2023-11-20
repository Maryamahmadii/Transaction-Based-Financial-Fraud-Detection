from base_fraud_detector.py import *
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc


class BaseFraudDetector:

    def __init__(self, file_path, sample_size=None, random_state=42):
        self.df = pd.read_csv(file_path)

        # If sample_size is provided, perform stratified sampling
        if sample_size is not None:
            # Stratify based on 'isFraud' to keep the class ratio
            _, sample_df = train_test_split(
                self.df,
                stratify=self.df['isFraud'],
                test_size=sample_size,
                random_state=random_state
            )
            self.df = sample_df

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

    def preprocess_data(self, encoding_type,  method="None"):
    # Handle missing values
      self.df.fillna(0, inplace=True)

    # Feature engineering: Dropping unnecessary columns
      self.df.drop(columns=['nameOrig','oldbalanceOrg','newbalanceOrig','nameDest', 'oldbalanceDest', 'newbalanceDest','isFlaggedFraud'], inplace=True)

    # Encode categorical data
      if encoding_type == "onehot":
            # One-Hot Encoding
            self.df = pd.get_dummies(self.df, columns=['type'])
      elif encoding_type == "label":
            # Label Encoding
            le = LabelEncoder()
            self.df['type'] = le.fit_transform(self.df['type'])
      else:
            raise ValueError("Invalid encoding type. Choose 'onehot' or 'label'.")

        # Feature Scaling
      scaler = StandardScaler()
        # Ensure to update the columns to scale according to the encoding used
      columns_to_scale = ['step', 'amount']
      if encoding_type == "onehot":
        columns_to_scale += [col for col in self.df.columns if col.startswith('type_')]
      self.df[columns_to_scale] = scaler.fit_transform(self.df[columns_to_scale])

    # Handle outliers
    # Extracting the target variable for outlier detection
      X_outliers = self.df.drop(columns=['isFraud'])
      y_outliers = self.df['isFraud']

      if method == "IsolationForest":
        iso = IsolationForest(contamination=0.05)
        outliers = iso.fit_predict(self.df)
        self.df = self.df[outliers == 1]

      elif method == "OneClassSVM":
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(self.df)
        osvm = OneClassSVM(nu=0.05)
        outliers = osvm.fit_predict(data_scaled)
        self.df = self.df[outliers == 1]

      elif method == "LOF":
        lof = LocalOutlierFactor()
        outliers = lof.fit_predict(self.df)
        self.df = self.df[outliers == 1]

      elif method == "DBSCAN":
        dbscan = DBSCAN()
        clusters = dbscan.fit_predict(self.df)
        self.df = self.df[clusters != -1]

      elif method == "IQR":
        Q1 = self.df.quantile(0.25)
        Q3 = self.df.quantile(0.75)
        IQR = Q3 - Q1
        self.df = self.df[~((self.df < (Q1 - 1.5 * IQR)) | (self.df > (Q3 + 1.5 * IQR))).any(axis=1)]

      elif method == "None":
        pass

      else:
        raise ValueError("Invalid method provided for outlier detection.")
      self.df['isFraud'] = y_outliers


    def split_data(self, test_size=0.2, random_state=42):
        # Splitting data into training and testing sets
        X = self.df.drop(columns=['isFraud'])
        y = self.df['isFraud']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
