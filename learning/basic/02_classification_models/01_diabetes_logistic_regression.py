import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler


class DiabetesLogisticRegressionDemo:
    def __init__(self):
        self.feature_indices = [2, 3]
        self.feature_names = ["bmi", "bp"]
        self.random_state = 42
        self.scaler = StandardScaler()
        self.model = LogisticRegression(solver="lbfgs", max_iter=1000)
        self.X = None
        self.y = None
        self.y_binary = None
        self.X_selected = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.y_prob = None

    def load_dataset(self):
        ## load the diabetes dataset from sklearn
        ## this dataset has continuous target values, so later we convert it into classes
        diabetes = datasets.load_diabetes()
        self.X = diabetes.data
        self.y = diabetes.target

        print("Dataset loaded!")
        print(f"Data shape   : {self.X.shape}")
        print(f"Features     : {diabetes.feature_names}")
        print(f"Target range : {self.y.min():.1f} to {self.y.max():.1f}")

    def create_binary_target(self):
        ## logistic regression needs class labels, not continuous regression targets
        ## we create a binary target: 0 for low progression and 1 for high progression
        median_target = np.median(self.y)
        self.y_binary = (self.y > median_target).astype(int)

        print(f"\nMedian disease progression: {median_target:.1f}")
        print(f"Class distribution        : {np.bincount(self.y_binary)}")

    def prepare_features(self):
        ## use only bmi and blood pressure so the example stays simple and easy to visualize
        self.X_selected = self.X[:, self.feature_indices]

        ## standardization is important for logistic regression because feature scale affects optimization
        X_scaled = self.scaler.fit_transform(self.X_selected)

        ## split into training and test sets so evaluation happens on unseen data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled,
            self.y_binary,
            test_size=0.2,
            random_state=self.random_state,
            stratify=self.y_binary,
        )

        print(f"\nSelected features : {self.feature_names}")
        print(f"X train shape     : {self.X_train.shape}")
        print(f"X test shape      : {self.X_test.shape}")
        print(f"y train shape     : {self.y_train.shape}")
        print(f"y test shape      : {self.y_test.shape}")

    def train_model(self):
        ## fit logistic regression on the standardized training data
        ## the model learns weights for the decision boundary between class 0 and class 1
        self.model.fit(self.X_train, self.y_train)

        print("\n--- Logistic Regression Model ---")
        print(f"Learned w (BMI) : {self.model.coef_[0][0]:.4f}")
        print(f"Learned w (BP)  : {self.model.coef_[0][1]:.4f}")
        print(f"Learned b       : {self.model.intercept_[0]:.4f}")

    def evaluate_model(self):
        ## predict class labels and class probabilities on the test set
        self.y_pred = self.model.predict(self.X_test)
        self.y_prob = self.model.predict_proba(self.X_test)[:, 1]

        ## accuracy tells how many class labels were predicted correctly
        test_accuracy = accuracy_score(self.y_test, self.y_pred)
        train_accuracy = accuracy_score(self.y_train, self.model.predict(self.X_train))

        ## cross-validation gives a more reliable estimate than a single split
        cv_scores = cross_val_score(
            LogisticRegression(solver="lbfgs", max_iter=1000),
            self.scaler.transform(self.X_selected),
            self.y_binary,
            cv=5,
            scoring="accuracy",
        )

        print("\n--- Evaluation ---")
        print(f"Train accuracy         : {train_accuracy * 100:.2f}%")
        print(f"Test accuracy          : {test_accuracy * 100:.2f}%")
        print(f"5-fold CV accuracy     : {cv_scores.mean() * 100:.2f}%")
        print(f"5-fold CV accuracy std : {cv_scores.std() * 100:.2f}%")
        print("\nConfusion Matrix:")
        print(confusion_matrix(self.y_test, self.y_pred))
        print("\nClassification Report:")
        print(classification_report(self.y_test, self.y_pred, target_names=["Low", "High"]))

        print("First 10 test samples:")
        print("  BMI      BP       Actual  Predicted  Prob(High)")
        for i in range(min(10, len(self.X_test))):
            print(
                f"  {self.X_test[i][0]:6.2f}   {self.X_test[i][1]:6.2f}   "
                f"{self.y_test[i]:6d}   {self.y_pred[i]:9d}   {self.y_prob[i]:10.4f}"
            )

    def plot_decision_boundary(self):
        ## create a mesh grid so we can draw the probability surface learned by the classifier
        x_min, x_max = self.X_test[:, 0].min() - 1, self.X_test[:, 0].max() + 1
        y_min, y_max = self.X_test[:, 1].min() - 1, self.X_test[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 200),
            np.linspace(y_min, y_max, 200),
        )

        grid_prob = self.model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        grid_prob = grid_prob.reshape(xx.shape)

        plt.figure(figsize=(7, 5))
        plt.contourf(xx, yy, grid_prob, levels=30, cmap="RdBu_r", alpha=0.7)
        plt.colorbar(label="Probability of High Progression")
        plt.scatter(
            self.X_test[:, 0],
            self.X_test[:, 1],
            c=self.y_test,
            cmap="RdBu_r",
            edgecolors="k",
            s=60,
        )
        plt.xlabel("BMI (standardized)")
        plt.ylabel("Blood Pressure (standardized)")
        plt.title("Logistic Regression Decision Boundary")
        plt.tight_layout()
        plt.show()

    def run(self):
        self.load_dataset()
        self.create_binary_target()
        self.prepare_features()
        self.train_model()
        self.evaluate_model()
        self.plot_decision_boundary()


if __name__ == "__main__":
    demo = DiabetesLogisticRegressionDemo()
    demo.run()
