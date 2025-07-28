import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import seaborn as sns
from tabulate import tabulate
import sys
import os
import time



class RegressionSimulator:
    def __init__(self):
        self.data = None
        self.X = None
        self.y = None
        self.feature_names = []
        self.target_name = ""
        self.models = {}
        self.results = {}
        plt.style.use('seaborn-v0_8-darkgrid')
        
    def clear_screen(self):
        """Clear the terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
        
    def print_header(self):
        """Print the simulator header"""
        self.clear_screen()
        print("\n" + "="*70)
        print("                 INTERACTIVE REGRESSION SIMULATOR")
        print("="*70 + "\n")
        
    def load_sample_data(self):
        """Load built-in sample datasets"""
        self.print_header()
        print("Select a sample dataset:")
        print("  1. Housing Data (Multiple variables)")
        print("  2. Simple Linear Data (x-y relationship)")
        print("  3. Nonlinear Data (polynomial relationship)")
        print("  4. Return to main menu")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == '1':
            # Create a simple housing dataset
            np.random.seed(42)
            n_samples = 100
            area = np.random.uniform(500, 5000, n_samples)
            bedrooms = np.random.randint(1, 6, n_samples)
            bathrooms = np.random.randint(1, 4, n_samples)
            age = np.random.uniform(0, 50, n_samples)
            
            # Price = 50000 + 100*area + 25000*bedrooms + 30000*bathrooms - 2000*age + noise
            price = 50000 + 100*area + 25000*bedrooms + 30000*bathrooms - 2000*age + np.random.normal(0, 50000, n_samples)
            
            self.data = pd.DataFrame({
                'area': area,
                'bedrooms': bedrooms,
                'bathrooms': bathrooms,
                'age': age,
                'price': price
            })
            self.feature_names = ['area', 'bedrooms', 'bathrooms', 'age']
            self.target_name = 'price'
            
        elif choice == '2':
            # Create simple linear data
            n_samples = 100
            x = np.linspace(0, 10, n_samples)
            y = 2*x + 5 + np.random.normal(0, 1, n_samples)
            
            self.data = pd.DataFrame({
                'x': x,
                'y': y
            })
            self.feature_names = ['x']
            self.target_name = 'y'
            
        elif choice == '3':
            # Create nonlinear data
            n_samples = 100
            x = np.linspace(-5, 5, n_samples)
            y = 2*x**2 - 3*x + 1 + np.random.normal(0, 5, n_samples)
            
            self.data = pd.DataFrame({
                'x': x,
                'y': y
            })
            self.feature_names = ['x']
            self.target_name = 'y'
            
        elif choice == '4':
            return
            
        else:
            print("\nInvalid choice. Please try again.")
            time.sleep(1.5)
            self.load_sample_data()
            return
            
        print(f"\nLoaded dataset with {self.data.shape[0]} samples, {len(self.feature_names)} features")
        print(f"Features: {', '.join(self.feature_names)}")
        print(f"Target: {self.target_name}")
        print("\nPreview of the data:")
        print(self.data.head())
        
        input("\nPress Enter to continue...")
        self.prepare_data()
    
    def load_csv_data(self):
        """Load data from a CSV file"""
        self.print_header()
        
        print("To load CSV data, please provide the file path:")
        file_path = input("CSV file path: ")
        
        try:
            self.data = pd.read_csv(file_path)
            print(f"\nSuccessfully loaded data with {self.data.shape[0]} rows and {self.data.shape[1]} columns.")
            print("\nColumns in the dataset:")
            for i, col in enumerate(self.data.columns):
                print(f"  {i+1}. {col}")
            
            # Select target variable
            target_idx = int(input("\nSelect the target variable (enter the number): ")) - 1
            self.target_name = self.data.columns[target_idx]
            
            # Select feature variables
            print("\nSelect feature variables (comma-separated numbers, e.g., 1,2,3):")
            feature_idxs = [int(x.strip())-1 for x in input().split(',')]
            self.feature_names = [self.data.columns[i] for i in feature_idxs]
            
            print(f"\nSelected target: {self.target_name}")
            print(f"Selected features: {', '.join(self.feature_names)}")
            
            input("\nPress Enter to continue...")
            self.prepare_data()
            
        except Exception as e:
            print(f"\nError loading file: {e}")
            input("\nPress Enter to try again...")
            self.load_csv_data()
    
    def manually_enter_data(self):
        """Manually enter data points for regression"""
        self.print_header()
        
        try:
            num_features = int(input("Enter number of independent variables: "))
            num_samples = int(input("Enter number of data points: "))
            
            # Create empty dataframe
            columns = [f"x{i+1}" for i in range(num_features)] + ["y"]
            df = pd.DataFrame(columns=columns)
            
            print("\nEnter your data points:")
            for i in range(num_samples):
                print(f"\nData point {i+1}:")
                row = {}
                for j in range(num_features):
                    row[f"x{j+1}"] = float(input(f"  Enter value for x{j+1}: "))
                row["y"] = float(input("  Enter value for y: "))
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            
            self.data = df
            self.feature_names = [f"x{i+1}" for i in range(num_features)]
            self.target_name = "y"
            
            print("\nEntered data:")
            print(self.data)
            
            input("\nPress Enter to continue...")
            self.prepare_data()
            
        except Exception as e:
            print(f"\nError entering data: {e}")
            input("\nPress Enter to try again...")
            self.manually_enter_data()
    
    def prepare_data(self):
        """Prepare the data for regression analysis"""
        if self.data is None:
            print("No data loaded. Please load data first.")
            input("\nPress Enter to continue...")
            return
            
        self.X = self.data[self.feature_names].values
        self.y = self.data[self.target_name].values
        
        self.main_menu()
    
    def train_linear_model(self):
        """Train a linear regression model"""
        if self.X is None or self.y is None:
            print("No data prepared. Please load and prepare data first.")
            input("\nPress Enter to continue...")
            return
            
        self.print_header()
        print("Training Linear Regression Model...\n")
        
        # Train-test split
        test_size = 0.2
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=42)
        
        # Create and train the model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        # Store model and results
        self.models['Linear'] = model
        self.results['Linear'] = {
            'coefficients': model.coef_,
            'intercept': model.intercept_,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred
        }
        
        # Display results
        print("Model trained successfully!\n")
        print("Model Formula:")
        formula = f"{self.target_name} = {model.intercept_:.4f}"
        for i, feature in enumerate(self.feature_names):
            coef = model.coef_[i]
            sign = "+" if coef >= 0 else ""
            formula += f" {sign} {coef:.4f} * {feature}"
        print(formula)
        
        print("\nModel Performance:")
        print(f"  Training MSE: {train_mse:.4f}")
        print(f"  Test MSE: {test_mse:.4f}")
        print(f"  Training R²: {train_r2:.4f}")
        print(f"  Test R²: {test_r2:.4f}")
        
        input("\nPress Enter to continue...")
        self.main_menu()
    
    def train_polynomial_model(self):
        """Train a polynomial regression model"""
        if self.X is None or self.y is None:
            print("No data prepared. Please load and prepare data first.")
            input("\nPress Enter to continue...")
            return
            
        self.print_header()
        print("Training Polynomial Regression Model...\n")
        
        degree = int(input("Enter the polynomial degree (2-10): "))
        if degree < 2 or degree > 10:
            print("Degree must be between 2 and 10. Using degree = 2.")
            degree = 2
        
        # Train-test split
        test_size = 0.2
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=42)
        
        # Create polynomial features
        poly = PolynomialFeatures(degree=degree)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)
        
        # Create and train the model
        model = LinearRegression()
        model.fit(X_train_poly, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train_poly)
        y_test_pred = model.predict(X_test_poly)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        # Store model and results
        model_name = f"Polynomial (degree={degree})"
        self.models[model_name] = {
            'poly': poly,
            'linear': model
        }
        self.results[model_name] = {
            'coefficients': model.coef_,
            'intercept': model.intercept_,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred,
            'degree': degree
        }
        
        # Display results
        print("\nModel trained successfully!\n")
        print(f"Polynomial Regression (degree={degree})")
        
        # Only show formula for simple cases
        if self.X.shape[1] == 1 and degree <= 3:
            print("\nModel Formula:")
            formula = f"{self.target_name} = {model.intercept_:.4f}"
            feature = self.feature_names[0]
            for i in range(1, degree + 1):
                coef = model.coef_[i]
                sign = "+" if coef >= 0 else ""
                formula += f" {sign} {coef:.4f} * {feature}^{i}"
            print(formula)
        else:
            print("\nModel Formula: (Complex polynomial with multiple terms)")
        
        print("\nModel Performance:")
        print(f"  Training MSE: {train_mse:.4f}")
        print(f"  Test MSE: {test_mse:.4f}")
        print(f"  Training R²: {train_r2:.4f}")
        print(f"  Test R²: {test_r2:.4f}")
        
        input("\nPress Enter to continue...")
        self.main_menu()
    
    def compare_models(self):
        """Compare multiple regression models"""
        if not self.models:
            print("No models trained. Please train some models first.")
            input("\nPress Enter to continue...")
            return
            
        self.print_header()
        print("Model Comparison\n")
        
        # Prepare comparison table
        comparison_data = []
        for model_name, results in self.results.items():
            comparison_data.append([
                model_name,
                results['train_mse'],
                results['test_mse'],
                results['train_r2'],
                results['test_r2']
            ])
        
        # Sort by test R2 score (descending)
        comparison_data.sort(key=lambda x: x[4], reverse=True)
        
        # Display comparison table
        headers = ["Model", "Train MSE", "Test MSE", "Train R²", "Test R²"]
        print(tabulate(comparison_data, headers=headers, floatfmt=".4f"))
        
        # Visualize comparison
        if len(self.results) > 1:
            self.visualize_model_comparison()
        
        input("\nPress Enter to continue...")
        self.main_menu()
    
    def visualize_data(self):
        """Visualize the dataset"""
        if self.data is None:
            print("No data loaded. Please load data first.")
            input("\nPress Enter to continue...")
            return
            
        self.print_header()
        print("Data Visualization\n")
        
        # Check number of features
        if len(self.feature_names) == 1:
            # Simple scatter plot for one feature
            plt.figure(figsize=(10, 6))
            plt.scatter(self.data[self.feature_names[0]], self.data[self.target_name], 
                      alpha=0.7, c='blue', label='Data')
            plt.xlabel(self.feature_names[0])
            plt.ylabel(self.target_name)
            plt.title(f"{self.target_name} vs {self.feature_names[0]}")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.show()
            
        elif len(self.feature_names) == 2:
            # 3D plot for two features
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            x1 = self.data[self.feature_names[0]]
            x2 = self.data[self.feature_names[1]]
            y = self.data[self.target_name]
            
            ax.scatter(x1, x2, y, c='blue', marker='o', alpha=0.7)
            ax.set_xlabel(self.feature_names[0])
            ax.set_ylabel(self.feature_names[1])
            ax.set_zlabel(self.target_name)
            ax.set_title(f"{self.target_name} vs {self.feature_names[0]} and {self.feature_names[1]}")
            plt.tight_layout()
            plt.show()
            
        else:
            # Pair plot for multiple features
            selected_columns = self.feature_names + [self.target_name]
            subset_data = self.data[selected_columns]
            
            # Use only a sample if there are too many data points
            if len(subset_data) > 1000:
                subset_data = subset_data.sample(1000, random_state=42)
                
            sns.pairplot(subset_data, y_vars=[self.target_name], x_vars=self.feature_names,
                         height=3, aspect=1.5, plot_kws={'alpha': 0.6, 's': 30})
            plt.suptitle('Pairwise Relationships', y=1.02)
            plt.tight_layout()
            plt.show()
            
            # Correlation heatmap
            plt.figure(figsize=(10, 8))
            corr = subset_data.corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', 
                      square=True, linewidths=0.5)
            plt.title('Feature Correlation Heatmap')
            plt.tight_layout()
            plt.show()
        
        input("\nPress Enter to continue...")
        self.main_menu()
    
    def visualize_model_predictions(self):
        """Visualize model predictions"""
        if not self.models:
            print("No models trained. Please train some models first.")
            input("\nPress Enter to continue...")
            return
            
        self.print_header()
        print("Model Prediction Visualization\n")
        
        # Select a model
        print("Select a model to visualize:")
        for i, model_name in enumerate(self.models.keys()):
            print(f"  {i+1}. {model_name}")
        
        try:
            choice = int(input("\nEnter your choice: ")) - 1
            model_name = list(self.models.keys())[choice]
            results = self.results[model_name]
        except (ValueError, IndexError):
            print("Invalid choice. Please try again.")
            time.sleep(1.5)
            self.visualize_model_predictions()
            return
        
        # For simple linear regression with one feature
        if len(self.feature_names) == 1:
            plt.figure(figsize=(12, 8))
            
            # Plot training data
            plt.scatter(results['X_train'][:, 0], results['y_train'], 
                      alpha=0.7, c='blue', label='Training Data')
            
            # Plot test data
            plt.scatter(results['X_test'][:, 0], results['y_test'], 
                      alpha=0.7, c='green', label='Test Data')
            
            # Plot predictions
            if 'Linear' in model_name:
                # For linear model
                x_line = np.linspace(
                    min(self.data[self.feature_names[0]]), 
                    max(self.data[self.feature_names[0]]), 
                    100
                ).reshape(-1, 1)
                y_line = self.models[model_name].predict(x_line)
                plt.plot(x_line, y_line, 'r-', linewidth=2, label=f'{model_name} Fit')
                
            elif 'Polynomial' in model_name:
                # For polynomial model
                x_line = np.linspace(
                    min(self.data[self.feature_names[0]]), 
                    max(self.data[self.feature_names[0]]), 
                    100
                ).reshape(-1, 1)
                poly = self.models[model_name]['poly']
                linear = self.models[model_name]['linear']
                x_poly = poly.transform(x_line)
                y_line = linear.predict(x_poly)
                plt.plot(x_line, y_line, 'r-', linewidth=2, label=f'{model_name} Fit')
            
            plt.xlabel(self.feature_names[0])
            plt.ylabel(self.target_name)
            plt.title(f"{model_name} Regression: {self.target_name} vs {self.feature_names[0]}")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.show()
            
        elif len(self.feature_names) == 2 and 'Linear' in model_name:
            # 3D plot for two features (linear model only)
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Get data
            x1 = self.data[self.feature_names[0]]
            x2 = self.data[self.feature_names[1]]
            y = self.data[self.target_name]
            
            # Plot actual data points
            ax.scatter(results['X_train'][:, 0], results['X_train'][:, 1], results['y_train'], 
                     c='blue', marker='o', alpha=0.5, label='Training Data')
            ax.scatter(results['X_test'][:, 0], results['X_test'][:, 1], results['y_test'], 
                     c='green', marker='o', alpha=0.5, label='Test Data')
            
            # Create a meshgrid for the regression plane
            x1_min, x1_max = min(x1), max(x1)
            x2_min, x2_max = min(x2), max(x2)
            xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 20), 
                                  np.linspace(x2_min, x2_max, 20))
            grid_points = np.c_[xx1.ravel(), xx2.ravel()]
            
            # Predict values
            model = self.models[model_name]
            z_pred = model.predict(grid_points).reshape(xx1.shape)
            
            # Plot the regression plane
            surf = ax.plot_surface(xx1, xx2, z_pred, alpha=0.3, color='red', 
                                 rstride=1, cstride=1, linewidth=0, antialiased=True)
            
            ax.set_xlabel(self.feature_names[0])
            ax.set_ylabel(self.feature_names[1])
            ax.set_zlabel(self.target_name)
            ax.set_title(f"{model_name} Regression Plane")
            ax.legend()
            plt.tight_layout()
            plt.show()
            
        else:
            # For multiple features or polynomial with 2+ features
            # Show actual vs predicted plots
            plt.figure(figsize=(12, 6))
            
            # Training data
            plt.subplot(1, 2, 1)
            plt.scatter(results['y_train'], results['y_train_pred'], alpha=0.7)
            plt.plot([min(results['y_train']), max(results['y_train'])], 
                   [min(results['y_train']), max(results['y_train'])], 'r--')
            plt.xlabel("Actual Values")
            plt.ylabel("Predicted Values")
            plt.title("Training: Actual vs Predicted")
            plt.grid(True, alpha=0.3)
            
            # Test data
            plt.subplot(1, 2, 2)
            plt.scatter(results['y_test'], results['y_test_pred'], alpha=0.7)
            plt.plot([min(results['y_test']), max(results['y_test'])], 
                   [min(results['y_test']), max(results['y_test'])], 'r--')
            plt.xlabel("Actual Values")
            plt.ylabel("Predicted Values")
            plt.title("Test: Actual vs Predicted")
            plt.grid(True, alpha=0.3)
            
            plt.suptitle(f"{model_name} - Actual vs Predicted Values")
            plt.tight_layout()
            plt.show()
            
            # Residuals plot
            plt.figure(figsize=(12, 6))
            
            # Training residuals
            plt.subplot(1, 2, 1)
            train_residuals = results['y_train'] - results['y_train_pred']
            plt.scatter(results['y_train_pred'], train_residuals, alpha=0.7)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel("Predicted Values")
            plt.ylabel("Residuals")
            plt.title("Training: Residuals Plot")
            plt.grid(True, alpha=0.3)
            
            # Test residuals
            plt.subplot(1, 2, 2)
            test_residuals = results['y_test'] - results['y_test_pred']
            plt.scatter(results['y_test_pred'], test_residuals, alpha=0.7)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel("Predicted Values")
            plt.ylabel("Residuals")
            plt.title("Test: Residuals Plot")
            plt.grid(True, alpha=0.3)
            
            plt.suptitle(f"{model_name} - Residuals Analysis")
            plt.tight_layout()
            plt.show()
        
        input("\nPress Enter to continue...")
        self.main_menu()
    
    def visualize_model_comparison(self):
        """Visualize model comparison"""
        # Prepare data for comparison
        model_names = list(self.results.keys())
        train_r2 = [results['train_r2'] for results in self.results.values()]
        test_r2 = [results['test_r2'] for results in self.results.values()]
        train_mse = [results['train_mse'] for results in self.results.values()]
        test_mse = [results['test_mse'] for results in self.results.values()]
        
        # Sort all data by test R2 (descending)
        sorted_data = sorted(zip(model_names, train_r2, test_r2, train_mse, test_mse), 
                           key=lambda x: x[2], reverse=True)
        model_names = [x[0] for x in sorted_data]
        train_r2 = [x[1] for x in sorted_data]
        test_r2 = [x[2] for x in sorted_data]
        train_mse = [x[3] for x in sorted_data]
        test_mse = [x[4] for x in sorted_data]
        
        # Create bar plot for R2 scores
        plt.figure(figsize=(14, 6))
        
        x = np.arange(len(model_names))
        width = 0.35
        
        plt.bar(x - width/2, train_r2, width, label='Training R²', color='royalblue')
        plt.bar(x + width/2, test_r2, width, label='Test R²', color='lightcoral')
        
        plt.xlabel('Model')
        plt.ylabel('R² Score')
        plt.title('R² Score Comparison Across Models')
        plt.xticks(x, model_names, rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.ylim(0, 1.05)  # R² is typically between 0 and 1
        plt.tight_layout()
        plt.show()
        
        # Create bar plot for MSE
        plt.figure(figsize=(14, 6))
        
        plt.bar(x - width/2, train_mse, width, label='Training MSE', color='royalblue')
        plt.bar(x + width/2, test_mse, width, label='Test MSE', color='lightcoral')
        
        plt.xlabel('Model')
        plt.ylabel('Mean Squared Error')
        plt.title('MSE Comparison Across Models')
        plt.xticks(x, model_names, rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
    
    def predict_with_model(self):
        """Use a trained model to make predictions"""
        if not self.models:
            print("No models trained. Please train some models first.")
            input("\nPress Enter to continue...")
            return
            
        self.print_header()
        print("Make Predictions with Model\n")
        
        # Select a model
        print("Select a model to use for prediction:")
        for i, model_name in enumerate(self.models.keys()):
            print(f"  {i+1}. {model_name}")
        
        try:
            choice = int(input("\nEnter your choice: ")) - 1
            model_name = list(self.models.keys())[choice]
        except (ValueError, IndexError):
            print("Invalid choice. Please try again.")
            time.sleep(1.5)
            self.predict_with_model()
            return
        
        print(f"\nEnter values for {len(self.feature_names)} features:")
        feature_values = []
        for feature in self.feature_names:
            value = float(input(f"  Enter value for {feature}: "))
            feature_values.append(value)
        
        # Make prediction
        features = np.array(feature_values).reshape(1, -1)
        
        if 'Linear' in model_name:
            # Linear model
            prediction = self.models[model_name].predict(features)[0]
        else:
            # Polynomial model
            poly = self.models[model_name]['poly']
            linear = self.models[model_name]['linear']
            features_poly = poly.transform(features)
            prediction = linear.predict(features_poly)[0]
        
        print(f"\nPrediction for {self.target_name}: {prediction:.4f}")
        
        input("\nPress Enter to continue...")
        self.main_menu()
    
    def interactive_regression_explorer(self):
        """Interactive regression explorer with sliders"""
        if not self.models or len(self.feature_names) > 2:
            print("This feature works best with 1-2 features and at least one trained model.")
            input("\nPress Enter to continue...")
            return
            
        self.print_header()
        print("Interactive Regression Explorer\n")
        
        if len(self.feature_names) == 1:
            self._interactive_1d_explorer()
        elif len(self.feature_names) == 2:
            self._interactive_2d_explorer()
        else:
            print("Interactive explorer only supports 1 or 2 features.")
            input("\nPress Enter to continue...")
            
        self.main_menu()
    
    def _interactive_1d_explorer(self):
        """Interactive explorer for 1D regression"""
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(bottom=0.25)
        
        # Get data
        x = self.data[self.feature_names[0]].values
        y = self.data[self.target_name].values
        
        # Plot data points
        scatter = ax.scatter(x, y, c='blue', alpha=0.6, label='Data Points')
        
        # Create model lines dictionary
        model_lines = {}
        x_range = np.linspace(min(x), max(x), 200).reshape(-1, 1)
        
        for model_name, model in self.models.items():
            if 'Linear' in model_name:
                y_pred = model.predict(x_range)
                line, = ax.plot(x_range, y_pred, alpha=0.7, linewidth=3, label=model_name)
                model_lines[model_name] = line
            elif 'Polynomial' in model_name:
                poly = model['poly']
                linear = model['linear']
                x_poly = poly.transform(x_range)
                y_pred = linear.predict(x_poly)
                line, = ax.plot(x_range, y_pred, alpha=0.7, linewidth=3, label=model_name)
                model_lines[model_name] = line
        
        # Add custom polynomial slider
        ax_degree = plt.axes([0.25, 0.1, 0.65, 0.03])
        degree_slider = Slider(ax_degree, 'Custom Polynomial Degree', 1, 10, valinit=1, valstep=1)
        
        custom_poly_line, = ax.plot([], [], 'r-', linewidth=3, label='Custom Polynomial')
        
        def update_custom_poly(val):
            degree = int(degree_slider.val)
            poly = PolynomialFeatures(degree=degree)
            x_poly = poly.fit_transform(x.reshape(-1, 1))
            model = LinearRegression().fit(x_poly, y)
            
            # Calculate R² score
            y_pred = model.predict(x_poly)
            r2 = r2_score(y, y_pred)
            
            # Update plot
            x_range_reshaped = x_range.reshape(-1, 1)
            x_range_poly = poly.transform(x_range_reshaped)
            y_range_pred = model.predict(x_range_poly)
            
            custom_poly_line.set_data(x_range.flatten(), y_range_pred)
            custom_poly_line.set_label(f'Custom Polynomial (d={degree}, R²={r2:.4f})')
            ax.legend()
            fig.canvas.draw_idle()
        
        degree_slider.on_changed(update_custom_poly)
        update_custom_poly(1)  # Initialize with degree 1
        
        # Set labels and title
        ax.set_xlabel(self.feature_names[0])
        ax.set_ylabel(self.target_name)
        ax.set_title('Interactive Regression Explorer')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.show()
    
    def _interactive_2d_explorer(self):
        """Interactive explorer for 2D regression with a plane"""
        if 'Linear' not in self.models and not any('Polynomial' in m for m in self.models):
            print("Please train at least one linear or polynomial model first.")
            input("\nPress Enter to continue...")
            return
            
        # Setup 3D plot
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Get data
        x1 = self.data[self.feature_names[0]].values
        x2 = self.data[self.feature_names[1]].values
        y = self.data[self.target_name].values
        
        # Plot actual data points
        scatter = ax.scatter(x1, x2, y, c='blue', marker='o', alpha=0.6, label='Data Points')
        
        # Create meshgrid for surface
        x1_min, x1_max = min(x1), max(x1)
        x2_min, x2_max = min(x2), max(x2)
        xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 20), 
                             np.linspace(x2_min, x2_max, 20))
        grid_points = np.c_[xx1.ravel(), xx2.ravel()]
        
        # Select initial model (prefer linear if available)
        if 'Linear' in self.models:
            model_name = 'Linear'
            model = self.models[model_name]
            z_pred = model.predict(grid_points).reshape(xx1.shape)
        else:
            # Take the first polynomial model
            model_name = next(m for m in self.models if 'Polynomial' in m)
            poly = self.models[model_name]['poly']
            linear = self.models[model_name]['linear']
            grid_poly = poly.transform(grid_points)
            z_pred = linear.predict(grid_poly).reshape(xx1.shape)
        
        # Plot surface
        surf = ax.plot_surface(xx1, xx2, z_pred, alpha=0.5, cmap='viridis', 
                            antialiased=True, linewidth=0)
        
        # Add title and labels
        ax.set_xlabel(self.feature_names[0])
        ax.set_ylabel(self.feature_names[1])
        ax.set_zlabel(self.target_name)
        ax.set_title(f'Interactive 3D Regression Surface - {model_name}')
        
        # Create model selector
        ax_model = plt.axes([0.25, 0.02, 0.65, 0.03])
        model_slider = Slider(
            ax_model, 'Model', 0, len(self.models)-1, 
            valinit=list(self.models.keys()).index(model_name) if model_name in self.models else 0, 
            valstep=1
        )
        
        def update_model(val):
            # Clear the current surface
            for c in ax.collections:
                if isinstance(c, plt.cm.ScalarMappable):
                    c.remove()
            
            # Get selected model
            idx = int(model_slider.val)
            model_name = list(self.models.keys())[idx]
            
            # Make predictions with selected model
            if 'Linear' in model_name:
                model = self.models[model_name]
                z_pred = model.predict(grid_points).reshape(xx1.shape)
            else:
                poly = self.models[model_name]['poly']
                linear = self.models[model_name]['linear']
                grid_poly = poly.transform(grid_points)
                z_pred = linear.predict(grid_poly).reshape(xx1.shape)
            
            # Update surface
            surf = ax.plot_surface(xx1, xx2, z_pred, alpha=0.5, cmap='viridis', 
                                antialiased=True, linewidth=0)
            
            # Update title
            ax.set_title(f'Interactive 3D Regression Surface - {model_name}')
            
            fig.canvas.draw_idle()
        
        model_slider.on_changed(update_model)
        update_model(model_slider.val)  # Initialize
        
        plt.show()
    
    def residual_analysis(self):
        """Perform detailed residual analysis"""
        if not self.models:
            print("No models trained. Please train some models first.")
            input("\nPress Enter to continue...")
            return
            
        self.print_header()
        print("Residual Analysis\n")
        
        # Select a model
        print("Select a model to analyze:")
        for i, model_name in enumerate(self.models.keys()):
            print(f"  {i+1}. {model_name}")
        
        try:
            choice = int(input("\nEnter your choice: ")) - 1
            model_name = list(self.models.keys())[choice]
            results = self.results[model_name]
        except (ValueError, IndexError):
            print("Invalid choice. Please try again.")
            time.sleep(1.5)
            self.residual_analysis()
            return
        
        # Create figure
        plt.figure(figsize=(15, 10))
        
        # Calculate residuals
        train_residuals = results['y_train'] - results['y_train_pred']
        test_residuals = results['y_test'] - results['y_test_pred']
        
        # Residuals vs Predicted
        plt.subplot(2, 2, 1)
        plt.scatter(results['y_train_pred'], train_residuals, alpha=0.6, c='blue', label='Training')
        plt.scatter(results['y_test_pred'], test_residuals, alpha=0.6, c='red', label='Test')
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Predicted Values')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Histogram of residuals
        plt.subplot(2, 2, 2)
        plt.hist(train_residuals, bins=20, alpha=0.5, color='blue', label='Training')
        plt.hist(test_residuals, bins=20, alpha=0.5, color='red', label='Test')
        plt.xlabel('Residual Value')
        plt.ylabel('Frequency')
        plt.title('Distribution of Residuals')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Q-Q Plot
        plt.subplot(2, 2, 3)
        from scipy import stats
        
        # Training data QQ plot
        stats.probplot(train_residuals, plot=plt)
        plt.title('Q-Q Plot (Training Residuals)')
        plt.grid(True, alpha=0.3)
        
        # Residuals vs Feature (for first feature only)
        plt.subplot(2, 2, 4)
        feat_idx = 0  # Use first feature
        plt.scatter(results['X_train'][:, feat_idx], train_residuals, alpha=0.6, c='blue', label='Training')
        plt.scatter(results['X_test'][:, feat_idx], test_residuals, alpha=0.6, c='red', label='Test')
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.xlabel(f'Feature: {self.feature_names[feat_idx]}')
        plt.ylabel('Residuals')
        plt.title(f'Residuals vs {self.feature_names[feat_idx]}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.suptitle(f'Residual Analysis for {model_name}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
        
        # Print residual statistics
        print("\nResidual Statistics:")
        print(f"  Training Mean Residual: {np.mean(train_residuals):.4f}")
        print(f"  Test Mean Residual: {np.mean(test_residuals):.4f}")
        print(f"  Training Residual Std Dev: {np.std(train_residuals):.4f}")
        print(f"  Test Residual Std Dev: {np.std(test_residuals):.4f}")
        
        # Durbin-Watson test for autocorrelation
        from statsmodels.stats.stattools import durbin_watson
        dw_stat = durbin_watson(train_residuals)
        print(f"\nDurbin-Watson Statistic: {dw_stat:.4f}")
        print("  (Values close to 2 indicate no autocorrelation,")
        print("   Values <1 or >3 indicate potential autocorrelation)")
        
        input("\nPress Enter to continue...")
        self.main_menu()
    
    def feature_importance_analysis(self):
        """Analyze feature importance"""
        if not self.models or 'Linear' not in self.models:
            print("This analysis requires a linear regression model.")
            input("\nPress Enter to continue...")
            return
            
        self.print_header()
        print("Feature Importance Analysis\n")
        
        # Use the linear model
        model = self.models['Linear']
        
        # Calculate standardized coefficients
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        model_scaled = LinearRegression()
        model_scaled.fit(X_scaled, self.y)
        
        # Get coefficients and prepare data
        coeffs = model.coef_
        std_coeffs = model_scaled.coef_
        
        # Combine feature names with coefficients
        feature_importance = sorted(zip(self.feature_names, coeffs, std_coeffs), 
                                  key=lambda x: abs(x[2]), reverse=True)
        
        # Print results
        print("Feature importance (based on standardized coefficients):")
        print("-" * 60)
        print(f"{'Feature':<20} {'Coefficient':<15} {'Std. Coefficient':<20} {'Abs. Importance':<15}")
        print("-" * 60)
        
        for feature, coef, std_coef in feature_importance:
            print(f"{feature:<20} {coef:<15.4f} {std_coef:<20.4f} {abs(std_coef):<15.4f}")
        
        # Visualize feature importance
        features = [x[0] for x in feature_importance]
        importances = [abs(x[2]) for x in feature_importance]
        
        plt.figure(figsize=(10, 6))
        plt.barh(features, importances, color='skyblue')
        plt.xlabel('Absolute Standardized Coefficient')
        plt.ylabel('Feature')
        plt.title('Feature Importance')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
        
        input("\nPress Enter to continue...")
        self.main_menu()
    
    def cross_validation_analysis(self):
        """Perform cross-validation analysis"""
        if self.X is None or self.y is None:
            print("No data prepared. Please load and prepare data first.")
            input("\nPress Enter to continue...")
            return
            
        self.print_header()
        print("Cross-Validation Analysis\n")
        
        from sklearn.model_selection import cross_val_score, KFold
        
        # Get number of folds
        try:
            n_folds = int(input("Enter number of folds for cross-validation (2-10): "))
            if n_folds < 2 or n_folds > 10:
                print("Number of folds must be between 2 and 10. Using 5 folds.")
                n_folds = 5
        except ValueError:
            print("Invalid input. Using 5 folds.")
            n_folds = 5
        
        # Create KFold object
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        # Select models to evaluate
        print("\nSelect models to evaluate (comma-separated numbers):")
        for i, model_name in enumerate(self.models.keys()):
            print(f"  {i+1}. {model_name}")
        
        try:
            choices = input("\nEnter your choices (or 'all' for all models): ")
            
            if choices.lower() == 'all':
                selected_models = list(self.models.keys())
            else:
                indices = [int(x.strip())-1 for x in choices.split(',')]
                selected_models = [list(self.models.keys())[i] for i in indices]
        except:
            print("Invalid input. Evaluating all models.")
            selected_models = list(self.models.keys())
        
        # Results storage
        cv_results = {}
        
        # Perform cross-validation
        for model_name in selected_models:
            print(f"\nEvaluating {model_name}...")
            
            if 'Linear' in model_name:
                model = LinearRegression()
                scores = cross_val_score(model, self.X, self.y, cv=kf, scoring='r2')
                neg_mse_scores = cross_val_score(model, self.X, self.y, cv=kf, scoring='neg_mean_squared_error')
                mse_scores = -neg_mse_scores
                
            elif 'Polynomial' in model_name:
                degree = self.results[model_name]['degree']
                poly = PolynomialFeatures(degree=degree)
                X_poly = poly.fit_transform(self.X)
                
                model = LinearRegression()
                scores = cross_val_score(model, X_poly, self.y, cv=kf, scoring='r2')
                neg_mse_scores = cross_val_score(model, X_poly, self.y, cv=kf, scoring='neg_mean_squared_error')
                mse_scores = -neg_mse_scores
            
            cv_results[model_name] = {
                'r2_scores': scores,
                'mse_scores': mse_scores,
                'mean_r2': np.mean(scores),
                'std_r2': np.std(scores),
                'mean_mse': np.mean(mse_scores),
                'std_mse': np.std(mse_scores)
            }
        
        # Print results
        print("\nCross-Validation Results:")
        print("-" * 70)
        print(f"{'Model':<25} {'Mean R²':<10} {'Std R²':<10} {'Mean MSE':<15} {'Std MSE':<10}")
        print("-" * 70)
        
        # Sort by mean R2 score
        sorted_results = sorted(cv_results.items(), key=lambda x: x[1]['mean_r2'], reverse=True)
        
        for model_name, results in sorted_results:
            print(f"{model_name:<25} {results['mean_r2']:<10.4f} {results['std_r2']:<10.4f} "
                  f"{results['mean_mse']:<15.4f} {results['std_mse']:<10.4f}")
        
        # Visualize CV results
        plt.figure(figsize=(12, 6))
        
        # R2 scores
        plt.subplot(1, 2, 1)
        model_names = [m for m, _ in sorted_results]
        mean_r2 = [r['mean_r2'] for _, r in sorted_results]
        std_r2 = [r['std_r2'] for _, r in sorted_results]
        
        x = np.arange(len(model_names))
        plt.bar(x, mean_r2, yerr=std_r2, capsize=10, alpha=0.7)
        plt.xlabel('Model')
        plt.ylabel('R² Score')
        plt.title('Cross-Validation R² Scores')
        plt.xticks(x, model_names, rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.ylim(0, 1.1)
        
        # MSE scores
        plt.subplot(1, 2, 2)
        mean_mse = [r['mean_mse'] for _, r in sorted_results]
        std_mse = [r['std_mse'] for _, r in sorted_results]
        
        plt.bar(x, mean_mse, yerr=std_mse, capsize=10, alpha=0.7, color='orange')
        plt.xlabel('Model')
        plt.ylabel('Mean Squared Error')
        plt.title('Cross-Validation MSE Scores')
        plt.xticks(x, model_names, rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.suptitle(f'{n_folds}-Fold Cross-Validation Results', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
        
        input("\nPress Enter to continue...")
        self.main_menu()
    
    def learning_curve_analysis(self):
        """Generate learning curves to analyze model performance vs training size"""
        if self.X is None or self.y is None:
            print("No data prepared. Please load and prepare data first.")
            input("\nPress Enter to continue...")
            return
            
        self.print_header()
        print("Learning Curve Analysis\n")
        
        from sklearn.model_selection import learning_curve
        
        # Select model
        print("Select a model to analyze:")
        for i, model_name in enumerate(self.models.keys()):
            print(f"  {i+1}. {model_name}")
        
        try:
            choice = int(input("\nEnter your choice: ")) - 1
            model_name = list(self.models.keys())[choice]
        except (ValueError, IndexError):
            print("Invalid choice. Please try again.")
            time.sleep(1.5)
            self.learning_curve_analysis()
            return
        
        # Create model instance for learning curve
        if 'Linear' in model_name:
            model = LinearRegression()
            X_data = self.X
            
        elif 'Polynomial' in model_name:
            degree = self.results[model_name]['degree']
            poly = PolynomialFeatures(degree=degree)
            X_data = poly.fit_transform(self.X)
            model = LinearRegression()
        
        # Generate learning curve data
        train_sizes = np.linspace(0.1, 1.0, 10)
        train_sizes, train_scores, test_scores = learning_curve(
            model, X_data, self.y, 
            train_sizes=train_sizes, cv=5, 
            scoring='r2', n_jobs=-1
        )
        
        # Calculate mean and std
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        # Plot learning curve
        plt.figure(figsize=(10, 6))
        plt.grid(True, alpha=0.3)
        
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                       alpha=0.1, color='blue')
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, 
                       alpha=0.1, color='orange')
        
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
        plt.plot(train_sizes, test_mean, 'o-', color='orange', label='Cross-Val Score')
        
        plt.xlabel('Training Set Size')
        plt.ylabel('R² Score')
        plt.title(f'Learning Curve - {model_name}')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()
        
        # Interpret results
        print("\nLearning Curve Interpretation:")
        
        # Check convergence
        final_gap = train_mean[-1] - test_mean[-1]
        
        print(f"  Final training score: {train_mean[-1]:.4f}")
        print(f"  Final validation score: {test_mean[-1]:.4f}")
        print(f"  Gap between training and validation: {final_gap:.4f}")
        
        if final_gap > 0.2:
            print("\nThe model shows signs of overfitting (high variance):")
            print("  - Training score is significantly higher than validation score")
            print("  - Consider using regularization or reducing model complexity")
        
        elif test_mean[-1] < 0.5:
            print("\nThe model shows signs of underfitting (high bias):")
            print("  - Both training and validation scores are low")
            print("  - Consider increasing model complexity or adding features")
        
        else:
            print("\nThe model shows good balance between bias and variance:")
            print("  - Training and validation scores are reasonably close")
            print("  - The model generalizes well to unseen data")
        
        slope = (test_mean[-1] - test_mean[-2]) / (train_sizes[-1] - train_sizes[-2])
        if slope > 0.01:
            print("\nThe learning curve is still rising:")
            print("  - The model might benefit from more training data")
        
        input("\nPress Enter to continue...")
        self.main_menu()
    
    def export_model(self):
        """Export a trained model to a file"""
        if not self.models:
            print("No models trained. Please train some models first.")
            input("\nPress Enter to continue...")
            return
            
        self.print_header()
        print("Export Model\n")
        
        # Select a model
        print("Select a model to export:")
        for i, model_name in enumerate(self.models.keys()):
            print(f"  {i+1}. {model_name}")
        
        try:
            choice = int(input("\nEnter your choice: ")) - 1
            model_name = list(self.models.keys())[choice]
        except (ValueError, IndexError):
            print("Invalid choice. Please try again.")
            time.sleep(1.5)
            self.export_model()
            return
        
        # Get file path
        file_path = input("\nEnter file path to save model (e.g., model.pkl): ")
        
        # Export model using pickle
        import pickle
        
        try:
            with open(file_path, 'wb') as f:
                model_data = {
                    'name': model_name,
                    'model': self.models[model_name],
                    'feature_names': self.feature_names,
                    'target_name': self.target_name,
                    'results': self.results[model_name]
                }
                pickle.dump(model_data, f)
            
            print(f"\nModel '{model_name}' successfully exported to {file_path}")
        except Exception as e:
            print(f"\nError exporting model: {e}")
        
        input("\nPress Enter to continue...")
        self.main_menu()
    
    def generate_report(self):
        """Generate a comprehensive analysis report"""
        if not self.models:
            print("No models trained. Please train some models first.")
            input("\nPress Enter to continue...")
            return
            
        self.print_header()
        print("Generate Analysis Report\n")
        
        # Select file path
        file_path = input("Enter file path to save report (e.g., report.txt): ")
        
        try:
            with open(file_path, 'w') as f:
                # Report header
                f.write("=" * 80 + "\n")
                f.write("                     REGRESSION ANALYSIS REPORT\n")
                f.write("=" * 80 + "\n\n")
                
                # Dataset information
                f.write("DATASET INFORMATION\n")
                f.write("-" * 50 + "\n")
                f.write(f"Number of samples: {self.data.shape[0]}\n")
                f.write(f"Features: {', '.join(self.feature_names)}\n")
                f.write(f"Target variable: {self.target_name}\n\n")
                
                # Basic statistics
                f.write("STATISTICAL SUMMARY\n")
                f.write("-" * 50 + "\n")
                f.write(self.data.describe().to_string() + "\n\n")
                
                # Feature correlation with target
                f.write("FEATURE CORRELATION WITH TARGET\n")
                f.write("-" * 50 + "\n")
                for feature in self.feature_names:
                    corr = np.corrcoef(self.data[feature], self.data[self.target_name])[0, 1]
                    f.write(f"{feature}: {corr:.4f}\n")
                f.write("\n")
                
                # Model comparison
                f.write("MODEL COMPARISON\n")
                f.write("-" * 50 + "\n")
                f.write(f"{'Model':<25} {'Train MSE':<15} {'Test MSE':<15} {'Train R²':<15} {'Test R²':<15}\n")
                f.write("-" * 85 + "\n")
                
                # Sort models by test R2 score
                sorted_models = sorted(self.results.items(), key=lambda x: x[1]['test_r2'], reverse=True)
                
                for model_name, results in sorted_models:
                    f.write(f"{model_name:<25} {results['train_mse']:<15.4f} {results['test_mse']:<15.4f} {results['train_r2']:<15.4f} {results['test_r2']:<15.4f}\n")
                
                f.write("\n")
                
                # Detailed model information
                f.write("DETAILED MODEL INFORMATION\n")
                f.write("-" * 50 + "\n")
                
                for model_name, results in sorted_models:
                    f.write(f"\nMODEL: {model_name}\n")
                    f.write("-" * 20 + "\n")
                    
                    # Write model formula for linear models
                    if 'Linear' in model_name:
                        f.write("Formula: ")
                        formula = f"{self.target_name} = {results['intercept']:.4f}"
                        for i, feature in enumerate(self.feature_names):
                            coef = results['coefficients'][i]
                            sign = "+" if coef >= 0 else ""
                            formula += f" {sign} {coef:.4f} * {feature}"
                        f.write(f"{formula}\n\n")
                    
                    # Performance metrics
                    f.write(f"Train MSE: {results['train_mse']:.4f}\n")
                    f.write(f"Test MSE: {results['test_mse']:.4f}\n")
                    f.write(f"Train R²: {results['train_r2']:.4f}\n")
                    f.write(f"Test R²: {results['test_r2']:.4f}\n")
                    
                    # Polynomial degree if applicable
                    if 'degree' in results:
                        f.write(f"Polynomial Degree: {results['degree']}\n")
                    
                    f.write("\n")
                
                # Conclusion
                f.write("CONCLUSION\n")
                f.write("-" * 50 + "\n")
                best_model = sorted_models[0][0]
                best_r2 = sorted_models[0][1]['test_r2']
                f.write(f"The best performing model is {best_model} with a test R² score of {best_r2:.4f}.\n\n")
                
                if best_r2 > 0.8:
                    f.write("This model shows strong predictive performance.\n")
                elif best_r2 > 0.5:
                    f.write("This model shows moderate predictive performance.\n")
                else:
                    f.write("This model shows weak predictive performance. Consider exploring additional features or different modeling approaches.\n")
                
            print(f"\nReport successfully generated and saved to {file_path}")
        except Exception as e:
            print(f"\nError generating report: {e}")
        
        input("\nPress Enter to continue...")
        self.main_menu()

