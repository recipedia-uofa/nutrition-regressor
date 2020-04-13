import csv
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

INPUT_CSV = 'data_labeled.csv'

CSV_COLUMNS = [
    'id',
    'url',
    'title',
    'img_url',
    'score',
    'servings',
    'prep_time',
    'rating',
    'reviews',
    'made_it_count',
    'calories',
    'total_fat',
    'saturated_fat',
    'cholesterol',
    'sodium',
    'potassium',
    'carbs',
    'dietary_fiber',
    'protein',
    'sugars',
    'vitamin_a',
    'vitamin_c',
    'calcium',
    'iron',
    'thiamin',
    'niacin',
    'vitamin_b6',
    'magnesium',
    'folate',
    # Everything above this index is a <contains> relationship
]

CSV_COLUMN_TYPES = dict({
    'id': 'string',
    'url': 'string',
    'title': 'string',
    'img_url': 'string',
    'score': 'int32',
    'servings': 'int32',
    'prep_time': 'int32',
    'rating': 'float32',
    'reviews': 'int32',
    'made_it_count': 'int32',
    'calories': 'int32',
    'total_fat': 'float32',
    'saturated_fat': 'float32',
    'cholesterol': 'float32',
    'sodium': 'float32',
    'potassium': 'float32',
    'carbs': 'float32',
    'dietary_fiber': 'float32',
    'protein': 'float32',
    'sugars': 'float32',
    'vitamin_a': 'float32',
    'vitamin_c': 'float32',
    'calcium': 'float32',
    'iron': 'float32',
    'thiamin': 'float32',
    'niacin': 'float32',
    'vitamin_b6': 'float32',
    'magnesium': 'float32',
    'folate': 'float32',
    # Everything above this index is a <contains> relationship
})


def parse_data(data_csv, max_rows=None):
    with open(data_csv, 'r') as f:
        reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_ALL)

        col_len = len(CSV_COLUMNS)
        row_count = 0

        for row in reader:
            row_arr = [col.replace(r'"', r'\"') for col in row]
            yield row_arr[:col_len]
            row_count += 1

            if max_rows is not None and row_count >= max_rows:
                break


BASE_FEATURES = [
    'calories',
    'total_fat',
    'saturated_fat',
    'cholesterol',
    'sodium',
    'potassium',
    'carbs',
    'dietary_fiber',
    'protein',
    'sugars',
    'vitamin_a',
    'vitamin_c',
    'calcium',
    'iron',
    'thiamin',
    'niacin',
    'vitamin_b6',
    'magnesium',
    'folate',
]

RATE_FEATURES = [
    'rating',
    'reviews',
    'made_it_count',
]

# Augmented columns?
# Calorie percentage from protein
# Calorie percentage from fat
# Calorie percentage from carbs

AUG_FEATURES = [
    'cal_protein',
    'cal_fat',
    'cal_carbs'
]


def compute_aug_cols(row):
    protein_cals = row['protein'] * 4.0
    fat_cals = row['total_fat'] * 9.0
    carb_cals = row['carbs'] * 4.0

    total_cals = sum([protein_cals, fat_cals, carb_cals])
    row['cal_protein'] = protein_cals / total_cals
    row['cal_fat'] = fat_cals / total_cals
    row['cal_carbs'] = carb_cals / total_cals

    return row


# Select base features and augmented features only for training
FEATURES = BASE_FEATURES + AUG_FEATURES


# Split the data into a training, validation and test set
def train_val_test_split(data, train_frac, val_frac, test_frac):
    assert (train_frac + val_frac + test_frac == 1)
    M = data.shape[0]
    np.random.seed(628)
    np.random.shuffle(data)
    return np.split(data, [int(train_frac * M), int((train_frac + val_frac) * M)])


# Perform kfold analysis on the data.
# Returns a list of validation scores for each fold in the data
def kfold_analysis(model, X_, y_, cv=5):
    return cross_val_score(model, X_, y_, cv=cv)


# Find the data samples with the largest absolute error
# We are doing this because the data was manually labeled and may contain errors
def outlier_analysis(model, X_, y_):
    model.fit(X_, y_)
    y_hat = model.predict(X_)
    abs_error = np.abs(y_hat - y_)
    # abs_error, y, calories
    zipped = list(zip(abs_error, y_, X_[:, FEATURES.index('calories')]))
    return sorted(zipped, key=lambda x: x[0], reverse=True)


#################################################

def merge_dicts(*dicts):
    res = dicts[0].copy()
    for d in dicts[1:]:
        res.update(d)
    return res


def index_divide(l, i):
    for e in l[:i]:
        yield e
    for e in l[i + 1:]:
        yield e


def k_hold_split(split, i):
    train = np.concatenate(tuple(index_divide(split, i)), axis=0)
    val = split[i]
    return train, val


def k_hold_validation(model, D_split, y_split, k):
    """Returns the risk of a model based on k-hold validation"""
    K_1 = 1.0 / k
    acc_risk = 0

    for i in range(k):
        D_train, D_val = k_hold_split(D_split, i)
        y_train, y_val = k_hold_split(y_split, i)

        model.fit(D_train, y_train)
        acc_risk += risk(model, D_val, y_val)[0]

    # Return the average risk
    return K_1 * acc_risk


# Train the model using coordinate descent with k-hold validation
# If verbose is True, then everytime the algorithm finds better hyperparameters the risk
# and parameters will be printed to the console.
# Returns the best hyperparameters and risk
def fit_coordinate_descent(model_type, _X_label, _y_label, k=5, parameters=dict(), verbose=True):
    # Split the data into k parts
    D_split = np.array_split(_X_label, k)
    y_split = np.array_split(_y_label, k)

    var_params = []
    common_params = dict()
    for p, v in parameters.items():
        if type(v) is list:
            var_params.append(p)
        else:
            common_params[p] = v

    # Train with default hyperparameters
    model = model_type(**common_params)
    best_params = common_params  # default
    best_risk = k_hold_validation(model, D_split, y_split, k)

    if verbose:
        print(best_risk, 'default')

    fixed_params = {p: parameters[p][0] for p in var_params}

    # We try and fit each parameter one at a time
    for p_fit in var_params:
        for v in parameters[p_fit]:
            fitting_param = {p_fit: v}
            params = merge_dicts(common_params, fixed_params, fitting_param)
            model = model_type(**params)

            # Run a full k-hold validation to test the parameter combination
            iter_risk = k_hold_validation(model, D_split, y_split, k)

            if iter_risk < best_risk:
                # If the hyperparameters are better, save it to the fixed parameters list
                fixed_params[p_fit] = v
                best_risk = iter_risk
                best_params = params

                if verbose:
                    print(iter_risk, params)

    return best_params, best_risk

#################################################


if __name__ == "__main__":
    ###################################################

    # Create the data frame
    # Only the first 831 rows contain labelled data
    MAX_ROWS = 831
    df = pd.DataFrame(data=parse_data(INPUT_CSV, max_rows=MAX_ROWS), columns=CSV_COLUMNS)

    ###################################################

    # Filter data for desired columns and transform data type
    DF_COLUMNS = BASE_FEATURES + ['score']
    df = df[DF_COLUMNS]

    type_dict = {col: CSV_COLUMN_TYPES[col] for col in DF_COLUMNS}
    df = df.astype(type_dict)

    df = df[df['score'] >= 0]

    # Compute augmented columns
    df = df.apply(compute_aug_cols, axis=1)

    ###################################################

    # Turn pandas data frame into numpy data, then normalize
    X = df[FEATURES].to_numpy()
    y = df['score'].to_numpy()

    # normalize features
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    # normalize labels
    mean_y = np.mean(y)
    std_y = np.std(y)
    y = (y - np.mean(y)) / np.std(y)

    print("X shape: ", X.shape)
    print("y shape: ", y.shape)
    print()

    ###################################################

    # We use a 3:1:1 split for test train and validation
    X_train, X_val, X_test = train_val_test_split(X, 0.6, 0.2, 0.2)
    y_train, y_val, y_test = train_val_test_split(y, 0.6, 0.2, 0.2)

    # Combine the training and validation set since we plan on doing k-fold validation
    X_label = np.concatenate((X_train, X_val), axis=0)
    y_label = np.concatenate((y_train, y_val), axis=0)

    print("Training, validation and test set sizes")
    print(X_train.shape[0], X_val.shape[0], X_test.shape[0])
    print()

    ###################################################

    # The risk is the average absolute difference
    # between the predicted and true nutritional score
    # Defined here because we need std_y
    def risk(model, X_, y_):
        M_1 = 1.0 / X_.shape[0]
        y_hat = model.predict(X_)
        abs_error = np.abs(y_hat - y_) * std_y
        return M_1 * np.sum(abs_error), np.std(abs_error)

    ###################################################

    # Get the top 20 bad samples. Data should be unnormalized before running so that samples can be identified
    # outlier_analysis(LinearRegression(), X, y)[:20]

    ###################################################

    # Run k-fold cross validation analysis. We expect that the validation score for each fold
    # is roughly equal if the partitions are iid.
    print("K-fold Analysis")
    print(kfold_analysis(LinearRegression(), X_label, y_label))
    print()

    ###################################################

    # Implement trivial estimator (guesses mean) to get upper bound on acceptable error
    class TrivialEstimator:
        def __init__(self):
            self.mean = 0

        def fit(self, X_, y_):
            self.mean = np.mean(y_)

        def predict(self, _):
            return self.mean


    trivial = TrivialEstimator()
    trivial.fit(X_train, y_train)
    print("Trivial Estimator")
    print('Validation Risk', risk(trivial, X_val, y_val)[0])
    print()

    ###################################################

    from sklearn.linear_model import LinearRegression

    print("Fitting LinearRegression model...")
    best_linreg_params, best_linreg_risk = fit_coordinate_descent(LinearRegression, X_label, y_label)
    print()

    ###################################################

    from sklearn.linear_model import Ridge

    ridge_params = {
        'random_state': 628,
        'max_iter': [None, 100, 1000],
        'alpha': [0.01, 0.3, 1, 3, 10],  # Controls strength of l2 regularization
    }

    print("Fitting Ridge model...")
    best_ridge_params, best_ridge_risk = fit_coordinate_descent(Ridge, X_label, y_label, parameters=ridge_params)
    print()

    ###################################################

    from sklearn.linear_model import ElasticNet

    enet_params = {
        'random_state': 628,
        'max_iter': 10000,
        # controls weighted portion of L1 regularization. 1.0 means full L1 regularization
        'l1_ratio': [1.0, 0.7, 0.5, 0.3, 0.1, 0.01, 0.001],
        'alpha': [0.001, 0.01, 0.1],  # Controls strength of regularization
    }

    print("Fitting ElasticNet model...")
    best_enet_params, best_enet_risk = fit_coordinate_descent(ElasticNet, X_label, y_label, parameters=enet_params)
    print()

    ###################################################

    from sklearn.linear_model import TheilSenRegressor

    print("Fitting TheilSenRegressor...")
    best_ts_params, best_ts_risk = fit_coordinate_descent(TheilSenRegressor, X_label, y_label)
    print()

    ###################################################

    from sklearn.neural_network import MLPRegressor

    nn_params = {
        'random_state': 628,
        'verbose': False,
        'max_iter': 1000,
        'solver': ['adam', 'lbfgs', 'sgd'],  # adam solver, newton's method or stochastic gradient descent
        'activation': ['relu', 'identity'],
        'hidden_layer_sizes': [(100,), (len(FEATURES),), (len(FEATURES * 2),), (100, 100),
                               (len(FEATURES * 2), len(FEATURES * 2))],
        'alpha': [0.0001, 0.001, 0.01, 0.1],
    }

    print("Fitting Neural Network...")
    best_nn_params, best_nn_risk = fit_coordinate_descent(MLPRegressor, X_label, y_label, parameters=nn_params)
    print()

    ###################################################

    def to_summary_row(params):
        name, rsk = params
        return '{:10} | {:2.8f}'.format(name, rsk)

    summary_table = list(map(to_summary_row, [
        ('OLS', best_linreg_risk),
        ('Ridge', best_ridge_risk),
        ('ElasticNet', best_enet_risk),
        ('TheilSen', best_ts_risk),
        ('NeuralNet', best_nn_risk)
    ]))

    print("Summary\n{:10} | {:10}\n{}\n{}".format('Model Type', 'Val Risk', '-' * 20, '\n'.join(summary_table)))
    print()

    ###################################################

    print("Training Neural Network with params:")
    print(best_nn_params)
    reg_final = MLPRegressor(**best_nn_params).fit(X_label, y_label)
    print('Test Score: ', risk(reg_final, X_test, y_test))
    print()
