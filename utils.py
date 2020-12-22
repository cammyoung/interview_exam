import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def read_json(file_name):
    """
    Read in a JSON file formatted as a list of lists of dicts.
    
    Parameters
    ----------
    file_name : str
        File path to the JSON file
    
    Returns
    -------
    A list of lists of dicts
    """
    
    with open(file_name) as f:
        data = json.load(f)
    
    return data

def json_to_pandas(json_data):
    """
    Convert a list of list of dicts into a Pandas DataFrame
    
    Parameters
    ----------
    json_data : list of lists of dicts
        Data to be converted to a Pandas DataFrame
    
    Returns
    -------
    A DataFrame
    """
    
    df_list = []
    for n, item in enumerate(json_data):
        df = pd.DataFrame(item)
        df['container_id'] = n

        df_list.append(df)

    data = pd.concat(df_list)
    
    data['measurement_id'] = data.index
    
    data = data.reset_index(drop=True)
    
    return data

def scale_data(train, test, select_cols):
    """
    Fit a sklearn.preprocessing.StandardScaler to the training data,
    and then apply that transform to the train and test data
    
    Parameters
    ----------
    train : DataFrame or array
        Training data
    test : DataFrame or array
        Testing data
    select_cols : list
        Columns to scale.
        
    Returns
    -------
    DataFrame, DataFrame, StandardScaler
        Return the training and testing data with added scaled columns,
        and the scaler in order to deploy to a production pipeline.
    """
    
    scl = StandardScaler()
    scl.fit(train[select_cols])
    
    train_scl = scl.transform(train[select_cols])
    test_scl = scl.transform(test[select_cols])
    
    new_cols = [f'{i}_scl' for i in select_cols]
    
    train[new_cols] = pd.DataFrame(train_scl)
    test[new_cols] = pd.DataFrame(test_scl)
    
    return train, test, scl

def preprocess_data(df, max_lag=4):
    """
    Apply preprocessing steps to
        * Turn the time column into a datetime type
        * Create columns with the time difference between a row previous rows
        * Create a numeric label to fit models against
    
    Parameters
    ----------
    df : DataFrame
        Training or testing data
    max_lag : int
        Maximum number of lagged time difference columns to create
    select_cols : list
        Columns to scale.
        
    Returns
    -------
    DataFrame
        Return the data with processed and additional features.
    """
    
    label_dict = {'UNLOAD_SHIP': 0,
                  'LOAD_TRAIN': 1,
                  'NOISE': 2,
                  'DEPART_TRAIN': 3}
    
    df['TIME'] = pd.to_datetime(df.TIME)
    
    df = df.sort_values(['container_id', 'measurement_id'])
    
    for i in np.arange(1, max_lag + 1):
        df[f'time_diff{i}'] = (df.TIME - df.groupby('container_id').TIME.shift(i)).astype('timedelta64[s]')
        df[f'time_diff{i}'] = df[f'time_diff{i}'].fillna(df[f'time_diff{i}'].mean())
    
    df['y'] = df.LABEL.map(label_dict)
    
    return df

def load_and_process_data(max_lag=4):
    """
    Read in the JSON files, convert them to DataFrames, and then preprocess
    and scale the inputs.
    
    Parameters
    ----------
    None
        
    Returns
    -------
    DataFrame, DataFrame, StandardScaler, list
        Return the training data, testing data, scaler, and selected columns.
    """
    
    print('Reading files')
    train_json = read_json('train.json')
    test_json = read_json('test.json')
    
    print('Converting files to pandas DataFrames')
    train = json_to_pandas(train_json)
    test = json_to_pandas(test_json)
    
    print('Preprocessing data')
    train = preprocess_data(train, max_lag=max_lag)
    test = preprocess_data(test, max_lag=max_lag)
    
    select_cols = ['SENSOR_A', 'SENSOR_B'] + [f'time_diff{i}' for i in np.arange(1, max_lag + 1)]
    
    print('Scaling data')
    train, test, scl = scale_data(train, test, select_cols)
    
    return train, test, scl, select_cols

def plot_2d_kde(x, y, hue, data):
    """
    Plot a bivariate kernel density estimate
    
    Parameters
    ----------
    x : array
        x-axis variable
    y : array
        y-axis variable
    hue : array
        Variable to map to colors in order to visually distinguish separate bivariate densities
        
    Returns
    -------
    None
    """
    
    plt.figure(figsize=(16,16))
    sns.displot(x=x,
                y=y,
                hue=hue,
                kind='kde',
                data=data)
    
    plt.show()
    
def plot_time_series(x, y, hue, data):
    plt.figure(figsize=(16,8))
    sns.scatterplot(x=x,
                    y=y,
                    hue=hue,
                    data=data)
    
    plt.show()

def model_performance(actual, prediction, target_names=['UNLOAD_SHIP', 'LOAD_TRAIN', 'NOISE', 'DEPART_TRAIN']):
    """
    Print out model performance statistics.
    From https://towardsdatascience.com/confusion-matrix-for-your-multi-class-machine-learning-model-ff9aa3bf7826
    
    Parameters
    ----------
    actual : array
        Ground truth labels
    prediction : array
        Predicted labels
        
    Returns
    -------
    
    """
    
    confusion = confusion_matrix(actual, prediction)
    print('Confusion Matrix\n')
    print(confusion)
    
    print('\nAccuracy: {:.2f}\n'.format(accuracy_score(actual, prediction)))

    print('Micro Precision: {:.2f}'.format(precision_score(actual, prediction, average='micro')))
    print('Micro Recall: {:.2f}'.format(recall_score(actual, prediction, average='micro')))
    print('Micro F1-score: {:.2f}\n'.format(f1_score(actual, prediction, average='micro')))

    print('Macro Precision: {:.2f}'.format(precision_score(actual, prediction, average='macro')))
    print('Macro Recall: {:.2f}'.format(recall_score(actual, prediction, average='macro')))
    print('Macro F1-score: {:.2f}\n'.format(f1_score(actual, prediction, average='macro')))

    print('Weighted Precision: {:.2f}'.format(precision_score(actual, prediction, average='weighted')))
    print('Weighted Recall: {:.2f}'.format(recall_score(actual, prediction, average='weighted')))
    print('Weighted F1-score: {:.2f}'.format(f1_score(actual, prediction, average='weighted')))
    
    print('\nClassification Report\n')
    print(classification_report(actual, prediction, target_names=target_names))

def fit_predict(train, test, select_cols, params):
    """
    Fit an XGBoost classification model to the train data, generate predictions,
    and print test performance.
    
    Parameters
    ----------
    actual : array
        Ground truth labels
    prediction : array
        Predicted labels
        
    Returns
    -------
    List of an XGBoost model and NumPy arrays
        Fit model, train predictions, train predicted probabilities, test predictions,
        test predicted probabilities
    """
    
    clf = xgb.XGBClassifier(**params)
    
    X_train = train[select_cols]
    y_train = train['y']
    X_test = test[select_cols]
    y_test = test['y']
    
    clf.fit(X_train, y_train)
    
    train_pred = clf.predict(X_train)
    train_probs = clf.predict_proba(X_train)
    test_pred = clf.predict(X_test)
    test_probs = clf.predict_proba(X_test)
    
    model_performance(y_test, test_pred)
    
    return clf, train_pred, train_probs, test_pred, test_probs

def create_filter_cols(train, test):
    """
    Instantiate columns of zeros. Columns named 'pred_*' capture the raw one-vs-rest
    predictions. Columns named 'past_*' capture whether or not the model has already
    predicted that label for that container, which will be used to limit the data
    passed into each successive one-vs-rest model.
    
    Parameters
    ----------
    train : DataFrame
        Training data
    test : DataFrame
        Testing data
        
    Returns
    -------
    List of DataFrames
        Training and testing data with zero-columns instantiated
    """
    
    labs = [i for i in train.LABEL.unique() if i != 'NOISE']
    cols = ['past_' + i for i in labs] + ['pred_' + i for i in labs]
    
    for df in [train, test]:
        for col in cols:
            df[col] = 0
    
    return train, test

def update_filter_cols(data, prediction, label, prev_label=None):
    """
    Update the columns instantiated in create_filter_cols(). Columns named 'pred_*'
    are updated with the model predictions, and columns named 'past_*' are assigned a
    lagged cumulative sum within each container. This allows the next measurement for
    each container to be associated with the past predictions of one-vs-rest models.
    
    Parameters
    ----------
    data : DataFrame
        Train or test data
    prediction : array
        Predicted labels
    label : str
        Which label the positive predictions correspond to
    prev_label : None or str
        The previous label in the ordered list
        
    Returns
    -------
    List of an XGBoost model and NumPy arrays
        Fit model, train predictions, train predicted probabilities, test predictions,
        test predicted probabilities
    """
    
    ## Assign predictions, filtering data that was not trained on if necessary
    if prev_label is not None:
        data.loc[(data[f'past_{label}'] == 0) & (data[f'past_{prev_label}'] == 1), f'pred_{label}'] = prediction
    else:
        data.loc[data[f'past_{label}'] == 0, f'pred_{label}'] = prediction
    
    data[f'past_{label}'] = data.groupby('container_id')[f'pred_{label}'].shift(1, fill_value=0)
    data[f'past_{label}'] = data.groupby('container_id')[f'past_{label}'].cumsum()
    
    return data

def ovr_model_series(train, test, select_cols):
    """
    Train a series of one-vs-rest models as described in the notebook. Filter out input
    data from measurements that have already received a positive prediction from a 
    previous model.
    
    Parameters
    ----------
    train : DataFrame
        Train data
    test : DataFrame
        Test data
    select_cols : list
        Selected columns to scale and pass to the model
        
    Returns
    -------
    List of DataFrames and dicts
        Returns train data with added columns, test data with added columns, a dict
        of the scalers used for each model, and a dict of the trained models.
    """
    
    # List of labels to iterate over
    label_list = ['UNLOAD_SHIP', 'LOAD_TRAIN', 'DEPART_TRAIN']
    
    # Instantiate zero-columns
    train, test = create_filter_cols(train, test)
    
    # Store fit scalers in dictionary
    scl_dict = {}
    
    # Store trained models in dictionary
    model_dict = {}
    
    # Train a one-vs-rest model for each label
    for ind, item in enumerate(label_list):
        # Filter data if a past model has predicted the previous label in the list
        # for that container. If no previous model exists, return all rows.
        if ind > 0:
            train_prev_condition = train[f'past_{label_list[ind - 1]}'] == 1
            test_prev_condition = test[f'past_{label_list[ind - 1]}'] == 1
        else:
            train_prev_condition = [True]*len(train)
            test_prev_condition = [True]*len(test)
        
        X_train = train.loc[(train[f'past_{item}'] == 0) & (train_prev_condition)]
        X_test = test.loc[(test[f'past_{item}'] == 0) & (test_prev_condition)]
        
        # Create one-vs-rest labels
        y_train = np.where(X_train.LABEL == item,
                            1,
                            0)
        y_test = np.where(X_test.LABEL == item,
                            1,
                            0)
        
        X_train, X_test, scl = scale_data(X_train, X_test, select_cols=select_cols)
        
        scl_dict[item] = scl
        
        # The one-vs-rest model for Load Train does not require a more complex model
        if ind == 0:
            max_depth = 1
        else:
            max_depth = 12
        
        clf = xgb.XGBClassifier(max_depth=max_depth,
                                n_estimators=1000,
                                num_classes=2)
        
        clf.fit(X_train[select_cols], y_train)
        
        train_pred = clf.predict(X_train[select_cols])
        test_pred = clf.predict(X_test[select_cols])
        
        prev_label = None if ind == 0 else label_list[ind - 1]
        train = update_filter_cols(train, train_pred, item, prev_label)
        test = update_filter_cols(test, test_pred, item, prev_label)
        
        print(f'OVR Model performance for {item}\n')
        print('----------------')
        model_performance(y_test, test_pred, ['Rest', item])
    
        model_dict[item] = clf
    
    train['ovr_pred'] = np.where(train.pred_UNLOAD_SHIP == 1,
                                       0,
                                       np.where(train.pred_LOAD_TRAIN == 1,
                                                1,
                                                np.where(train.pred_DEPART_TRAIN == 1,
                                                         3,
                                                         2)))
    test['ovr_pred'] = np.where(test.pred_UNLOAD_SHIP == 1,
                                       0,
                                       np.where(test.pred_LOAD_TRAIN == 1,
                                                1,
                                                np.where(test.pred_DEPART_TRAIN == 1,
                                                         3,
                                                         2)))
    
    print('Total OVR Model performance')
    print('----------------')
    model_performance(test.y, test.ovr_pred)
    
    return train, test, scl_dict, model_dict

def main(single_model = True):
    """
    Main function to load and process data, and fit the model.
    
    Parameters
    ----------
    single_model : bool
        If True, fit a single multiclass classification model. If False, fit a series
        of one-vs-rest models.
        
    Returns
    -------
    None
    """
    
    train, test, scl, select_cols = load_and_process_data()
    
    if single_model:
        params = {'max_depth': 7,
                  'n_estimators': 1000,
                  'num_classes': train.y.max() + 1}
        print('Training model')
        print('---------------')
        clf, train_pred, train_probs, test_pred, test_probs = fit_predict(train, test, select_cols, params)
    else:
        train, test, scl_dict, model_dict = ovr_model_series(train, test, select_cols)
        
if __name__ == '__main__':
    main()