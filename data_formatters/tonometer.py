import data_formatters.base
import libs.utils as utils
import pandas as pd
import sklearn.preprocessing

GenericDataFormatter = data_formatters.base.GenericDataFormatter
DataTypes = data_formatters.base.DataTypes
InputTypes = data_formatters.base.InputTypes

class TonometerFormatter(GenericDataFormatter):
    """Defines and formats data for the tonometer dataset
    """
    _column_definition = [
        ('time_from_start', DataTypes.REAL_VALUED, InputTypes.TIME),
        ('time', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('id', DataTypes.REAL_VALUED, InputTypes.ID),
        ('tono', DataTypes.REAL_VALUED, InputTypes.TARGET),
        ('age', DataTypes.REAL_VALUED, InputTypes.STATIC_INPUT),
        ('bmi', DataTypes.REAL_VALUED, InputTypes.STATIC_INPUT),
        ('gender', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
        ('m', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT)
    ]
    
    def __init__(self) -> None:
        """Initialize formatter"""
        self.identifiers = None
        self._real_scalers = None
        self._cat_scalers = None
        self._target_scalers = None
        self._num_classes_per_cat_input = None
        self._time_steps = self.get_fixed_params()['total_time_steps']

    def split_data(self, df, valid_boundary=350, test_boundary=400):
        """Splits data frame into training-validation-test data frames.

        This also calibrates scaling object, and transforms data for each split.

        Args:
        df: Source data frame to split.
        valid_boundary: Starting year for validation data
        test_boundary: Starting year for test data

        Returns:
        Tuple of transformed (train, valid, test) data.
        """

        print('Formatting train-valid-test splits.')

        index = df['time_from_start']
        train = df.loc[index < valid_boundary]
        valid = df.loc[(index >= valid_boundary) & (index < test_boundary)]
        test = df.loc[index >= test_boundary]

        self.set_scalers(train)

        return (self.transform_inputs(data) for data in [train, valid, test])
    
    def set_scalers(self, df):
        """Calibrates scalers using the data supplied.

        Args:
        df: Data to use to calibrate scalers.
        """
        print('Setting scalers with training data...')

        column_definitions = self.get_column_definition()
        id_column = utils.get_single_col_by_input_type(InputTypes.ID,
                                                       column_definitions)
        target_column = utils.get_single_col_by_input_type(InputTypes.TARGET,column_definitions)

        real_inputs = utils.extract_cols_from_data_type(
            DataTypes.REAL_VALUED, column_definitions,
            {InputTypes.ID, InputTypes.TIME}
        )

        self._real_scalers = {}
        self._target_scalers = {}
        identifiers = []
        
        for identifier, sliced in df.groupby(id_column):
            if len(sliced) >= self._time_steps:
                data = sliced[real_inputs].values
                targets = sliced[[target_column]].values
                self._real_scalers[identifier] \
                = sklearn.preprocessing.StandardScaler().fit(data)

                self._target_scalers[identifier] \
                = sklearn.preprocessing.StandardScaler().fit(targets)

                identifiers.append(identifier)

        categorial_inputs = utils.extract_cols_from_data_type(
            DataTypes.CATEGORICAL, column_definitions,
            {InputTypes.ID, InputTypes.TIME}
            )
        
        categorial_scalers = {}
        num_classes = []
        for col in categorial_inputs:
            srs = df[col].apply(str)
            categorial_scalers[col] = sklearn.preprocessing.LabelEncoder().fit(srs.values)
            num_classes.append(srs.nunique())

        self._cat_scalers = categorial_scalers
        self._num_classes_per_cat_input = num_classes

        self.identifiers = identifiers

    def transform_inputs(self, df):

        if self._real_scalers is None and self._cat_scalers is None:
            raise ValueError('Scalers have not been set')
        
        column_definitions = self.get_column_definition()
        id_col = utils.get_single_col_by_input_type(InputTypes.ID,
                                                    column_definitions)
        
        real_inputs = utils.extract_cols_from_data_type(
            DataTypes.REAL_VALUED, column_definitions,
            {InputTypes.TIME, InputTypes.ID}
        )

        categorical_inputs = utils.extract_cols_from_data_type(
            DataTypes.CATEGORICAL, column_definitions,
            {InputTypes.ID, InputTypes.TIME}
        )

        df_list = []

        for identifier, sliced in df.groupby(id_col):

            if len(sliced) >= self._time_steps:
                sliced_copy = sliced.copy()
                sliced_copy[real_inputs] = self._real_scalers[identifier].transform(sliced_copy[real_inputs].values)

                df_list.append(sliced_copy)

        output = pd.concat(df_list, axis=0)

        for col in categorical_inputs:
            string_df = df[col].apply(str)
            output[col] = self._cat_scalers[col].transform(string_df)
        
        return output
    
    def format_predictions(self, predictions):

        if self._target_scalers is None:
            raise ValueError('Scalers have not been set')
        
        column_names = predictions.columns

        df_list = []
        for identifier, sliced in predictions.groupby('identifier'):
            sliced_copy = sliced.copy()
            target_scaler = self._target_scalers[identifier]

            for col in column_names:
                if col not in {'forecast_time', 'identifier'}:
                    sliced_copy[col] = target_scaler.inverse_transform([sliced_copy[col]])[0]
            df_list.append(sliced_copy)
        output = pd.concat(df_list, axis=0)
        return output
    
    def get_fixed_params(self):
        fixed_params = {
            'total_time_steps': 8 * 50,
            'num_encoder_steps': 7 * 50,
            'num_epochs': 100,
            'early_stopping_patience': 5,
            'multiprocessing_workers': 5
        }
        return fixed_params

    def get_default_model_params(self):
        model_params = {
        'dropout_rate': 0.1,
        'hidden_layer_size': 160,
        'learning_rate': 0.001,
        'minibatch_size': 256,
        'max_gradient_norm': 0.01,
        'num_heads': 4,
        'stack_size': 1
        }
        return model_params
    
    def get_num_samples_for_calibration(self):
        return 160000, 11010
