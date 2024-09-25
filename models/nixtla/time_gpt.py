import pandas as pd
from nixtla import TimeGPT

class TimeSeriesForecaster:
    def __init__(self, model=None):
        self.model = model if model else TimeGPT()

    def fit(self, df, target_column, time_column):
        self.df = df
        self.target_column = target_column
        self.time_column = time_column
        self.model.fit(df[[time_column, target_column]])

    def predict(self, periods):
        future_dates = pd.date_range(start=self.df[self.time_column].max(), periods=periods + 1, closed='right')
        future_df = pd.DataFrame({self.time_column: future_dates})
        forecast = self.model.predict(future_df)
        return forecast

# Example usage:
# df = pd.read_csv('your_time_series_data.csv')
# forecaster = TimeSeriesForecaster()
# forecaster.fit(df, target_column='your_target_column', time_column='your_time_column')
# forecast = forecaster.predict(periods=10)
# print(forecast)