from gquant.dataframe_flow import Node
import cudf
import pandas as pd


class CsvStockLoader(Node):

    def columns_setup(self):
        self.required = {}
        self.addition = {"datetime": "datetime64[ms]",
                         "asset": "int64",
                         "volume": "float64",
                         "close": "float64",
                         "open": "float64",
                         "high": "float64",
                         "low": "float64"}
        self.deletion = None
        self.retention = None

    def process(self, inputs):
        """
        Load the end of day stock CSV data into cuDF dataframe

        Arguments
        -------
         inputs: list
             empty list
        Returns
        -------
        cudf.DataFrame
        """

        df = pd.read_csv(self.conf['path'],
                         converters={'DTE': lambda x: pd.Timestamp(str(x))})
        df = df[['DTE', 'OPEN',
                 'CLOSE', 'HIGH',
                 'LOW', 'SM_ID', 'VOLUME']]
        df['VOLUME'] /= 1000
        output = cudf.from_pandas(df)
        # change the names
        output.columns = ['datetime', 'open', 'close', 'high',
                          'low', "asset", 'volume']
        return output


if __name__ == "__main__":
    conf = {"path": "/home/yi/Projects/stocks/stock_price_hist.csv.gz"}
    loader = CsvStockLoader("node_csvdata", conf, False, True)
    df = loader([])
