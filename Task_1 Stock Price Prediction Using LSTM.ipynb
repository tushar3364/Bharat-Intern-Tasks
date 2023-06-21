{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "c025a943",
   "metadata": {},
   "source": [
    "# DATA SCIENCE INTERN @BHARAT INTERN"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9d59812b",
   "metadata": {},
   "source": [
    "### AUTHOR : TUSHAR KUMAR"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "00b402f2",
   "metadata": {},
   "source": [
    "# TASK 1 : STOCK PREDICTION"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f8e051f0",
   "metadata": {},
   "source": [
    "## PURPOSE : TO PREDICT THE STOCK PRICE OF A COMPANY USING LSTM."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6d67fee9",
   "metadata": {},
   "source": [
    "## ABOUT DATASET\n",
    "### Google Stock Prediction\n",
    "\n",
    "This dataset contains historical data of **Google's stock prices** and related attributes. It consists of 14 columns and a smaller subset of 1257 rows. Each column represents a specific attribute, and each row contains the corresponding values for that attribute.\n",
    "\n",
    "The columns in the dataset are as follows:\n",
    "\n",
    "1. **Symbol**: The name of the company, which is **GOOG** in this case.\n",
    "2. **Date**: The year and date of the stock data.\n",
    "3. **Close**: The closing price of Google's stock on a particular day.\n",
    "4. **High**: The highest value reached by Google's stock on the given day.\n",
    "5. **Low**: The lowest value reached by Google's stock on the given day.\n",
    "6. **Open**: The opening value of Google's stock on the given day.\n",
    "7. **Volume**: The trading volume of Google's stock on the given day, i.e., the number of shares traded.\n",
    "8. **adjClose**: The adjusted closing price of Google's stock, considering factors such as dividends and stock splits.\n",
    "9. **adjHigh**: The adjusted highest value reached by Google's stock on the given day.\n",
    "10. **adjLow**: The adjusted lowest value reached by Google's stock on the given day.\n",
    "11. **adjOpen**: The adjusted opening value of Google's stock on the given day.\n",
    "12. **adjVolume**: The adjusted trading volume of Google's stock on the given day, accounting for factors such as stock splits.\n",
    "13. **divCash**: The amount of cash dividend paid out to shareholders on the given day.\n",
    "14. **splitFactor**: The split factor, if any, applied to Google's stock on the given day. A split factor of 1 indicates no split.\n",
    "\n",
    "The dataset is available at Kaggle : https://www.kaggle.com/datasets/shreenidhihipparagi/google-stock-prediction"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5ec3be7e",
   "metadata": {},
   "source": [
    "## STEPS INVOLVED : \n",
    "### 1 . IMPORTING LIBRARIES AND DATA TO BE USED\n",
    "### 2. GATHERING INSIGHTS\n",
    "### 3. DATA PRE-PROCESSING\n",
    "### 4. CREATING LSTM MODEL\n",
    "### 5. VISUALIZING ACTUAL VS PREDICTED DATA\n",
    "### 6. PREDICTING UPCOMING 15 DAYS"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6901025f",
   "metadata": {},
   "source": []
  },
  {
   "cell_type": "markdown",
   "id": "67f14f64",
   "metadata": {},
   "source": [
    "## STEP 1 : IMPORTING LIBRARIES AND DATA TO BE USED"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "f6fe173f",
   "metadata": {},
   "outputs": [],
   "source": [
    "#importing libraries to be used\n",
    "import numpy as np # for linear algebra\n",
    "import pandas as pd # data preprocessing\n",
    "import matplotlib.pyplot as plt # data visualization library\n",
    "import seaborn as sns # data visualization library\n",
    "%matplotlib inline\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore') # ignore warnings \n",
    "\n",
    "from sklearn.preprocessing import MinMaxScaler # for normalization\n",
    "from keras.models import Sequential\n",
    "from keras.layers import Dense, Dropout, LSTM, Bidirectional"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "0310c3d2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>symbol</th>\n",
       "      <th>date</th>\n",
       "      <th>close</th>\n",
       "      <th>high</th>\n",
       "      <th>low</th>\n",
       "      <th>open</th>\n",
       "      <th>volume</th>\n",
       "      <th>adjClose</th>\n",
       "      <th>adjHigh</th>\n",
       "      <th>adjLow</th>\n",
       "      <th>adjOpen</th>\n",
       "      <th>adjVolume</th>\n",
       "      <th>divCash</th>\n",
       "      <th>splitFactor</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>GOOG</td>\n",
       "      <td>2016-06-14 00:00:00+00:00</td>\n",
       "      <td>718.27</td>\n",
       "      <td>722.47</td>\n",
       "      <td>713.1200</td>\n",
       "      <td>716.48</td>\n",
       "      <td>1306065</td>\n",
       "      <td>718.27</td>\n",
       "      <td>722.47</td>\n",
       "      <td>713.1200</td>\n",
       "      <td>716.48</td>\n",
       "      <td>1306065</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>GOOG</td>\n",
       "      <td>2016-06-15 00:00:00+00:00</td>\n",
       "      <td>718.92</td>\n",
       "      <td>722.98</td>\n",
       "      <td>717.3100</td>\n",
       "      <td>719.00</td>\n",
       "      <td>1214517</td>\n",
       "      <td>718.92</td>\n",
       "      <td>722.98</td>\n",
       "      <td>717.3100</td>\n",
       "      <td>719.00</td>\n",
       "      <td>1214517</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>GOOG</td>\n",
       "      <td>2016-06-16 00:00:00+00:00</td>\n",
       "      <td>710.36</td>\n",
       "      <td>716.65</td>\n",
       "      <td>703.2600</td>\n",
       "      <td>714.91</td>\n",
       "      <td>1982471</td>\n",
       "      <td>710.36</td>\n",
       "      <td>716.65</td>\n",
       "      <td>703.2600</td>\n",
       "      <td>714.91</td>\n",
       "      <td>1982471</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>GOOG</td>\n",
       "      <td>2016-06-17 00:00:00+00:00</td>\n",
       "      <td>691.72</td>\n",
       "      <td>708.82</td>\n",
       "      <td>688.4515</td>\n",
       "      <td>708.65</td>\n",
       "      <td>3402357</td>\n",
       "      <td>691.72</td>\n",
       "      <td>708.82</td>\n",
       "      <td>688.4515</td>\n",
       "      <td>708.65</td>\n",
       "      <td>3402357</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>GOOG</td>\n",
       "      <td>2016-06-20 00:00:00+00:00</td>\n",
       "      <td>693.71</td>\n",
       "      <td>702.48</td>\n",
       "      <td>693.4100</td>\n",
       "      <td>698.77</td>\n",
       "      <td>2082538</td>\n",
       "      <td>693.71</td>\n",
       "      <td>702.48</td>\n",
       "      <td>693.4100</td>\n",
       "      <td>698.77</td>\n",
       "      <td>2082538</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>GOOG</td>\n",
       "      <td>2016-06-21 00:00:00+00:00</td>\n",
       "      <td>695.94</td>\n",
       "      <td>702.77</td>\n",
       "      <td>692.0100</td>\n",
       "      <td>698.40</td>\n",
       "      <td>1465634</td>\n",
       "      <td>695.94</td>\n",
       "      <td>702.77</td>\n",
       "      <td>692.0100</td>\n",
       "      <td>698.40</td>\n",
       "      <td>1465634</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>GOOG</td>\n",
       "      <td>2016-06-22 00:00:00+00:00</td>\n",
       "      <td>697.46</td>\n",
       "      <td>700.86</td>\n",
       "      <td>693.0819</td>\n",
       "      <td>699.06</td>\n",
       "      <td>1184318</td>\n",
       "      <td>697.46</td>\n",
       "      <td>700.86</td>\n",
       "      <td>693.0819</td>\n",
       "      <td>699.06</td>\n",
       "      <td>1184318</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>GOOG</td>\n",
       "      <td>2016-06-23 00:00:00+00:00</td>\n",
       "      <td>701.87</td>\n",
       "      <td>701.95</td>\n",
       "      <td>687.0000</td>\n",
       "      <td>697.45</td>\n",
       "      <td>2171415</td>\n",
       "      <td>701.87</td>\n",
       "      <td>701.95</td>\n",
       "      <td>687.0000</td>\n",
       "      <td>697.45</td>\n",
       "      <td>2171415</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>GOOG</td>\n",
       "      <td>2016-06-24 00:00:00+00:00</td>\n",
       "      <td>675.22</td>\n",
       "      <td>689.40</td>\n",
       "      <td>673.4500</td>\n",
       "      <td>675.17</td>\n",
       "      <td>4449022</td>\n",
       "      <td>675.22</td>\n",
       "      <td>689.40</td>\n",
       "      <td>673.4500</td>\n",
       "      <td>675.17</td>\n",
       "      <td>4449022</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>GOOG</td>\n",
       "      <td>2016-06-27 00:00:00+00:00</td>\n",
       "      <td>668.26</td>\n",
       "      <td>672.30</td>\n",
       "      <td>663.2840</td>\n",
       "      <td>671.00</td>\n",
       "      <td>2641085</td>\n",
       "      <td>668.26</td>\n",
       "      <td>672.30</td>\n",
       "      <td>663.2840</td>\n",
       "      <td>671.00</td>\n",
       "      <td>2641085</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  symbol                       date   close    high       low    open  \\\n",
       "0   GOOG  2016-06-14 00:00:00+00:00  718.27  722.47  713.1200  716.48   \n",
       "1   GOOG  2016-06-15 00:00:00+00:00  718.92  722.98  717.3100  719.00   \n",
       "2   GOOG  2016-06-16 00:00:00+00:00  710.36  716.65  703.2600  714.91   \n",
       "3   GOOG  2016-06-17 00:00:00+00:00  691.72  708.82  688.4515  708.65   \n",
       "4   GOOG  2016-06-20 00:00:00+00:00  693.71  702.48  693.4100  698.77   \n",
       "5   GOOG  2016-06-21 00:00:00+00:00  695.94  702.77  692.0100  698.40   \n",
       "6   GOOG  2016-06-22 00:00:00+00:00  697.46  700.86  693.0819  699.06   \n",
       "7   GOOG  2016-06-23 00:00:00+00:00  701.87  701.95  687.0000  697.45   \n",
       "8   GOOG  2016-06-24 00:00:00+00:00  675.22  689.40  673.4500  675.17   \n",
       "9   GOOG  2016-06-27 00:00:00+00:00  668.26  672.30  663.2840  671.00   \n",
       "\n",
       "    volume  adjClose  adjHigh    adjLow  adjOpen  adjVolume  divCash  \\\n",
       "0  1306065    718.27   722.47  713.1200   716.48    1306065      0.0   \n",
       "1  1214517    718.92   722.98  717.3100   719.00    1214517      0.0   \n",
       "2  1982471    710.36   716.65  703.2600   714.91    1982471      0.0   \n",
       "3  3402357    691.72   708.82  688.4515   708.65    3402357      0.0   \n",
       "4  2082538    693.71   702.48  693.4100   698.77    2082538      0.0   \n",
       "5  1465634    695.94   702.77  692.0100   698.40    1465634      0.0   \n",
       "6  1184318    697.46   700.86  693.0819   699.06    1184318      0.0   \n",
       "7  2171415    701.87   701.95  687.0000   697.45    2171415      0.0   \n",
       "8  4449022    675.22   689.40  673.4500   675.17    4449022      0.0   \n",
       "9  2641085    668.26   672.30  663.2840   671.00    2641085      0.0   \n",
       "\n",
       "   splitFactor  \n",
       "0          1.0  \n",
       "1          1.0  \n",
       "2          1.0  \n",
       "3          1.0  \n",
       "4          1.0  \n",
       "5          1.0  \n",
       "6          1.0  \n",
       "7          1.0  \n",
       "8          1.0  \n",
       "9          1.0  "
      ]
     },
     "execution_count": 43,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = pd.read_csv('Google Stocks.csv') # data_importing\n",
    "df.head(10) # fetching first 10 rows of dataset"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e9f948f7",
   "metadata": {},
   "source": [
    "## STEP 2 : GATHERING INSIGHTS"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "c56f5d52",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Shape of data: (1258, 14)\n"
     ]
    }
   ],
   "source": [
    "# shape of data\n",
    "print(\"Shape of data:\",df.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "dec59878",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>close</th>\n",
       "      <th>high</th>\n",
       "      <th>low</th>\n",
       "      <th>open</th>\n",
       "      <th>volume</th>\n",
       "      <th>adjClose</th>\n",
       "      <th>adjHigh</th>\n",
       "      <th>adjLow</th>\n",
       "      <th>adjOpen</th>\n",
       "      <th>adjVolume</th>\n",
       "      <th>divCash</th>\n",
       "      <th>splitFactor</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>1258.000000</td>\n",
       "      <td>1258.000000</td>\n",
       "      <td>1258.000000</td>\n",
       "      <td>1258.000000</td>\n",
       "      <td>1.258000e+03</td>\n",
       "      <td>1258.000000</td>\n",
       "      <td>1258.000000</td>\n",
       "      <td>1258.000000</td>\n",
       "      <td>1258.000000</td>\n",
       "      <td>1.258000e+03</td>\n",
       "      <td>1258.0</td>\n",
       "      <td>1258.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>1216.317067</td>\n",
       "      <td>1227.430934</td>\n",
       "      <td>1204.176430</td>\n",
       "      <td>1215.260779</td>\n",
       "      <td>1.601590e+06</td>\n",
       "      <td>1216.317067</td>\n",
       "      <td>1227.430936</td>\n",
       "      <td>1204.176436</td>\n",
       "      <td>1215.260779</td>\n",
       "      <td>1.601590e+06</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>383.333358</td>\n",
       "      <td>387.570872</td>\n",
       "      <td>378.777094</td>\n",
       "      <td>382.446995</td>\n",
       "      <td>6.960172e+05</td>\n",
       "      <td>383.333358</td>\n",
       "      <td>387.570873</td>\n",
       "      <td>378.777099</td>\n",
       "      <td>382.446995</td>\n",
       "      <td>6.960172e+05</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>668.260000</td>\n",
       "      <td>672.300000</td>\n",
       "      <td>663.284000</td>\n",
       "      <td>671.000000</td>\n",
       "      <td>3.467530e+05</td>\n",
       "      <td>668.260000</td>\n",
       "      <td>672.300000</td>\n",
       "      <td>663.284000</td>\n",
       "      <td>671.000000</td>\n",
       "      <td>3.467530e+05</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>960.802500</td>\n",
       "      <td>968.757500</td>\n",
       "      <td>952.182500</td>\n",
       "      <td>959.005000</td>\n",
       "      <td>1.173522e+06</td>\n",
       "      <td>960.802500</td>\n",
       "      <td>968.757500</td>\n",
       "      <td>952.182500</td>\n",
       "      <td>959.005000</td>\n",
       "      <td>1.173522e+06</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>1132.460000</td>\n",
       "      <td>1143.935000</td>\n",
       "      <td>1117.915000</td>\n",
       "      <td>1131.150000</td>\n",
       "      <td>1.412588e+06</td>\n",
       "      <td>1132.460000</td>\n",
       "      <td>1143.935000</td>\n",
       "      <td>1117.915000</td>\n",
       "      <td>1131.150000</td>\n",
       "      <td>1.412588e+06</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>1360.595000</td>\n",
       "      <td>1374.345000</td>\n",
       "      <td>1348.557500</td>\n",
       "      <td>1361.075000</td>\n",
       "      <td>1.812156e+06</td>\n",
       "      <td>1360.595000</td>\n",
       "      <td>1374.345000</td>\n",
       "      <td>1348.557500</td>\n",
       "      <td>1361.075000</td>\n",
       "      <td>1.812156e+06</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>2521.600000</td>\n",
       "      <td>2526.990000</td>\n",
       "      <td>2498.290000</td>\n",
       "      <td>2524.920000</td>\n",
       "      <td>6.207027e+06</td>\n",
       "      <td>2521.600000</td>\n",
       "      <td>2526.990000</td>\n",
       "      <td>2498.290000</td>\n",
       "      <td>2524.920000</td>\n",
       "      <td>6.207027e+06</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "             close         high          low         open        volume  \\\n",
       "count  1258.000000  1258.000000  1258.000000  1258.000000  1.258000e+03   \n",
       "mean   1216.317067  1227.430934  1204.176430  1215.260779  1.601590e+06   \n",
       "std     383.333358   387.570872   378.777094   382.446995  6.960172e+05   \n",
       "min     668.260000   672.300000   663.284000   671.000000  3.467530e+05   \n",
       "25%     960.802500   968.757500   952.182500   959.005000  1.173522e+06   \n",
       "50%    1132.460000  1143.935000  1117.915000  1131.150000  1.412588e+06   \n",
       "75%    1360.595000  1374.345000  1348.557500  1361.075000  1.812156e+06   \n",
       "max    2521.600000  2526.990000  2498.290000  2524.920000  6.207027e+06   \n",
       "\n",
       "          adjClose      adjHigh       adjLow      adjOpen     adjVolume  \\\n",
       "count  1258.000000  1258.000000  1258.000000  1258.000000  1.258000e+03   \n",
       "mean   1216.317067  1227.430936  1204.176436  1215.260779  1.601590e+06   \n",
       "std     383.333358   387.570873   378.777099   382.446995  6.960172e+05   \n",
       "min     668.260000   672.300000   663.284000   671.000000  3.467530e+05   \n",
       "25%     960.802500   968.757500   952.182500   959.005000  1.173522e+06   \n",
       "50%    1132.460000  1143.935000  1117.915000  1131.150000  1.412588e+06   \n",
       "75%    1360.595000  1374.345000  1348.557500  1361.075000  1.812156e+06   \n",
       "max    2521.600000  2526.990000  2498.290000  2524.920000  6.207027e+06   \n",
       "\n",
       "       divCash  splitFactor  \n",
       "count   1258.0       1258.0  \n",
       "mean       0.0          1.0  \n",
       "std        0.0          0.0  \n",
       "min        0.0          1.0  \n",
       "25%        0.0          1.0  \n",
       "50%        0.0          1.0  \n",
       "75%        0.0          1.0  \n",
       "max        0.0          1.0  "
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# statistical description of data\n",
    "df.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "1f78fdf3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 1258 entries, 0 to 1257\n",
      "Data columns (total 14 columns):\n",
      " #   Column       Non-Null Count  Dtype  \n",
      "---  ------       --------------  -----  \n",
      " 0   symbol       1258 non-null   object \n",
      " 1   date         1258 non-null   object \n",
      " 2   close        1258 non-null   float64\n",
      " 3   high         1258 non-null   float64\n",
      " 4   low          1258 non-null   float64\n",
      " 5   open         1258 non-null   float64\n",
      " 6   volume       1258 non-null   int64  \n",
      " 7   adjClose     1258 non-null   float64\n",
      " 8   adjHigh      1258 non-null   float64\n",
      " 9   adjLow       1258 non-null   float64\n",
      " 10  adjOpen      1258 non-null   float64\n",
      " 11  adjVolume    1258 non-null   int64  \n",
      " 12  divCash      1258 non-null   float64\n",
      " 13  splitFactor  1258 non-null   float64\n",
      "dtypes: float64(10), int64(2), object(2)\n",
      "memory usage: 137.7+ KB\n"
     ]
    }
   ],
   "source": [
    "# summary of data\n",
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "4e9bcc70",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "symbol         0\n",
       "date           0\n",
       "close          0\n",
       "high           0\n",
       "low            0\n",
       "open           0\n",
       "volume         0\n",
       "adjClose       0\n",
       "adjHigh        0\n",
       "adjLow         0\n",
       "adjOpen        0\n",
       "adjVolume      0\n",
       "divCash        0\n",
       "splitFactor    0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 47,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# checking null values\n",
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1db53322",
   "metadata": {},
   "source": [
    "### There are no null values in the dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "e45349ad",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>open</th>\n",
       "      <th>close</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>date</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2016-06-14</th>\n",
       "      <td>716.48</td>\n",
       "      <td>718.27</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2016-06-15</th>\n",
       "      <td>719.00</td>\n",
       "      <td>718.92</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2016-06-16</th>\n",
       "      <td>714.91</td>\n",
       "      <td>710.36</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2016-06-17</th>\n",
       "      <td>708.65</td>\n",
       "      <td>691.72</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2016-06-20</th>\n",
       "      <td>698.77</td>\n",
       "      <td>693.71</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2016-06-21</th>\n",
       "      <td>698.40</td>\n",
       "      <td>695.94</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2016-06-22</th>\n",
       "      <td>699.06</td>\n",
       "      <td>697.46</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2016-06-23</th>\n",
       "      <td>697.45</td>\n",
       "      <td>701.87</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2016-06-24</th>\n",
       "      <td>675.17</td>\n",
       "      <td>675.22</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2016-06-27</th>\n",
       "      <td>671.00</td>\n",
       "      <td>668.26</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "              open   close\n",
       "date                      \n",
       "2016-06-14  716.48  718.27\n",
       "2016-06-15  719.00  718.92\n",
       "2016-06-16  714.91  710.36\n",
       "2016-06-17  708.65  691.72\n",
       "2016-06-20  698.77  693.71\n",
       "2016-06-21  698.40  695.94\n",
       "2016-06-22  699.06  697.46\n",
       "2016-06-23  697.45  701.87\n",
       "2016-06-24  675.17  675.22\n",
       "2016-06-27  671.00  668.26"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = df[['date','open','close']] # Extracting required columns\n",
    "df['date'] = pd.to_datetime(df['date'].apply(lambda x: x.split()[0])) # converting object dtype of date column to datetime dtype\n",
    "df.set_index('date',drop=True,inplace=True) # Setting date column as index\n",
    "df.head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "9b83eebb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAABJ0AAAGuCAYAAAA6ZnfjAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8QVMy6AAAACXBIWXMAAAsTAAALEwEAmpwYAACmX0lEQVR4nOzddZhd1dXH8e8e95nIxBUiRICEBAjuFiRAgBIKBC9aoBQohbZQCqXwosWKu7tDCBA0QALxELdJJjYZdznvH3vOlZk7fmXk93meec45+8jdd0KbnXXWXts4joOIiIiIiIiIiEgwRUW6AyIiIiIiIiIi0vko6CQiIiIiIiIiIkGnoJOIiIiIiIiIiASdgk4iIiIiIiIiIhJ0CjqJiIiIiIiIiEjQxUS6A+HUs2dPZ8iQIZHuhoiIiITI3LlztzuOkxnpfoiXxl8iIiKdX0NjsC4VdBoyZAhz5syJdDdEREQkRIwx6yLdB/Gn8ZeIiEjn19AYTNPrREREREREREQk6BR0EhERERERERGRoFPQSUREREREREREgq5L1XQKpLKykqysLMrKyiLdlbBJSEhgwIABxMbGRrorIiIi0gV1xfGXS+MwERHpSrp80CkrK4vU1FSGDBmCMSbS3Qk5x3HIyckhKyuLoUOHRro7IiIi0gV1tfGXS+MwERHparr89LqysjJ69OjRZQY8xhh69OjRJd8sioiISPvQ1cZfLo3DRESkq+nyQSegSw54RERERCKpq45Huur3FhGRrklBJxERERERERERCToFndqBrKwspkyZwvDhw9l555258sorqaioiHS3RERERDq1zZs3c/rpp7PzzjszevRoJk+ezPLlyxk7dmykuyYiItIpKOgUYY7jcPLJJ3PiiSeyYsUKli9fTlFRETfeeGOkuyYiIiLSaTmOw0knncTBBx/MqlWrWLJkCbfffjtbtmyJdNdEREQ6DQWdIuyLL74gISGBc889F4Do6GjuvfdennrqKR5++GGmTJnC0UcfzciRI7nllls8973wwgvstddejBs3jj/84Q9UV1cDkJKSwo033sjuu+/OpEmTNHASERERCeDLL78kNjaWiy++2NM2btw4Bg4c6DkuKyvj3HPPZdddd2X8+PF8+eWXACxevNgzDtttt91YsWIF0PD4TEREpKuKiXQH2pOrPrmKeZvnBfWZ4/qM476j72vw/OLFi5kwYYJfW1paGoMGDaKqqoqffvqJRYsWkZSUxJ577smxxx5LcnIyr776Kt999x2xsbFceumlvPjii5x99tkUFxczadIkbrvtNq677joef/xxbrrppqB+JxEREZGgueoqmDcvuM8cNw7uu6/RSxYtWlRvDFbXQw89BMDChQv57bffOPLII1m+fDmPPvooV155Jb///e+pqKigurqapUuXNjg+ExER6aoUdIowx3ECrmLith9xxBH06NEDgJNPPplvv/2WmJgY5s6dy5577glAaWkpvXr1AiAuLo7jjjsOgAkTJjBjxowwfRMRERGRzuXbb7/liiuuAGCXXXZh8ODBLF++nH322YfbbruNrKwsTj75ZIYPH87MmTMbHJ+JiIh0VQo6+WgsIylUxowZw5tvvunXVlBQwIYNG4iOjq4XkDLG4DgO06dP59///ne958XGxnruiY6OpqqqKnSdFxEREWmrJjKSQmXMmDG88cYbjV7jOE7A9jPOOIO9996bDz/8kKOOOoonnnii0fGZiIhIV6WaThF22GGHUVJSwnPPPQdAdXU111xzDeeccw5JSUnMmDGDHTt2UFpayjvvvMN+++3HYYcdxhtvvMHWrVsB2LFjB+vWrYvk1xAREWmW37b/RlZBVqS7IcKhhx5KeXk5jz/+uKft559/9htTHXjggbz44osALF++nPXr1zNy5EhWr17NTjvtxB//+EdOOOEEFixYoPGZiIi0X44DubkR+WgFnSLMGMPbb7/N66+/zvDhwxkxYgQJCQncfvvtAOy///6cddZZjBs3jqlTpzJx4kRGjx7Nv/71L4488kh22203jjjiCLKzsyP8TURERJp28qsnc/WnV0e6GyKeMdiMGTPYeeedGTNmDDfffDP9+vXzXHPppZdSXV3Nrrvuyu9+9zueeeYZ4uPjefXVVxk7dizjxo3jt99+4+yzz9b4TERE2q9nnoHu3WHp0rB/tGkobbgzmjhxojNnzhy/tqVLlzJq1KgI9ahxzzzzDHPmzOHBBx8M+rPb8/cWEZHOa/B9gzlkyCE8c+IzIXm+MWau4zgTQ/JwaZWONv4Kh67+/UVEJMzOPx+eegoefhguuSQkH9HQGCysmU7GmIHGmC+NMUuNMYuNMVfWtt9sjNlojJlX+zPZ554bjDErjTHLjDFH+bRPMMYsrD33gAlUjVtE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AHdNwP7ABp/bsmrb+tfu120XERGRBnhqOpmu+aZNrHCMwYwxFxlj5hhj5mzbti3o30FERKRDUaZTeBljUoA3gascxylo7NIAbU4j7YE+S4MeERERlOkk4RuDOY7zmOM4Ex3HmZiZmdm6zoqIiHQWynQKH2NMLHaw86LjOG/VNm+pTdemdru1tj0LGOhz+wBgU237gADt9WjQIyIiYlU7qunUlYV7DCYiIiK1amoUdAqH2tVNngSWOo5zj8+p94DptfvTgXd92k83xsQbY4Zii1X+VJv+XWiMmVT7zLN97hEREZEAPNProrpmendXpjGYiIhIBHXh6XUxYf68/YCzgIXGmHm1bX8F7gBeM8acD6wHTgVwHGexMeY1YAl21ZXLHKf2NS1cAjwDJAIf1/6IiIhIAzS9rkvTGExERCRSuvD0urAGnRzH+ZbAtQAADmvgntuA2wK0zwHGBq93IiIinVt1jabXdVUag4mIiERQF8500qhTRESki9DqdSIiIiIR0IUznbrmtxYREemCNL1OREREJAKU6SQiIiKdnVavExEREYkAZTqJiIhIZ6fV60REREQiQJlOIiIi0tlpep2IiIhIBCjTSURERDo7rV4nIiIiEgE1NQo6iYiISOem1etEREREIkDT60RERKSz0/Q6ERERkQiorOyyQaeYSHdAREREQmtH6Q4WbFmg1etEREREwik/H1avhoICGDAg0r2JCI06RUREOrmDnzmYQ549hLKqMgBiovTOSURERCTkDj0U9tjDBp/S0yPdm4hQ0ElERKSTW7h1IQD5ZfmAgk4iIiIiYfHLL3ablwcZGZHsScQo6CQiItJF5JXnERMVgzEm0l0RERER6dzy8737ubnKdBIREZHOp6iiyLOfV5ZHbFRsBHsjIiIi0kX85z/+xwo6iYiISGdz2UeXefbzyvKIjVbQSURERCTklizxP1bQSURERDozZTqJiIiIhIHjwM8/+7cp6CQiIiKdTa+kXp79tXlrlekkIiIiEmplZbBpEwwY4G1T0ElEREQ6m8qaShJiEjAYZTqJiIiIhENVld0mJnrbFHQSERGRzqayupLk2GSSYpMAlOkkIiIiEmpu0CkhwdumoJOIiIh0NpU1lcRGx5IYa9+0KdNJREREJMSqq+02Pt7b1kWDTjGR7oCIiIiETlVNFTFRMZ5gkzKdRERERELMzXSK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\n",
      "text/plain": [
       "<Figure size 1440x504 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# plotting open and closing price on date index\n",
    "fig, ax =plt.subplots(1,2,figsize=(20,7))\n",
    "ax[0].plot(df['open'],label='Open',color='green')\n",
    "ax[0].set_xlabel('Date',size=15)\n",
    "ax[0].set_ylabel('Price',size=15)\n",
    "ax[0].legend()\n",
    "\n",
    "ax[1].plot(df['close'],label='Close',color='red')\n",
    "ax[1].set_xlabel('Date',size=15)\n",
    "ax[1].set_ylabel('Price',size=15)\n",
    "ax[1].legend()\n",
    "\n",
    "fig.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f77919e0",
   "metadata": {},
   "source": [
    "## STEP 3 : DATA PRE-PROCESSING"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "39a1a201",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>open</th>\n",
       "      <th>close</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>date</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2016-06-14</th>\n",
       "      <td>0.024532</td>\n",
       "      <td>0.026984</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2016-06-15</th>\n",
       "      <td>0.025891</td>\n",
       "      <td>0.027334</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2016-06-16</th>\n",
       "      <td>0.023685</td>\n",
       "      <td>0.022716</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2016-06-17</th>\n",
       "      <td>0.020308</td>\n",
       "      <td>0.012658</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2016-06-20</th>\n",
       "      <td>0.014979</td>\n",
       "      <td>0.013732</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2016-06-21</th>\n",
       "      <td>0.014779</td>\n",
       "      <td>0.014935</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2016-06-22</th>\n",
       "      <td>0.015135</td>\n",
       "      <td>0.015755</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2016-06-23</th>\n",
       "      <td>0.014267</td>\n",
       "      <td>0.018135</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2016-06-24</th>\n",
       "      <td>0.002249</td>\n",
       "      <td>0.003755</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2016-06-27</th>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                open     close\n",
       "date                          \n",
       "2016-06-14  0.024532  0.026984\n",
       "2016-06-15  0.025891  0.027334\n",
       "2016-06-16  0.023685  0.022716\n",
       "2016-06-17  0.020308  0.012658\n",
       "2016-06-20  0.014979  0.013732\n",
       "2016-06-21  0.014779  0.014935\n",
       "2016-06-22  0.015135  0.015755\n",
       "2016-06-23  0.014267  0.018135\n",
       "2016-06-24  0.002249  0.003755\n",
       "2016-06-27  0.000000  0.000000"
      ]
     },
     "execution_count": 50,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# normalizing all the values of all columns using MinMaxScaler\n",
    "MMS = MinMaxScaler()\n",
    "df[df.columns] = MMS.fit_transform(df)\n",
    "df.head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "568d3072",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "944"
      ]
     },
     "execution_count": 51,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# splitting the data into training and test set\n",
    "training_size = round(len(df) * 0.75) # Selecting 75 % for training and 25 % for testing\n",
    "training_size"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "4c507274",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "((944, 2), (314, 2))"
      ]
     },
     "execution_count": 52,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_data = df[:training_size]\n",
    "test_data  = df[training_size:]\n",
    "\n",
    "train_data.shape, test_data.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "c3d05650",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Function to create sequence of data for training and testing\n",
    "\n",
    "def create_sequence(dataset):\n",
    "  sequences = []\n",
    "  labels = []\n",
    "\n",
    "  start_idx = 0\n",
    "\n",
    "  for stop_idx in range(50,len(dataset)): # Selecting 50 rows at a time\n",
    "    sequences.append(dataset.iloc[start_idx:stop_idx])\n",
    "    labels.append(dataset.iloc[stop_idx])\n",
    "    start_idx += 1\n",
    "  return (np.array(sequences),np.array(labels))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "6c41f27f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "((894, 50, 2), (894, 2), (264, 50, 2), (264, 2))"
      ]
     },
     "execution_count": 54,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_seq, train_label = create_sequence(train_data) \n",
    "test_seq, test_label = create_sequence(test_data)\n",
    "train_seq.shape, train_label.shape, test_seq.shape, test_label.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "32f37ed4",
   "metadata": {},
   "source": [
    "## STEP 4 :  CREATING LSTM MODEL"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "46aa3643",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"sequential_1\"\n",
      "_________________________________________________________________\n",
      " Layer (type)                Output Shape              Param #   \n",
      "=================================================================\n",
      " lstm_2 (LSTM)               (None, 50, 50)            10600     \n",
      "                                                                 \n",
      " dropout_1 (Dropout)         (None, 50, 50)            0         \n",
      "                                                                 \n",
      " lstm_3 (LSTM)               (None, 50)                20200     \n",
      "                                                                 \n",
      " dense_1 (Dense)             (None, 2)                 102       \n",
      "                                                                 \n",
      "=================================================================\n",
      "Total params: 30,902\n",
      "Trainable params: 30,902\n",
      "Non-trainable params: 0\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "# imported Sequential from keras.models \n",
    "model = Sequential()\n",
    "# importing Dense, Dropout, LSTM, Bidirectional from keras.layers \n",
    "model.add(LSTM(units=50, return_sequences=True, input_shape = (train_seq.shape[1], train_seq.shape[2])))\n",
    "\n",
    "model.add(Dropout(0.1)) \n",
    "model.add(LSTM(units=50))\n",
    "\n",
    "model.add(Dense(2))\n",
    "\n",
    "model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])\n",
    "\n",
    "model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "id": "2308278a",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/100\n",
      "28/28 [==============================] - 6s 76ms/step - loss: 0.0075 - mean_absolute_error: 0.0615 - val_loss: 0.0177 - val_mean_absolute_error: 0.1109\n",
      "Epoch 2/100\n",
      "28/28 [==============================] - 1s 36ms/step - loss: 7.4780e-04 - mean_absolute_error: 0.0213 - val_loss: 0.0036 - val_mean_absolute_error: 0.0456\n",
      "Epoch 3/100\n",
      "28/28 [==============================] - 1s 37ms/step - loss: 5.1931e-04 - mean_absolute_error: 0.0167 - val_loss: 0.0063 - val_mean_absolute_error: 0.0636\n",
      "Epoch 4/100\n",
      "28/28 [==============================] - 1s 42ms/step - loss: 4.8920e-04 - mean_absolute_error: 0.0166 - val_loss: 0.0051 - val_mean_absolute_error: 0.0564\n",
      "Epoch 5/100\n",
      "28/28 [==============================] - 1s 34ms/step - loss: 4.5922e-04 - mean_absolute_error: 0.0155 - val_loss: 0.0068 - val_mean_absolute_error: 0.0673\n",
      "Epoch 6/100\n",
      "28/28 [==============================] - 1s 32ms/step - loss: 4.8714e-04 - mean_absolute_error: 0.0163 - val_loss: 0.0041 - val_mean_absolute_error: 0.0488\n",
      "Epoch 7/100\n",
      "28/28 [==============================] - 1s 33ms/step - loss: 4.3301e-04 - mean_absolute_error: 0.0154 - val_loss: 0.0038 - val_mean_absolute_error: 0.0476\n",
      "Epoch 8/100\n",
      "28/28 [==============================] - 1s 34ms/step - loss: 4.2331e-04 - mean_absolute_error: 0.0153 - val_loss: 0.0051 - val_mean_absolute_error: 0.0563\n",
      "Epoch 9/100\n",
      "28/28 [==============================] - 1s 34ms/step - loss: 4.1519e-04 - mean_absolute_error: 0.0148 - val_loss: 0.0037 - val_mean_absolute_error: 0.0467\n",
      "Epoch 10/100\n",
      "28/28 [==============================] - 1s 32ms/step - loss: 3.8853e-04 - mean_absolute_error: 0.0145 - val_loss: 0.0037 - val_mean_absolute_error: 0.0463\n",
      "Epoch 11/100\n",
      "28/28 [==============================] - 1s 34ms/step - loss: 4.1915e-04 - mean_absolute_error: 0.0150 - val_loss: 0.0043 - val_mean_absolute_error: 0.0507\n",
      "Epoch 12/100\n",
      "28/28 [==============================] - 1s 37ms/step - loss: 4.3963e-04 - mean_absolute_error: 0.0154 - val_loss: 0.0042 - val_mean_absolute_error: 0.0509\n",
      "Epoch 13/100\n",
      "28/28 [==============================] - 1s 47ms/step - loss: 3.5439e-04 - mean_absolute_error: 0.0138 - val_loss: 0.0040 - val_mean_absolute_error: 0.0491\n",
      "Epoch 14/100\n",
      "28/28 [==============================] - 1s 35ms/step - loss: 3.5638e-04 - mean_absolute_error: 0.0138 - val_loss: 0.0042 - val_mean_absolute_error: 0.0505\n",
      "Epoch 15/100\n",
      "28/28 [==============================] - 1s 34ms/step - loss: 3.2939e-04 - mean_absolute_error: 0.0133 - val_loss: 0.0032 - val_mean_absolute_error: 0.0436\n",
      "Epoch 16/100\n",
      "28/28 [==============================] - 1s 33ms/step - loss: 3.4633e-04 - mean_absolute_error: 0.0136 - val_loss: 0.0027 - val_mean_absolute_error: 0.0390\n",
      "Epoch 17/100\n",
      "28/28 [==============================] - 1s 33ms/step - loss: 3.3274e-04 - mean_absolute_error: 0.0133 - val_loss: 0.0061 - val_mean_absolute_error: 0.0638\n",
      "Epoch 18/100\n",
      "28/28 [==============================] - 1s 33ms/step - loss: 3.3437e-04 - mean_absolute_error: 0.0135 - val_loss: 0.0036 - val_mean_absolute_error: 0.0475\n",
      "Epoch 19/100\n",
      "28/28 [==============================] - 1s 33ms/step - loss: 3.2053e-04 - mean_absolute_error: 0.0132 - val_loss: 0.0039 - val_mean_absolute_error: 0.0500\n",
      "Epoch 20/100\n",
      "28/28 [==============================] - 1s 33ms/step - loss: 3.0536e-04 - mean_absolute_error: 0.0126 - val_loss: 0.0066 - val_mean_absolute_error: 0.0676\n",
      "Epoch 21/100\n",
      "28/28 [==============================] - 1s 33ms/step - loss: 3.1899e-04 - mean_absolute_error: 0.0130 - val_loss: 0.0037 - val_mean_absolute_error: 0.0475\n",
      "Epoch 22/100\n",
      "28/28 [==============================] - 1s 33ms/step - loss: 3.0243e-04 - mean_absolute_error: 0.0129 - val_loss: 0.0042 - val_mean_absolute_error: 0.0517\n",
      "Epoch 23/100\n",
      "28/28 [==============================] - 1s 34ms/step - loss: 2.8721e-04 - mean_absolute_error: 0.0125 - val_loss: 0.0038 - val_mean_absolute_error: 0.0481\n",
      "Epoch 24/100\n",
      "28/28 [==============================] - 1s 36ms/step - loss: 2.8524e-04 - mean_absolute_error: 0.0124 - val_loss: 0.0047 - val_mean_absolute_error: 0.0556\n",
      "Epoch 25/100\n",
      "28/28 [==============================] - 1s 34ms/step - loss: 2.9709e-04 - mean_absolute_error: 0.0127 - val_loss: 0.0020 - val_mean_absolute_error: 0.0329\n",
      "Epoch 26/100\n",
      "28/28 [==============================] - 1s 32ms/step - loss: 2.8813e-04 - mean_absolute_error: 0.0125 - val_loss: 0.0039 - val_mean_absolute_error: 0.0494\n",
      "Epoch 27/100\n",
      "28/28 [==============================] - 1s 33ms/step - loss: 3.0037e-04 - mean_absolute_error: 0.0128 - val_loss: 0.0072 - val_mean_absolute_error: 0.0716\n",
      "Epoch 28/100\n",
      "28/28 [==============================] - 1s 33ms/step - loss: 2.9664e-04 - mean_absolute_error: 0.0129 - val_loss: 0.0056 - val_mean_absolute_error: 0.0618\n",
      "Epoch 29/100\n",
      "28/28 [==============================] - 1s 33ms/step - loss: 2.9759e-04 - mean_absolute_error: 0.0128 - val_loss: 0.0014 - val_mean_absolute_error: 0.0276\n",
      "Epoch 30/100\n",
      "28/28 [==============================] - 1s 32ms/step - loss: 3.4480e-04 - mean_absolute_error: 0.0141 - val_loss: 0.0011 - val_mean_absolute_error: 0.0246\n",
      "Epoch 31/100\n",
      "28/28 [==============================] - 1s 32ms/step - loss: 2.6217e-04 - mean_absolute_error: 0.0120 - val_loss: 0.0023 - val_mean_absolute_error: 0.0377\n",
      "Epoch 32/100\n",
      "28/28 [==============================] - 1s 32ms/step - loss: 2.4907e-04 - mean_absolute_error: 0.0115 - val_loss: 0.0026 - val_mean_absolute_error: 0.0407\n",
      "Epoch 33/100\n",
      "28/28 [==============================] - 1s 32ms/step - loss: 2.7553e-04 - mean_absolute_error: 0.0119 - val_loss: 0.0054 - val_mean_absolute_error: 0.0621\n",
      "Epoch 34/100\n",
      "28/28 [==============================] - 1s 33ms/step - loss: 2.8729e-04 - mean_absolute_error: 0.0125 - val_loss: 0.0035 - val_mean_absolute_error: 0.0476\n",
      "Epoch 35/100\n",
      "28/28 [==============================] - 1s 32ms/step - loss: 2.7106e-04 - mean_absolute_error: 0.0119 - val_loss: 0.0032 - val_mean_absolute_error: 0.0434\n",
      "Epoch 36/100\n",
      "28/28 [==============================] - 1s 32ms/step - loss: 2.3782e-04 - mean_absolute_error: 0.0114 - val_loss: 0.0027 - val_mean_absolute_error: 0.0398\n",
      "Epoch 37/100\n",
      "28/28 [==============================] - 1s 34ms/step - loss: 2.2473e-04 - mean_absolute_error: 0.0111 - val_loss: 0.0026 - val_mean_absolute_error: 0.0394\n",
      "Epoch 38/100\n",
      "28/28 [==============================] - 1s 33ms/step - loss: 2.2428e-04 - mean_absolute_error: 0.0110 - val_loss: 0.0024 - val_mean_absolute_error: 0.0381\n",
      "Epoch 39/100\n",
      "28/28 [==============================] - 1s 32ms/step - loss: 2.2204e-04 - mean_absolute_error: 0.0109 - val_loss: 0.0035 - val_mean_absolute_error: 0.0477\n",
      "Epoch 40/100\n",
      "28/28 [==============================] - 1s 39ms/step - loss: 2.4818e-04 - mean_absolute_error: 0.0116 - val_loss: 0.0024 - val_mean_absolute_error: 0.0392\n",
      "Epoch 41/100\n",
      "28/28 [==============================] - 1s 33ms/step - loss: 2.4285e-04 - mean_absolute_error: 0.0115 - val_loss: 0.0025 - val_mean_absolute_error: 0.0393\n",
      "Epoch 42/100\n",
      "28/28 [==============================] - 1s 37ms/step - loss: 2.2973e-04 - mean_absolute_error: 0.0112 - val_loss: 0.0018 - val_mean_absolute_error: 0.0327\n",
      "Epoch 43/100\n",
      "28/28 [==============================] - 1s 35ms/step - loss: 2.6763e-04 - mean_absolute_error: 0.0122 - val_loss: 0.0021 - val_mean_absolute_error: 0.0345\n",
      "Epoch 44/100\n",
      "28/28 [==============================] - 1s 32ms/step - loss: 2.1582e-04 - mean_absolute_error: 0.0108 - val_loss: 0.0020 - val_mean_absolute_error: 0.0350\n",
      "Epoch 45/100\n",
      "28/28 [==============================] - 1s 32ms/step - loss: 2.2262e-04 - mean_absolute_error: 0.0111 - val_loss: 0.0024 - val_mean_absolute_error: 0.0388\n",
      "Epoch 46/100\n",
      "28/28 [==============================] - 1s 32ms/step - loss: 2.1803e-04 - mean_absolute_error: 0.0107 - val_loss: 0.0017 - val_mean_absolute_error: 0.0311\n",
      "Epoch 47/100\n",
      "28/28 [==============================] - 1s 31ms/step - loss: 1.9663e-04 - mean_absolute_error: 0.0102 - val_loss: 0.0016 - val_mean_absolute_error: 0.0301\n",
      "Epoch 48/100\n",
      "28/28 [==============================] - 1s 33ms/step - loss: 2.1869e-04 - mean_absolute_error: 0.0107 - val_loss: 0.0020 - val_mean_absolute_error: 0.0340\n",
      "Epoch 49/100\n",
      "28/28 [==============================] - 1s 33ms/step - loss: 2.0821e-04 - mean_absolute_error: 0.0105 - val_loss: 0.0029 - val_mean_absolute_error: 0.0432\n",
      "Epoch 50/100\n",
      "28/28 [==============================] - 1s 33ms/step - loss: 2.3735e-04 - mean_absolute_error: 0.0115 - val_loss: 0.0023 - val_mean_absolute_error: 0.0373\n",
      "Epoch 51/100\n",
      "28/28 [==============================] - 1s 32ms/step - loss: 1.9933e-04 - mean_absolute_error: 0.0103 - val_loss: 0.0022 - val_mean_absolute_error: 0.0362\n",
      "Epoch 52/100\n",
      "28/28 [==============================] - 1s 34ms/step - loss: 2.0226e-04 - mean_absolute_error: 0.0103 - val_loss: 0.0014 - val_mean_absolute_error: 0.0280\n",
      "Epoch 53/100\n",
      "28/28 [==============================] - 1s 32ms/step - loss: 2.1465e-04 - mean_absolute_error: 0.0107 - val_loss: 0.0019 - val_mean_absolute_error: 0.0339\n",
      "Epoch 54/100\n",
      "28/28 [==============================] - 1s 33ms/step - loss: 1.9009e-04 - mean_absolute_error: 0.0100 - val_loss: 0.0028 - val_mean_absolute_error: 0.0428\n",
      "Epoch 55/100\n",
      "28/28 [==============================] - 1s 32ms/step - loss: 2.0381e-04 - mean_absolute_error: 0.0104 - val_loss: 0.0015 - val_mean_absolute_error: 0.0290\n",
      "Epoch 56/100\n",
      "28/28 [==============================] - 1s 32ms/step - loss: 1.8664e-04 - mean_absolute_error: 0.0099 - val_loss: 0.0019 - val_mean_absolute_error: 0.0345\n",
      "Epoch 57/100\n",
      "28/28 [==============================] - 1s 32ms/step - loss: 1.7744e-04 - mean_absolute_error: 0.0096 - val_loss: 0.0015 - val_mean_absolute_error: 0.0293\n",
      "Epoch 58/100\n",
      "28/28 [==============================] - 1s 32ms/step - loss: 1.6983e-04 - mean_absolute_error: 0.0093 - val_loss: 0.0015 - val_mean_absolute_error: 0.0302\n",
      "Epoch 59/100\n",
      "28/28 [==============================] - 1s 35ms/step - loss: 1.7348e-04 - mean_absolute_error: 0.0096 - val_loss: 0.0010 - val_mean_absolute_error: 0.0238\n",
      "Epoch 60/100\n",
      "28/28 [==============================] - 1s 34ms/step - loss: 2.0711e-04 - mean_absolute_error: 0.0107 - val_loss: 6.9389e-04 - val_mean_absolute_error: 0.0199\n",
      "Epoch 61/100\n",
      "28/28 [==============================] - 1s 32ms/step - loss: 2.0330e-04 - mean_absolute_error: 0.0107 - val_loss: 0.0036 - val_mean_absolute_error: 0.0502\n",
      "Epoch 62/100\n",
      "28/28 [==============================] - 1s 32ms/step - loss: 1.7130e-04 - mean_absolute_error: 0.0095 - val_loss: 0.0023 - val_mean_absolute_error: 0.0377\n",
      "Epoch 63/100\n",
      "28/28 [==============================] - 1s 32ms/step - loss: 1.7350e-04 - mean_absolute_error: 0.0095 - val_loss: 0.0012 - val_mean_absolute_error: 0.0262\n",
      "Epoch 64/100\n",
      "28/28 [==============================] - 1s 32ms/step - loss: 1.7681e-04 - mean_absolute_error: 0.0097 - val_loss: 0.0017 - val_mean_absolute_error: 0.0324\n",
      "Epoch 65/100\n",
      "28/28 [==============================] - 1s 34ms/step - loss: 1.9607e-04 - mean_absolute_error: 0.0102 - val_loss: 0.0019 - val_mean_absolute_error: 0.0348\n",
      "Epoch 66/100\n",
      "28/28 [==============================] - 1s 33ms/step - loss: 1.7603e-04 - mean_absolute_error: 0.0098 - val_loss: 0.0011 - val_mean_absolute_error: 0.0252\n",
      "Epoch 67/100\n",
      "28/28 [==============================] - 1s 35ms/step - loss: 1.6057e-04 - mean_absolute_error: 0.0092 - val_loss: 0.0013 - val_mean_absolute_error: 0.0281\n",
      "Epoch 68/100\n",
      "28/28 [==============================] - 1s 33ms/step - loss: 1.6818e-04 - mean_absolute_error: 0.0094 - val_loss: 0.0014 - val_mean_absolute_error: 0.0293\n",
      "Epoch 69/100\n",
      "28/28 [==============================] - 1s 34ms/step - loss: 1.5278e-04 - mean_absolute_error: 0.0089 - val_loss: 0.0027 - val_mean_absolute_error: 0.0433\n",
      "Epoch 70/100\n",
      "28/28 [==============================] - 1s 33ms/step - loss: 1.6069e-04 - mean_absolute_error: 0.0093 - val_loss: 8.7465e-04 - val_mean_absolute_error: 0.0216\n",
      "Epoch 71/100\n",
      "28/28 [==============================] - 1s 32ms/step - loss: 1.5866e-04 - mean_absolute_error: 0.0092 - val_loss: 0.0010 - val_mean_absolute_error: 0.0235\n",
      "Epoch 72/100\n",
      "28/28 [==============================] - 1s 33ms/step - loss: 1.5660e-04 - mean_absolute_error: 0.0090 - val_loss: 0.0014 - val_mean_absolute_error: 0.0298\n",
      "Epoch 73/100\n",
      "28/28 [==============================] - 1s 32ms/step - loss: 1.6442e-04 - mean_absolute_error: 0.0093 - val_loss: 5.9834e-04 - val_mean_absolute_error: 0.0180\n",
      "Epoch 74/100\n",
      "28/28 [==============================] - 1s 32ms/step - loss: 1.5159e-04 - mean_absolute_error: 0.0088 - val_loss: 7.7283e-04 - val_mean_absolute_error: 0.0202\n",
      "Epoch 75/100\n",
      "28/28 [==============================] - 1s 33ms/step - loss: 1.5390e-04 - mean_absolute_error: 0.0090 - val_loss: 0.0016 - val_mean_absolute_error: 0.0320\n",
      "Epoch 76/100\n",
      "28/28 [==============================] - 1s 33ms/step - loss: 1.4731e-04 - mean_absolute_error: 0.0087 - val_loss: 0.0013 - val_mean_absolute_error: 0.0278\n",
      "Epoch 77/100\n",
      "28/28 [==============================] - 1s 34ms/step - loss: 1.4074e-04 - mean_absolute_error: 0.0086 - val_loss: 0.0015 - val_mean_absolute_error: 0.0320\n",
      "Epoch 78/100\n",
      "28/28 [==============================] - 1s 33ms/step - loss: 1.5305e-04 - mean_absolute_error: 0.0090 - val_loss: 0.0011 - val_mean_absolute_error: 0.0244\n",
      "Epoch 79/100\n",
      "28/28 [==============================] - 1s 33ms/step - loss: 1.4120e-04 - mean_absolute_error: 0.0088 - val_loss: 7.8814e-04 - val_mean_absolute_error: 0.0207\n",
      "Epoch 80/100\n",
      "28/28 [==============================] - 1s 33ms/step - loss: 1.6674e-04 - mean_absolute_error: 0.0094 - val_loss: 5.3724e-04 - val_mean_absolute_error: 0.0170\n",
      "Epoch 81/100\n",
      "28/28 [==============================] - 1s 34ms/step - loss: 1.7465e-04 - mean_absolute_error: 0.0097 - val_loss: 9.3549e-04 - val_mean_absolute_error: 0.0240\n",
      "Epoch 82/100\n",
      "28/28 [==============================] - 1s 32ms/step - loss: 1.4862e-04 - mean_absolute_error: 0.0088 - val_loss: 0.0013 - val_mean_absolute_error: 0.0280\n",
      "Epoch 83/100\n",
      "28/28 [==============================] - 1s 34ms/step - loss: 1.3837e-04 - mean_absolute_error: 0.0084 - val_loss: 9.0192e-04 - val_mean_absolute_error: 0.0227\n",
      "Epoch 84/100\n",
      "28/28 [==============================] - 1s 32ms/step - loss: 1.2769e-04 - mean_absolute_error: 0.0082 - val_loss: 0.0017 - val_mean_absolute_error: 0.0328\n",
      "Epoch 85/100\n",
      "28/28 [==============================] - 1s 34ms/step - loss: 1.4004e-04 - mean_absolute_error: 0.0086 - val_loss: 8.2948e-04 - val_mean_absolute_error: 0.0210\n",
      "Epoch 86/100\n",
      "28/28 [==============================] - 1s 32ms/step - loss: 1.4636e-04 - mean_absolute_error: 0.0089 - val_loss: 8.2554e-04 - val_mean_absolute_error: 0.0213\n",
      "Epoch 87/100\n",
      "28/28 [==============================] - 1s 32ms/step - loss: 1.2793e-04 - mean_absolute_error: 0.0082 - val_loss: 0.0016 - val_mean_absolute_error: 0.0319\n",
      "Epoch 88/100\n",
      "28/28 [==============================] - 1s 33ms/step - loss: 1.4175e-04 - mean_absolute_error: 0.0086 - val_loss: 5.3922e-04 - val_mean_absolute_error: 0.0172\n",
      "Epoch 89/100\n",
      "28/28 [==============================] - 1s 34ms/step - loss: 1.3783e-04 - mean_absolute_error: 0.0084 - val_loss: 6.6209e-04 - val_mean_absolute_error: 0.0185\n",
      "Epoch 90/100\n",
      "28/28 [==============================] - 1s 32ms/step - loss: 1.2677e-04 - mean_absolute_error: 0.0081 - val_loss: 0.0017 - val_mean_absolute_error: 0.0332\n",
      "Epoch 91/100\n",
      "28/28 [==============================] - 1s 33ms/step - loss: 1.2505e-04 - mean_absolute_error: 0.0082 - val_loss: 5.2412e-04 - val_mean_absolute_error: 0.0167\n",
      "Epoch 92/100\n",
      "28/28 [==============================] - 1s 32ms/step - loss: 1.2061e-04 - mean_absolute_error: 0.0079 - val_loss: 8.0274e-04 - val_mean_absolute_error: 0.0213\n",
      "Epoch 93/100\n",
      "28/28 [==============================] - 1s 34ms/step - loss: 1.2418e-04 - mean_absolute_error: 0.0080 - val_loss: 0.0021 - val_mean_absolute_error: 0.0386\n",
      "Epoch 94/100\n",
      "28/28 [==============================] - 1s 39ms/step - loss: 1.3460e-04 - mean_absolute_error: 0.0086 - val_loss: 6.5789e-04 - val_mean_absolute_error: 0.0188\n",
      "Epoch 95/100\n",
      "28/28 [==============================] - 1s 35ms/step - loss: 1.2995e-04 - mean_absolute_error: 0.0082 - val_loss: 0.0012 - val_mean_absolute_error: 0.0274\n",
      "Epoch 96/100\n",
      "28/28 [==============================] - 1s 36ms/step - loss: 1.4096e-04 - mean_absolute_error: 0.0086 - val_loss: 0.0012 - val_mean_absolute_error: 0.0272\n",
      "Epoch 97/100\n",
      "28/28 [==============================] - 1s 37ms/step - loss: 1.3420e-04 - mean_absolute_error: 0.0085 - val_loss: 0.0016 - val_mean_absolute_error: 0.0331\n",
      "Epoch 98/100\n",
      "28/28 [==============================] - 1s 33ms/step - loss: 1.2673e-04 - mean_absolute_error: 0.0081 - val_loss: 5.2320e-04 - val_mean_absolute_error: 0.0169\n",
      "Epoch 99/100\n",
      "28/28 [==============================] - 1s 33ms/step - loss: 1.1939e-04 - mean_absolute_error: 0.0078 - val_loss: 7.9773e-04 - val_mean_absolute_error: 0.0201\n",
      "Epoch 100/100\n",
      "28/28 [==============================] - 1s 33ms/step - loss: 1.2219e-04 - mean_absolute_error: 0.0079 - val_loss: 0.0012 - val_mean_absolute_error: 0.0266\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.callbacks.History at 0x1d101665a30>"
      ]
     },
     "execution_count": 56,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# fitting the model by iterating the dataset over 100 times(100 epochs)\n",
    "model.fit(train_seq, train_label, epochs=100,validation_data=(test_seq, test_label), verbose=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "id": "7ac85b1e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "9/9 [==============================] - 1s 10ms/step\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "array([[0.40134126, 0.39790046],\n",
       "       [0.4016015 , 0.39767465],\n",
       "       [0.397627  , 0.3933278 ],\n",
       "       [0.39950964, 0.39557028],\n",
       "       [0.402757  , 0.39912042]], dtype=float32)"
      ]
     },
     "execution_count": 57,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# predicting the values after running the model\n",
    "test_predicted = model.predict(test_seq)\n",
    "test_predicted[:5]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "id": "b2e6301c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1415.0546, 1405.7048],\n",
       "       [1415.5371, 1405.2864],\n",
       "       [1408.1687, 1397.2301],\n",
       "       [1411.6588, 1401.3862],\n",
       "       [1417.6792, 1407.9658]], dtype=float32)"
      ]
     },
     "execution_count": 58,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Inversing normalization/scaling on predicted data \n",
    "test_inverse_predicted = MMS.inverse_transform(test_predicted)\n",
    "test_inverse_predicted[:5]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6688e1e2",
   "metadata": {},
   "source": [
    "## STEP 5 :  VISUALIZING ACTUAL VS PREDICTED DATA"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "4913627a",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Merging actual and predicted data for better visualization\n",
    "df_merge = pd.concat([df.iloc[-264:].copy(),\n",
    "                          pd.DataFrame(test_inverse_predicted,columns=['open_predicted','close_predicted'],\n",
    "                                       index=df.iloc[-264:].index)], axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "id": "637b7853",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>open</th>\n",
       "      <th>close</th>\n",
       "      <th>open_predicted</th>\n",
       "      <th>close_predicted</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>date</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2020-05-27</th>\n",
       "      <td>1417.25</td>\n",
       "      <td>1417.84</td>\n",
       "      <td>1415.054565</td>\n",
       "      <td>1405.704834</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2020-05-28</th>\n",
       "      <td>1396.86</td>\n",
       "      <td>1416.73</td>\n",
       "      <td>1415.537109</td>\n",
       "      <td>1405.286377</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2020-05-29</th>\n",
       "      <td>1416.94</td>\n",
       "      <td>1428.92</td>\n",
       "      <td>1408.168701</td>\n",
       "      <td>1397.230103</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2020-06-01</th>\n",
       "      <td>1418.39</td>\n",
       "      <td>1431.82</td>\n",
       "      <td>1411.658813</td>\n",
       "      <td>1401.386230</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2020-06-02</th>\n",
       "      <td>1430.55</td>\n",
       "      <td>1439.22</td>\n",
       "      <td>1417.679199</td>\n",
       "      <td>1407.965820</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "               open    close  open_predicted  close_predicted\n",
       "date                                                         \n",
       "2020-05-27  1417.25  1417.84     1415.054565      1405.704834\n",
       "2020-05-28  1396.86  1416.73     1415.537109      1405.286377\n",
       "2020-05-29  1416.94  1428.92     1408.168701      1397.230103\n",
       "2020-06-01  1418.39  1431.82     1411.658813      1401.386230\n",
       "2020-06-02  1430.55  1439.22     1417.679199      1407.965820"
      ]
     },
     "execution_count": 60,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Inversing normalization/scaling \n",
    "df_merge[['open','close']] = MMS.inverse_transform(df_merge[['open','close']])\n",
    "df_merge.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "id": "0e552918",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x432 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# plotting the actual open and predicted open prices on date index\n",
    "df_merge[['open','open_predicted']].plot(figsize=(10,6))\n",
    "plt.xticks(rotation=45)\n",
    "plt.xlabel('Date',size=15)\n",
    "plt.ylabel('Stock Price',size=15)\n",
    "plt.title('Actual vs Predicted for open price',size=15)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "id": "8b19365b",
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "data": {
      "image/png": "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9HaTWpDgzd63v4esPV17zf5j4YIn26zyzjvv8PTTT6OUIiEhgcGDBzcc27lzZ8P2WIMHD+aNN94gODiY5557jpdffhk3NzdGjRrFokWLKCsr47777mPPnj3U1tby6KOPcumllzq851tvvcXnn39OVVUVycnJXH/99fzlL3/h2LFjXHDBBcybN48NGzbwxRdfsHjxYhYvXkxVVRWXX345jz32GABPPPEE77zzDgMGDCA8PJxJkyYBcMstt7BgwQKuvPJKtmzZwv33309ZWRmenp6sWLGCRx55hIqKCtatW8fDDz/MggULHLa7oqKCW2+9lf379zNy5EgqKiq66IcihBBCtFRZU8fvP93NFzszWJAQyT+vHIe3h2tvN6sFCd562b59+3jiiSdYv349YWFhFBQU8NxzzzUcv/nmm3n++eeZM2cOjzzyCI899hjPPvssTz75JMnJyXh6emKxWAATTJ111lm88cYbWCwWpk6dyjnnnIOvr6/De2/evJm9e/fi4+PDlClTuOiiiwgLC+PQoUO8+eabvPTSSyxfvpzExEQ2b96M1ppLLrmENWvW4Ovry6JFi9ixYwe1tbVMnDixIXirV11dzTXXXMNHH33ElClTKC4uxsfHh8cff5ytW7fywgsvAPCHP/zBYbv/97//4ePjw+7du9m9ezcTJ07snh+CEEIIAfzhsz18sTODB84bxj3zhqCU6u0mOSTBm712MmTd4YcffuDKK68kLCwMgJCQkIZjRUVFWCwW5syZA8DChQu56qqrAEhISOCGG27gsssu47LLLgNg+fLlLFmypGHsWWVlJSkpKa3u+XnuuecSGhoKwBVXXMG6deu47LLLiIuLY/r06Q3XXL58ORMmmLRxaWkpiYmJlJSUcPnll+PjY8YBXHLJJS2uf+jQISIjI5kyZQoAAQGOV6Zurd1r1qzhl7/8ZcPnTUhIaPf7FEIIIU7UtpRCLhobyb1nDe3tprRJgrdeprU+och+6dKlrFmzhiVLlvDXv/6Vffv2obXm008/Zfjw4U5do/l969/bZ+q01jz88MPcddddTeo+++yz7bbb2c/WVrv76r96hBBCnFqsVk2GpYILxkT2dlPaJbNNe9nZZ5/N4sWLyc/PB6CgoKDhWGBgIMHBwaxduxaAd999lzlz5mC1WklNTWXevHk89dRTWCwWSktLmT9/Ps8//zxa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\n",
      "text/plain": [
       "<Figure size 720x432 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# plotting the actual close and predicted close prices on date index \n",
    "df_merge[['close','close_predicted']].plot(figsize=(10,6))\n",
    "plt.xticks(rotation=45)\n",
    "plt.xlabel('Date',size=15)\n",
    "plt.ylabel('Stock Price',size=15)\n",
    "plt.title('Actual vs Predicted for close price',size=15)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "822b3580",
   "metadata": {},
   "source": [
    "## STEP 6. PREDICTING UPCOMING 10 DAYS"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "id": "13e1479e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>open</th>\n",
       "      <th>close</th>\n",
       "      <th>open_predicted</th>\n",
       "      <th>close_predicted</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2021-06-09</th>\n",
       "      <td>2499.50</td>\n",
       "      <td>2491.40</td>\n",
       "      <td>2419.937012</td>\n",
       "      <td>2351.718994</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-06-10</th>\n",
       "      <td>2494.01</td>\n",
       "      <td>2521.60</td>\n",
       "      <td>2429.809814</td>\n",
       "      <td>2357.459229</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-06-11</th>\n",
       "      <td>2524.92</td>\n",
       "      <td>2513.93</td>\n",
       "      <td>2435.215576</td>\n",
       "      <td>2360.057861</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-06-12</th>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-06-13</th>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-06-14</th>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-06-15</th>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-06-16</th>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "               open    close  open_predicted  close_predicted\n",
       "2021-06-09  2499.50  2491.40     2419.937012      2351.718994\n",
       "2021-06-10  2494.01  2521.60     2429.809814      2357.459229\n",
       "2021-06-11  2524.92  2513.93     2435.215576      2360.057861\n",
       "2021-06-12      NaN      NaN             NaN              NaN\n",
       "2021-06-13      NaN      NaN             NaN              NaN\n",
       "2021-06-14      NaN      NaN             NaN              NaN\n",
       "2021-06-15      NaN      NaN             NaN              NaN\n",
       "2021-06-16      NaN      NaN             NaN              NaN"
      ]
     },
     "execution_count": 63,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Creating a dataframe and adding 10 days to existing index \n",
    "\n",
    "df_merge = df_merge.append(pd.DataFrame(columns=df_merge.columns,\n",
    "                                        index=pd.date_range(start=df_merge.index[-1], periods=11, freq='D', closed='right')))\n",
    "df_merge['2021-06-09':'2021-06-16']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "id": "e6822e89",
   "metadata": {},
   "outputs": [],
   "source": [
    "# creating a DataFrame and filling values of open and close column\n",
    "upcoming_prediction = pd.DataFrame(columns=['open','close'],index=df_merge.index)\n",
    "upcoming_prediction.index=pd.to_datetime(upcoming_prediction.index)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "id": "8cb8738d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1/1 [==============================] - 0s 36ms/step\n",
      "1/1 [==============================] - 0s 23ms/step\n",
      "1/1 [==============================] - 0s 30ms/step\n",
      "1/1 [==============================] - 0s 24ms/step\n",
      "1/1 [==============================] - 0s 20ms/step\n",
      "1/1 [==============================] - 0s 21ms/step\n",
      "1/1 [==============================] - 0s 24ms/step\n",
      "1/1 [==============================] - 0s 31ms/step\n",
      "1/1 [==============================] - 0s 26ms/step\n",
      "1/1 [==============================] - 0s 30ms/step\n"
     ]
    }
   ],
   "source": [
    "curr_seq = test_seq[-1:]\n",
    "\n",
    "for i in range(-10,0):\n",
    "  up_pred = model.predict(curr_seq)\n",
    "  upcoming_prediction.iloc[i] = up_pred\n",
    "  curr_seq = np.append(curr_seq[0][1:],up_pred,axis=0)\n",
    "  curr_seq = curr_seq.reshape(test_seq[-1:].shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "id": "37525dd7",
   "metadata": {},
   "outputs": [],
   "source": [
    "# inversing Normalization/scaling\n",
    "upcoming_prediction[['open','close']] = MMS.inverse_transform(upcoming_prediction[['open','close']])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "id": "3c7187b0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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qVMZMWJE8coZb731FgMGDKBv376cdNJJ5cYhIiKHZ/e+fK58YiYzV2yJdSgJSS1vEX732nwWrN1Rqcfs0bI+t32rZ5nb582bx8CBAw/r2Onp6UyZMgWAX/ziF8WPN23axDnnnMN7771H3bp1ufvuu/nTn/7Eb37zGwCaNm3KrFmzePjhh7nvvvt49NFHueKKK8jIyOCGG2444HXmz59/QPmgQYP461//Wvx4165dfPLJJ0yePJnLLruMefPmceedd3LiiScyYcIEtm3bxpAhQzj55JOBoHVx9uzZpKWl0bVrV6666iratGlT5nt97bXX6N27d/Fr9erVi9tvv32/fbKzs/nxj3/M5MmT6dChA1u2BL9Uyoqjbt26h1rlIiIJb8uuffxg4mfMXb2Nk7pnMbBd41iHlHCUvMWx888/v9THn376KQsWLODYY48FYN++fRxzzDHF+51zzjkADBw4kBdffPGgr+PuB1ygX7LswgsvBIIWsh07drBt2zbeeecdXn31Ve677z4gGNm7cuVKAE466SQaNGgAQI8ePVixYkWpydtFF11E7dq1ad++PQ8++CAAycnJfPvb3z5g308//ZThw4cXtxA2bhz8Qikrju7dux/0vYuIyDdWb93NJROms2brHh753kBO6dk81iElJCVvEcprIYuWnj178vzzz5e6LSUlhcLCwuLHJaemKNlyVPTY3Rk1ahRPP/10qcdNS0sDgiQoPz+/QjHOmDFjv+vbZs2aRY8ePYofl0zuzAx354UXXqBr1677bZs2bVpxDAeLo+iat0jp6emlXtNXWpJZVF5aHCIiUnGL1udwyYRp7N5XwL9/eDRDOqjFLVZ0zVuMnXjiieTm5vKPf/yjuOyzzz7jo48+ol27dixYsIDc3Fy2b9/O+++/X6FjDh06lI8//rj4GrHdu3cfdPRqvXr1yMnJKXXbDTfcwF133cXy5csBWL58OX/4wx+4/vrri/d59tlnAZgyZQoNGjSgQYMGjB49mgcffBB3B2D27NkViv9wHXPMMXz00UcsW7YMoLjbtKrjEBGpaT5bvoXzHglmJPjPFccocYsxtbzFmJnx0ksvce211/LHP/6R9PR02rdvz5///GfatGnDd77zHfr06UPnzp3p379/hY6ZmZnJxIkTufDCC8nNzQXg97//fbmjV7/1rW9x7rnn8sorr/Dggw9y/PHHF2/r168fd999N9/61rfIy8sjNTWVe+65h379+hXv06hRI4YNG8aOHTuYMGECAL/+9a+59tpr6dOnD+5O+/btef311w+jliomMzOT8ePHc84551BYWEizZs149913qzwOEZF4snTjTv7wxpfMWL6FDpkZHJWZwVHNglvnZhks2pDD1U/PplWj2jx+2RBaN6oT65ATnhW1RiSCQYMGedGoxCJffvmlrn06QiNGjOC+++47oHuzptNnR0Ti2fbdefz5/cX8e+oKaqcmc2rv5qzZtoelG3eyYUfufvv2bdOQf31/MI3r1opRtInJzGa6+wF/XNXyJiIikkDyCwp5evpK/vTuYrbtyeOCwW25/pQuNM345lrkHXvzWLpxJ0s37iRnbz4XDG5D3TSlDNWFzoQcsUmTJsU6BBERqYBpX2/mN6/MZ9GGHIZ2bMxvzuhJj5b1D9ivfnoqA9o2YkDbRjGIUg5GyZuIiEgN5+489slybn99AS0b1uZvFw1gTK/mWqc5Til5ExERqcHyCgr53WvzeeLTlZzcPYu/XNBPXaBxTmdPRESkhtq+O4+fPjWTj5du5icndOTm0d1ISlJrW7xT8iYiIlIDLdu0ix9O/IxVW3dz77l9OG9Q2UsQSnzRJL0xtnz5cnr16rVf2W9/+9vipZyq0owZM7j66qsr7Xjjx4+nW7dudOvWjSFDhhSvwxpN7du3p3fv3vTt25dTTjmF9evXl7rfaaedxrZt26Iej4hILHyydBNj//oxW3fv48kfDVXiVsOo5U2KDRo0qNLmanv99df5+9//zpQpU2jatCmzZs1i7NixTJ8+nebNo7sW3ocffkjTpk355S9/yR/+8AceeOCB4m3ujrvzxhtvRDUGEZGqtDevgGnLtvDRomw+WryRr7J30blZBv+8dDBtm2hS3ZpGLW/V3IgRI7j22msZNmwYvXr1Yvr06QDs3LmTH/zgB/Tu3Zs+ffrwwgsvAPD000/Tu3dvevXqxc0331x8nIyMDG6++WYGDhzIySefzPTp0xkxYgQdO3bk1VdfBYIpP8444wwgaP277LLLiveJTIDuuOMOunXrxqhRo7jwwgtLbSW8++67uffee2natCkAAwYM4NJLL+Wvf/0rELSQ3XzzzQwZMoQhQ4YUL+WVnZ3Nt7/9bQYPHszgwYP5+OOPDxpPWYYPH87SpUtZvnw53bt356c//SkDBgxg1apVtG/fnk2bNgHw+OOP06dPH/r27cvFF19cbhwiItWBu/NV9k4mTFnGpROm0/d373DphOk8MW0FrRrV4ddn9ODFnw5T4lZDqeUt0pu/gPVzK/eYzXvDqX88okPs2rWLTz75hMmTJ3PZZZcxb9487rjjDho0aMDcuUG8W7duZe3atdx8883MnDmTRo0accopp/Dyyy8zduxYdu3axYgRI7j77rs5++yzufXWW3n33XdZsGABl156KWeeeeYBr7tw4UI+/PBDcnJy6Nq1K1deeSWff/45L7zwArNnzyY/P58BAwYwcODAA547f/78A8oHDRrEY489Vvy4fv36TJ8+nccff5xrr72W119/nWuuuYbrrruO4447jpUrVzJ69Gi+/PLLMuNJTU0ts95ef/11evfuDcCiRYv417/+xcMPP3xAnHfeeScff/wxTZs2LV4Ptbw4RKT6KCh03pi7jmb10hjSoXG1mfoir6CQzTv30bxBeqUdc2duPlO/2sxHizfy0eJsVm3ZA0DHzLp89+i2nNAlk6M7NKF2reRKe83DsmsT7N4CmWUvyShHRslbjJX1iyay/MILLwSClqQdO3awbds23nvvPZ555pnifRo1asTkyZMZMWIEmZmZAFx00UVMnjyZsWPHUqtWLcaMGQNA7969SUtLIzU1ld69excvOF/S6aefTlpaGmlpaTRr1owNGzYwZcoUzjrrLGrXrg0Ea6JWlLuX+r4uvPBCrrvuOgDee+89FixYULzPjh07yMnJKTOe1q1bH/A6I0eOJDk5mT59+vD73/+ebdu20a5dO4YOHXrAvh988AHnnntucQth48aNy42jXr16FX6/IhJdq7fu5ufPfs705cE/XT1b1ueyYztwRt8WpKXEJoFZtmkXz81YxfMzV7N5Zy6PXjqIE7tlHdax3J2F63P4aHE2Hy3KZsaKLeQVOHVrJTPsqKb8ZHgnTuiSSZvG1ax17bFvQUYzuOSVWEdSYyl5i3SELWSHo0mTJmzdunW/si1bttChQ4fixyUTPDM7IBGC4IteltTU1OL9k5KSSEtLK76fn59f6nOK9gFITk4mPz+/3NeI1KNHD2bOnMmJJ55YXDZr1ix69OhR6vsqul9YWMjUqVOLk8ODxVOaomveimzbto26deuWum9p9XiwOEQk9l6evYZfvzwPB+45tw8Fhc6EKcu4/j+f88e3FnLx0HZcdHRbmkQs+RQte/MKeGveep75bCWffr2F5CRjZNdmrN22h6uems3zVw6je4sDVzEozbbd+5iydBMfLcpm8pLs4jVGuzWvx2XHdeCELpkMateYWinV+KqnzqfA1Idg73ZIbxDraGokJW8xlpGRQYsWLXj//fc56aST2LJlC2+99RbXXHNN8T7PPvssI0eOZMqUKTRo0IAGDRpwyimn8NBDD/HnP/8ZCLpNjz76aK655ho2bdpEo0aNePrpp7nqqqsqNd7jjjuOn/zkJ9xyyy3k5+fz3//+lx//+McH7HfTTTdx880389Zbb9GkSRPmzJnDxIkTmTZt2n7v6xe/+AXPPvssxxxzDEDx+7rxxhsBmDNnDv369avU9xDppJNO4uyzz+a6666jSZMmbNmyhcaNG1d5HCIS2LJrH9OXbaZHiwa0aVz7gH+utu/J49cvz+PVz9cyqF0j7j+/X3HL0wWD2zB5ySYmTFnGn95dzEMfLqVfm4ZEc1ozd/hy3Q527M2nXZM63DSmK+cOaE2z+uls2LGXsx76mB89NoOXxg2jWb2yu1C37NrHTc9/zgcLN1LoUD89heO7ZHJCeMuqX3ndr1HX9VT4+M+w9H3odU6so6mRlLxVA48//jjjxo3j+uuvB+C2226jU6dOxdsbNWrEsGHD2LFjBxMmTADg1ltvZdy4cfTq1Yvk5GRuu+02zjnnHO666y5GjhyJu3Paaadx1llnVWqsgwcP5swzz6Rv3760a9eOQYMG0aDBgf9ZnXnmmaxZs4Zhw4ZhZtSrV48nnniCFi1aFO+Tm5vL0UcfTWFhIU8//TQADzzwAOPGjaNPnz7k5+czfPhwHnnkkUp9D5F69uzJr371K0444QSSk5Pp378/EydOrPI4RCRoCb/66dlMWRoMJmpUJ5U+rRvSt3UD+rZpSFKScetL81i/Yy/Xj+rClSM6kZL8TQuUmRUnO0s25PDY1OUsXr+TwijHfXL3LM4b1IajOzTebwLcrPrpPHrpIM57ZCqXPz6TZy4fSnrqgd2589du5/LHZ5K9M5crR3TixG5Z9G3dYL/3FldaD4Y6TWDRm0reosQq2g1WKS9m1gZ4HGgOFALj3f0vEdtvAO4FMt19U1h2C/BDoAC42t3fDssHAhOB2sAbwDV+kDczaNAgnzFjxn5lX375Jd27d6+U9xcNI0aM4L777qu0KTwqw86dO8nIyGD37t0MHz6c8ePHM2DAgEM6Rvv27ZkxY8Z+3Zvxprp/dsqSszeP7Jzc4LYzt/j+rtx82jWpS9fm9eiSVY+mGbWqzcXfkhiem7GKm57/gmtO6kxW/XQ+X7WNz1dvY/GGHArD3+7tm9Th/vP70T+OFkx/e/56rnhiJqf1bsGDF/TfL8F79fO13PT85zSqU4u/XzyQPq0bxi7QyvTSlbDoDbjxK0hWO9HhMrOZ7n5AAlDVNZoPXO/us8ysHjDTzN519wVhYjcKWFm0s5n1AC4AegItgffMrIu7FwB/Ay4HPiVI3sYAb1bt20lMl19+OQsWLGDv3r1ceumlh5y4Sez85b0l3P/e4gPKU5KM2qnJ5OR+cx1h47q16JKVQedm9ahfO4XkpCRSkoyUZAt+JiWF94Py5OJtSd/skxw+J9zWoHYtjmqWUZVvWeLExpy93PnfLxncvhHXnNSZpCTju0e3BWD3vnzmrdnB6q27Gd2zedytyzm6Z3N+MaYbd725kE5N6/LzU7pSUOjc8/ZC/v7R1wxu34iHLxpIZr3oX59XZbqOgc+fglXToP2xsY6mxqnSb4C7rwPWhfdzzOxLoBWwALgfuAmIHJ5yFvCMu+cCy8xsKTDEzJYD9d19KoCZPQ6MpQYmb5MmTYp1CAd46qmnjvgYZY1wlejJzsnl4UlLOb5zU749oDWZ9dLIrJdG04w0GtZOxQw27dzH4g05LFqfw+INwe3lOWvYs6+A/MLKaaW/+qTOXHdy56i06m3M2ctdbyykQ9O6HN2hMX3bNCy1m0qqn9+9uoA9+wq465w+B6y9WadWCkM6NGZIh8Yxiu7IXT68I19n7+KBD5bStF4a7325kcmLs/ne0Lb85oye1XsAwuHodCIk1wpa35S8VbqY/ftiZu2B/sA0MzsTWOPun5f4hd6KoGWtyOqwLC+8X7K8tNe5nKCFjrZt21ZW+CJxZ8LHy9hXUMjvzuxJx8zSW7+KErpjjzqwO9vdKSh08otuBYXkFwZleQWF4c/9HxftV1Do5BU6r85ZywPvLyE3v4BfjOlWqQmcu3PT81/wvyWbKHTHHWqlJNGvTUOGdmjM0R2bMKBto9jPgSUHeGf+ev47dx03nNKlxrbMmhl3jO3Fyi27+c0r80lNNu46pzcXDqmhf5fS6kH742HxWzD6zlhHU+PEJHkzswzgBeBagq7UXwGnlLZrKWVeTvmBhe7jgfEQXPNWxj66tkcOSVVeK1oZtu/O499TV3Ba7xZlJm4HYxZ2ix5B7nP8UU2pXSuJv3/0Nbl5hdz2rR6V9t17avpKJi3K5ndn9mRsv1Z8tnwL05ZtZtqyLTz04VIe+GApqclGn9YNOTpsxRnUvjEZcdYFV9Ps2JvHr1+ZR7fm9fjJCZ0O/oQ4Vislib99bwB3v7WIcwe2YmC7+G1JrJCup8IbN8CmJdC0c6yjqVGq/LeWmaUSJG5PuvuLZtYb6AAUtbq1BmaZ2RCCFrXI1XRbA2vD8tallB+y9PR0Nm/eTJMmTZTASYW4O5s3byY9PX6G7j82dTk7c/MZN+KomMaRlGTccVYv0lKS+eeUZeTmF3Ln2F4HdJMdquWbdvH717/k+M5NuXhoO5KSjJN7ZHFyj2By1Jy9ecxYsZVpX29h+rLNjJ/8NQ9P+orkJKNXy/oc3bEJR4fJXIPaZa/aIZXvj28uJDsnl/EXDyI1XkdXHoKGdWpx1zm9Yx1G1egyJkjeFr2p5K2SVWnyZkF29E/gS3f/E4C7zwWaReyzHBjk7pvM7FXgKTP7E8GAhc7AdHcvMLMcMxsKTAMuAR48nJhat27N6tWryc7OPpK3JgkmPT291NUdqqNduflM+HgZJ3VrRo+WFZsoNJrMjFtP705aShIPT/qKffmF3HNuH5IPM4ErKHR+/twcUpONe8498HopgHrpqYzs2oyRXYNfNbv35TNrxbagZe7rLUz8eDnjJ3+NGfRoUZ8hHRpzdIcmHNOxCQ3qKJmLlk+/3sxT01byo+M60LdNw1iHI5WtYRvI6h10nR57dayjqVGquuXtWOBiYK6ZzQnLfunub5S2s7vPN7PnCAY05APjwpGmAFfyzVQhb3KYgxVSU1P3W81ApKZ5evpKtu3O46cjY9vqFsnMuHF0V9JTk/nTu4uDa+BO7Ubt1GRq10omPSW5wq1xj3z0FbNWbuMvF/SjRYOKrYhRp1YKx3VuynGdg2v79uYVMGfVNqZ9HXS1Pj19Jf/6eDmN6qTy3s9PqJJZ+hPN3rwCbnlxLm0a1+bnp2gNzBqr6xj43/8Fa53WqeHdxFWoqkebTqH069Ui92lf4vGdwAFXO7r7DKBXZcYnUtPk5hcwfvLXHNOxCQPbVa95scyMq0/qTK2UJP745kJe/2LdftvTUpKoXSuZ3q0acN2oLgwoZV6v+Wu38+f3FnN6nxac2bflYceSnprM0I5NGNqxCdCZffmF/G9JNj98bAbPzVjNlSNq9rVYFbVqy27emLuO/EKnsNApdChwLx7MUuhQWHy/nH0KndXb9rBs0y6e+OHR1Kml6w5rrK6nwuR7Ycm70Pf8WEdTY+gbI1KDPT9zNRtzcrn//H6xDqVMV5zQiQFtG7F88y725hWwe18Be/YVsDevgJ25+bw1bz3nPPwJJ3dvxvWndC1eI3JvXgHXPTuHRnVqcefYXpV6zWqtlCRO6p7F0R0a89T0FfxkeMcjvi4v3q3ZtofzHpnK+h17D9iWZJBkRlKSkWwWPE4ykiyY/694e/jYDJKTguS9qPVTaqgW/SEjK5gyRMlbpVHyJlJD5RcU8shHX9GvTUOGdWoS63DKVd4cXr88rTsTP1nOIx99xal/+R/f6tuS607uzNPTV7J4w04m/mAwDevUikpc3xvajquens1HS7KLr5dLRNt27+PSCdPZlZvP61cdR+esjCARsyAR02AvKVNSUjBwYd6LkL8PUqLzXU00NX9oj0iCeu2LtazasodxI4+K6z+uddNSGDfyKKbcdCLjRnbivQUbGHX/ZB6dsoyLjm7LiCgmVaN7NqdpRhpPfroiaq9R3e3NK+CHj81g5ebdjL9kEL1aNSAtJZnU5CSSkiyuP1tSRbqeCvtyYMWUWEdSYyh5E6mBCgudhz/8im7N63FSt5rRYtSgTio3ju7G5JtGcskx7RjeOZNfnR7dtWVrpSRx/uDWfLBwI2u27Ynqa1VH+QWF/Oyp2cxauZX7z+/HMdW8BVeqqQ4nQEo6LHor1pHUGEreRGqgdxZsYMnGnfx05FE17lqtzHpp3Patnjx22ZAqudD9wiFtceDpaSsPum91lZtfcPCdSnB3fv3KfN77cgO3ndGD0/u0iEJkkhBq1YGOI2HxmxBnE5xXV0reRGqQffmF/GfGKm5/bT7tm9Th9N76g3ukWjeqw4ldm/HMZ6vYl18Y63AOSW5+Abe8+AV9f/cOi9bnHNJzH3h/KU9PX8lPR3Ti+8dqOiU5Ql3HwLaVsHFBrCOpEZS8idQAu/flM2HKMkbc+yE3Pv8FDerU4v++0/ewJ76V/X1vaDs27czlnQXrYx1KhW3M2ct3/zGNp6evoqDQ+dukpRV+7jPTV3L/e4v59oDW3Di6axSjlITRZUzwc9FhTckqJWi0qUgc27Z7H49PXcG/Pl7G1t15DGnfmDvP6c2ILpm6kLwSDe+SSetGtXni0xWc0efw55OrKp+v2sZP/j2T7XvyeOi7/fl81TYmfLyc60/pSpvGdcp97sYde/ndaws47qim/PHbvfU5kspRrzm0HBAkb8NviHU0cU/Jm0icySsoZMqSTbw8Zw3vzN/AnrwCTurWjCtHdGJQe81gHg3JScZFR7fj7rcWsnRjDkc1qxfrkMr0wszV3PLSXDIz0nj+ymPo2bIBg9s35rFPVjB+8tfcMbb8uc0f+GAJeQWF/H5sr4RYa1SqUOdT4KO7ITcH0qrvdygeKHkTiQPuzuxV23hl9hpe/2Idm3fto0HtVMb2b8Ulx7QrnrhWouc7g1pz/7uLeeLTlfz2zJ6xDucA+QWF3PXmQv45ZRlDOzbmr98dULysV1b9dM4Z0IrnZqzi6pM6k1mv9OW+lm/axTPTV3HBkDa0b1q3KsOXRNB6EOCw7nNof1yso4lrSt5EDuLr7J1M+HgZSWZcd3IXGtWtukkmv8reySuz1/DK52tZsXk3tVKSGNU9i7P6teSErpmkpSRXWSyJrklGGqf2bs4Ls1Zz05iuMVvSyd3Jzsll8YadLNqQw+L1OSzakMPSjTvZmZvP94e151endz+g1ezy4R15dsYq/vXxMm4a063UY9/3ziJSk5O4+qTOVfFWJNG06Bf8XDtbydsRUvImcW377jx+/co85q/dTvMG6TSvX5sWDdLJapBOi/rpQVmDdBrXqXXIU2Z8vmobj3z0FW/NX0+t5CQKCp035q7nzrN7Mbpn8yi9o+BC89c+X8crc9bwxertmMGwTk0YN/IoxvRqTv301Ki9tpTve0Pb8cqctbw6Zy0XDGkb9dfbtnsfi9bnsHjjzuIkbfGGHLbtzivep0ndWnTJqse5A1tz7FFNGdUjq9RjdczM4LReLfj31BVcMaLTAZ+juau38/oX6/jZyKNoVi89qu9LElRGJjRoEyRvckSUvEnc+mL1Nn765Cw27NjLCV2asXlXLp98tYmNObkUFO4/l1Ct5CSa1U+jRYN0mjeoTfP6aeHPILlr0SCdzHpppCQZ/1uyiUc++opPvtpM/fQUxo04iu8f254NO/Zyw3++4Cf/nsmZfVvy2zN70ricVrjd+/Ir3DqzMzeft+et5+U5a/h46SYKHXq2rM+tp3fnW31bklVff0yrg0HtGtE1qx5PTFvB+YPbVNrF/Dtz81kSJmaL1u9kycYcFq3PYWNObvE+9dJT6JJVj1N7taBrVgZdmtejS1Y9mmaU3gVamitO6MR/567jyU9XcuWITvttu+fthTSqk8rlJ3SslPckUqqW/ZW8VQIlbxJ33J0npq3kjtcWkFkvjed+cgz92zYq3l5Q6Gzamcv67XtZt30v67fvYf2O3PDnXuau3sa7O/ayN2//ObvMoH56Ktv35JFVP41fndadC49uS0Za8DVpmpHGqz87loc//IqHPlzCJ19t4vdjezGmVwvcna+yd/LZ8q3MWL6VGSu2sGLzbto1qcOILpmM6NqMoR2bULvWN92ceQWFTF6czctz1vLugvXszSukdaPa/HTEUYzt37JaXxSfqMyM7w1ty69fmc87CzYccQvsG3PX8cc3F7Jyy+7isvTUJLpk1eP4zpl0bZ5Bl6x6dG1ej+b10484WezdugHHd27KP6cs4wfHtic9Nfg8frx0E/9bsolbT++ull2Jrpb94ctXYc9WqN3o4PtLqcwTaLbjQYMG+YwZM2IdhhyBXbn53PLiXF79fC0jumZy/3f6HdY1aO7O9j15QXK3Yy/rtwe3jTl76d+mEWf1b1nu9WRfrtvBDf/5nPlrd9C3dQNWbtnN1rArq0ndWgxq34huzeszd812PvlqE3vzCklLSWJoxyYc37kpKzbv5vUv1rJ1dx6N6qRyep8WjO3XioHtGmlqhmpuV24+3/n7VBauz+EPZ/fi/MGH13362CfL+e1r8+nVsgFjejWnc7MMujavR5tGdaK6KsYnX23iu/+Yxp1n9+Kio9vh7pz114/ZvHMf719/QnFCJxIVX30I/x4LF78MnUbGOppqz8xmuvugkuVqeUtga7btIS0l6ZC6XWLB3dmTV8BXG3dx3XNz+Dp7JzeO7sqVJ3Q67D9yZkbDOrVoWKfWYY3U7N6iPi+PO5ZHJgXXxJ3cPYvB7RszqH0jOjStu18CtjevgOnLtjBpUTaTFm3k9//9kvTUJEb1aM7Yfi05vnMmtVI0JUO8qJuWwrM/OYafPjmLm1+Yy9pte7n25M4VTrrdnfveWcRfP/yKUT2yePDC/lWaMB3TsQl92zTk7x99zfmD2vD2/A18sXo7953XV4mbRF/LfsHPtbOVvB0BtbwlqI05exn1p8kUFDo3nNKFi49pH/PZ+Lfu2scLs1bzzvwNbNuzj51788nJzWdXbj5Fl7A1zUjjgQv7MaxT05jGeiTWbttD/dqpxd2xEp/yCgq55cW5PD9zNd8Z1Jo7z+590HnR8sPn/Gfmai4c0pY7zupJSgzmUntr3nqueGImf/pOXx78YCm1kpN445rjY/47QBLEA/0hqxec/+9YR1LtqeVNirk7v355HnvyChjUrhG/fW0BL85ewx/O7k2vVg0q7XXyCwrZsmsftVKSaFin9K5Nd+ez5Vt5atoK3pi3nn35hfRu1YBOmRnUTUsho+iWnkK99BRG9ciK+5FwLRvWjnUIUglSk5O499w+tGxYmwfeX8KGHbk8fNEA6paRlO/ZV8C4p2bxwcKNXHNS50Nqratsp/TIolNmXX750lz25hXy6CWDlLhJ1WnZH1ZNj3UUcU3JWwL679x1vD1/A7ec2o3Lh3fktS/WcftrCzjzoSn84NgO/HxUlzL/ALk723bnsWlnLtk5uWQX/cw58PGW3fsoathtUDuV9k3q0K5J3eKf2/bk8cz0lSzZuJN6aSlcMLgN3z26Ld2aa8JZiQ9mxs9HdaFlg3R+9fI8zh8/letO7kJqchIpSUZKchIpyYYBt7++gM9XbeP3Y3vxvaHtYhp3UpJxxQmduPH5LxjUrhEndW8W03gkwbTsD/NegJ3ZwfQhcsjUbZpgNu/MZdT9k2nTqDYvXDmsuMtm++487n57IU9NW0mLBulcOqw9u3LzD0jKNu3MJa/gwM9MrZQkMjPSyKyXRtPwZ2a9NDIzarE3r5Dlm3excstulm/exZqte4q7Qfu1ach3h7TljL4tYjbpqUhl+HDhRn765Cz25BWUur1WShIPXNCPMb1aVHFkpduXX8gdry/goqH6h0mq2PIpMPF0uOh56Dwq1tFUa2V1myp5SzBXPT2bt+at4/Wrjqdr8wOnopi5Ygu/fHEeizbkkGTBrPKZGWk0rZdWnJx9k5h9c79+ekqFu4D25ReyeutuHOiUmVHJ71AkdrJzclmzbQ/5BYXkFzr5BU5+YSH5BU7HzLp01OddJFjb9K42MPKXcMJNsY6mWtM1b8Lb89fz2udruX5Ul1ITN4CB7Rrz5jXHs3X3PhrWqRWV62BqpSTpj5jUSEX/zIhIOdLqQdMumqz3CGh+ggSxbfc+bn15Hj1a1OeKEjOrl5SUZDTJSNMFzCIiEh0t+8OaWbGOIm4peUsQd7z+JVt37eOec/scdDoDERGRqGrZH3auhx3rYh1JXNJf8QTw4aKNvDBrNVeO6FSpU4GIiIgclpb9g5/qOj0suuatBtmwYy+rt+5m/fZc1u/Yy4YdwdqeHy/dROdmGfzsxKNiHaKIiAg07w2WHCRv3U6LdTRxR8lbDfHKnDVc++wcIgcPp6Uk0bxBOt1b1OPW03uUu1aniIhIlalVB5p1h7W67u1wVGnyZmZtgMeB5kAhMN7d/2Jm9wLfAvYBXwE/cPdt4XNuAX4IFABXu/vbYflAYCJQG3gDuMYTad6TCPkFhfzfO4vp1rw+N43pSvP66TSvn07DOqla5FxERKqnlv1g0ZvgDvpbdUiq+pq3fOB6d+8ODAXGmVkP4F2gl7v3ARYDtwCE2y4AegJjgIfNrKj56G/A5UDn8DamKt9IdfLKnLWs3LKbn4/qwsiuzejeoj6N6tZS4iYiItVXy/6wezNsXxXrSOJOlSZv7r7O3WeF93OAL4FW7v6Ou+eHu30KtA7vnwU84+657r4MWAoMMbMWQH13nxq2tj0OjK3K91JdFBQ6f/1wKd1b1OdkLXEjIiLxouWA4KcGLRyymI02NbP2QH9gWolNlwFvhvdbAZEp+eqwrFV4v2R5wnn9i7V8vWkXV594lFraREQkfmT1hKRUzfd2GGKSvJlZBvACcK2774go/xVB1+qTRUWlPN3LKS/ttS43sxlmNiM7O/vIAq9mCgudhz5YSpesDEb3bB7rcERERCouJS1I4NTydsiqPHkzs1SCxO1Jd38xovxS4AzgooiBB6uBNhFPbw2sDctbl1J+AHcf7+6D3H1QZmZm5b2RauCt+etZsnEn40YeRZJWQxARkXjTsj+snQOJOd7wsFVp8mZBv94/gS/d/U8R5WOAm4Ez3X13xFNeBS4wszQz60AwMGG6u68DcsxsaHjMS4BXquyNVAOFhc4D7y+hY9O6nNGnZazDEREROXQt+0PudtjydawjiStV3fJ2LHAxcKKZzQlvpwEPAfWAd8OyRwDcfT7wHLAAeAsY5+4F4bGuBB4lGMTwFd9cJ5cQ3vtyAwvX5zBu5FFag1REROJTKw1aOBxVOs+bu0+h9OvV3ijnOXcCd5ZSPgPoVXnRxQ9358EPltK2cR3O6qdWNxERiVOZ3SAlPUjeep8b62jihtY2jUOTFmUzd812xo3sRIoWmRcRkXiVnBoslaWWt0Oiv/xxxt35y/tLaNWwNmf3b33wJ4iIiFRnLfvDus+hsODg+wqg5C3ufLhoI3NWbePKEZ2olaLTJyIica7lANi3EzYtiXUkcUML08eBvIJC3l2wgaemrWTK0k20alib8wap1U1ERGqAFn2Cn+vnQrNusY0lTih5q8ZWbdnN09NX8tyM1WzamUvLBun8fFQXLhjShrSU5IMfQEREpLpr2iVYaWHjfOC8WEcTF5S8VTP5BYW8v3AjT01byeQl2Rgwsmszvnt0W0Z0baZpQUREpGZJToXMrrBhfqwjiRtK3qqJtdv28Mxnq3j2s5Vs2JFLVv00rjqxM+cPbkOrhrVjHZ6IiEj0ZPWE5VNiHUXcUPIWQwWFzkeLN/Lkpyv5cNFGHBjeOZPbz2rLSd2aaRoQERFJDFk94YtnYfcWqNM41tFUe0reYmDDjr08+9kqnv1sFWu27aFpRhpXjujEBYPb0qZxnViHJyIiUrWyegY/Ny6A9sfFNpY4oOStihQWOv9buoknP13B+ws3UlDoHHdUU351endG9cgiVa1sIiKSqLLCBZM2zFfyVgFK3qIsOyeX52as4pnPVrJqyx4a163Fj47rwIVD2tK+ad1YhyciIhJ7GVlQpwlsmBfrSOKCkrcoKCx0pn69mSenreCd+RvIL3SGdmzMjaO7Mbpnlqb5EBERiWQWdJ1qxGmFKHmrRJt35vL8zNU8PX0lyzfvpmGdVC4d1p4Lh7TlqGYZsQ5PRESk+srqBTMnBstkJamRozxK3irRz56azdSvNzO4fSOuObkzp/ZqQXqqPoAiIiIHldUT8nbD1uXQpFOso6nWlLxVoptP7UadWsl0yaoX61BERETiS9GI0w3zlLwdhIY4VqJ+bRoqcRMRETkcmd3AknTdWwUoeRMREZHYS60NTY5S8lYBSt5ERESkesjqqelCKkDJm4iIiFQPWT2DAQu5ObGOpFo7pOTNzBqZ2fFm9l0zaxSWpZuZkkARERE5MkUrLWz8MrZxVHMVSrrMLNnM7gFWAx8B/wY6hJtfAG6LTngiIiKSMCJHnEqZKtpi9gfgx8DPgI6ARWx7BfhWJcclIiIiiaZBG0irr0ELB1HRed4uAX7h7v8ys5Kzzn5FkNCJiIiIHD4tk1UhFW15a0iQpJWmFqBlBEREROTIFSVv7rGOpNqqaPI2DzirjG2nArMqJxwRERFJaFk9IXcHbF8V60iqrYp2m/4eeMHMagP/ARzoZ2ZnAz8BzoxSfCIiIpJIikacbpgPDdvGNpZqqkItb+7+CvBd4GTgTYIBC48C3wcudve3oxWgiIiIJJBm3YOfGnFapgovTO/uzwHPmVkXoCmwBVjkrk5pERERqSRp9aBhOw1aKMchT67r7ovd/RN3X3ioiZuZtTGzD83sSzObb2bXhOWNzexdM1sS/mwU8ZxbzGypmS0ys9ER5QPNbG647QEzs9JeU0REROJMVi8lb+Wo6CS9E8zs2TK2PW1mj1bw9fKB6929OzAUGGdmPYBfAO+7e2fg/fAx4bYLgJ7AGODhiKlK/gZcDnQOb2MqGIOIiIhUZ1k9YfNSyNsT60iqpYq2vI0Cni9j2wvAKRU5iLuvc/dZ4f0c4EugFcFI1sfC3R4Dxob3zwKecfdcd18GLAWGmFkLoL67Tw1b/x6PeI6IiIjEs6ye4IWQvTDWkVRLFU3eMgmucSvNVqDZob6wmbUH+gPTgCx3XwdBghdxvFZA5Fjh1WFZq/B+yXIRERGJd5EjTuUAFU3eVgDDy9g2nP0TqYMyswyCFrtr3X1HebuWUubllJf2Wpeb2Qwzm5GdnX0oYYqIiEgsNO4AKbWVvJWhosnbROBmMxsXJl6YWYaZ/RS4iWDakAoxs1SCxO1Jd38xLN4QdoUS/twYlq8G2kQ8vTWwNixvXUr5Adx9vLsPcvdBmZmZFQ1TREREYiUpOZgyRNOFlKqiydvdwBPAg8B2M9sBbAceIrhG7e6KHCQcEfpP4Et3/1PEpleBS8P7lxIsdl9UfoGZpZlZB4KBCdPDrtUcMxsaHvOSiOeIiIhIvNMyWWWq0Dxv7l4I/MjM7gVGAk2AzcAH7r74EF7vWOBiYK6ZzQnLfgn8kWAOuR8CK4Hzwtedb2bPAQsIRqqOc/eC8HlXErQI1iaYOPjNQ4hDREREqrOsXjD737BzI9TLinU01UqFJ+kFcPdFwKLDfTF3n0Lp16sBnFTGc+4E7iylfAbQ63BjERERkWosq2fwc8M8JW8llJm8hXOsfeXuueH9crn7gkqNTERERBJXZPJ2VKntOwmrvJa3eQQT6U4P75fV6WzhtuQytouIiIgcmjqNoV5LjTgtRXnJ20iCa82K7ouIiIhUnea9YL1GnJZUZvLm7h8BmFkawVQc0919SVUFJiIiIgkuqyd89QHk74OUWrGOpto46FQh7p5LMI9by+iHIyIiIhLK6gWF+bDpsMdK1kgVnedtLtAlmoGIiIiI7KdomSx1ne6nolOFXAdMNLN1wFvunh/FmERERESgyVGQnKaVFkqoaPL2MlCHYBUDN7OtlBh96u6HvDi9iIiISJmSU7RMVikqmrz9lbKnChERERGJjqxesOTtWEdRrVR0eazfRjkOERERkQM17wVznoCcDVppIVTugAUzO83MXjOzuWb2rpldGS4ELyIiIhJ9RYMWNsyNbRzVSJnJm5mdB7wOdAbmA/WAh4C7qyY0ERERSXjFy2RppYUi5bW83QQ8DXR39wvcfShwC3C1mR3SgvYiIiIih6VOY6jfStOFRCgveesK/MvdIwcq/AOoBXSIalQiIiIiRbJ6quUtQnnJWwawo0RZ0eN60QlHREREpISsXsEqC/m5sY6kWjhY9+cwM2sa8TiJYMqQY82seeSO7v5GZQcnIiIiQvNwmazsRdCiT6yjibmDJW9/KqP8LyUeO5B85OGIiIiIlFA84nS+kjfKT950XZuIiIjEXuNOkJKulRZCZSZv7r6iKgMRERERKVVyCmR2U/IWKneSXhEREZFqoXmvYLoQ12qdSt5ERESk+svqDbs3wc4NsY4k5pS8iYiISPVXvNKCuk6VvImIiEj11zwccaqVFiqWvJlZk4Ns71054YiIiIiUonYjqN9aLW9UvOXtPTNrUNoGMzsamFRpEYmIiIiURstkARVP3nYDb5tZRmShmY0A3gVerdywREREREpo3gs2LU74ZbIqmrydSrCCwhtmVhvAzE4H3gQed/cfRCk+ERERkUBWz2+WyUpgFUre3H0HMBpoALxmZpcCLwF/dvefRTE+ERERkUBWeIl9gl/3VuHRpu6+BTgJaAFMAG5z91sO5cXMbIKZbTSzeRFl/czsUzObY2YzzGxIxLZbzGypmS0ys9ER5QPNbG647QEzs0OJQ0REROJQk3CZrAQfcVrm8lhm9lwZmzYDW4H+Efu4u59fgdebCDwEPB5Rdg/wO3d/08xOCx+PMLMewAVAT6AlwaCJLu5eAPwNuBz4FHgDGEPQhSsiIiI1VVIyNOue8C1v5S1Mn1lGeQEwt5ztZXL3yWbWvmQxUD+83wBYG94/C3jG3XOBZWa2FBhiZsuB+u4+FcDMHgfGouRNRESk5svqBYveCJbJStCOt/IWph9ZRTFcSzCS9T6CbtxhYXkrgpa1IqvDsrzwfsnyUpnZ5QStdLRt27bSghYREZEYyOoFs/8dLJNVr3mso4mJ6rDCwpXAde7eBrgO+GdYXlo67eWUl8rdx7v7IHcflJl5yI2FIiIiUp1opYUKr7AwwcyeLWPb02b26BHEcCnwYnj/P0DRgIXVQJuI/VoTdKmuDu+XLBcREZGarniN07mxjSOGKtryNgp4voxtLwCnHEEMa4ETwvsnAkvC+68CF5hZmpl1ADoD0919HZBjZkPDUaaXAK8cweuLiIhIvCheJitxV1oob8BCpExgSxnbtgLNKnIQM3saGAE0NbPVwG3Aj4G/mFkKsJfw+jR3nx+OZl0A5APjwpGmEHS1TgRqEwxU0GAFERGRRNG8N6z7PNZRxExFk7cVwHDg/VK2DWf/AQRlcvcLy9g0sIz97wTuLKV8BtCrIq8pIiIiNUzL/rD4LcjNgbR6sY6mylW023QicLOZjSta39TMMszsp8BNwJFc8yYiIiJScS37Aw7rvoh1JDFR0eTtbuAJ4EFgu5ntALYTTLj7WLhdREREJPpa9g9+rp0V2zhipELdpu5eCPwonIttJNCYYKWFD9x9cRTjExEREdlfRiY0aANrZ8c6kpio6DVvALj7QmBhlGIRERERqZiW/ZS8HYyZNQR+AhxH0PK2BfgfMN7dt0UjOBEREZFStewPX74Ge7YG04ckkIpO0tuJYD3T24G6wMrw5+3AF+F2ERERkapRfN3bnJiGEQsVHbBwP7AN6OjuJ7r7he5+ItApLP9TdMITERERKUVx8pZ4XacVTd5GAL9x9zWRheHj3xEMYhARERGpGrUbQaMOSt7K4UByOccoc2F4ERERkaho2V/dpuX4ELjDzNpFFoaPb6f0lRdEREREoqdlf9i+EnZtinUkVaqiydu1QBqwxMw+NbNXzGwqwSLytYCfRyk+ERERkdK1GhD8TLCu0wolb+6+HOgGXA3MB1IJFoz/GdA93C4iIiJSdZr3ASzhkrcKz/Pm7vuAR8KbiIiISGyl14emnWFNYi2TVdF53grMbEgZ2waaWUHlhiUiIiJSAS0HJFzLW0WvebNytqUC+ZUQi4iIiMihadkfdq6HHetiHUmVKbPb1MzaAu0jivqbWXqJ3dKBS4FllR+aiIiIyEFETtZbv0VsY6ki5V3z9gPgNoI53Bz4Wxn77QF+VMlxiYiIiBxc895gybB2FnQ7LdbRVInykreHgecJuky/AC4Kf0baB6x099zohCciIiJSjlp1oFn3hLrurczkzd2zgWwAM+sArAtHnIqIiIhUHy37waI3wR2svMv0a4aKzvO2oihxM7M6ZnaVmf3VzH5dctUFERERkSrVsj/s3gzbV8U6kipR3oCF/wO+5e5dIsrqAZ8BnYGtQAPgejMb4u6Lox2siIiIyAGKBi2smQUN28Y2lipQXsvbSOCJEmU3AF2AH7t7U6AlsBz4dVSiExERETmYrF6QlJow172Vl7y1B2aWKPs2sMDdJ0DxdXH/BxwblehEREREDiYlDbJ6Knkj6FLdW/TAzBoD3YEPSuy3HGhe6ZGJiIiIVFTL/rB2TjBooYYrL3lbDIyIeHxG+PPtEvs1A7ZUYkwiIiIih6Zlf8jdDlu+jnUkUVfePG8PAf8wswbABuBqgpUU3imx3ynAvOiEJyIiIlIBrQYEP9fOhiadYhtLlJXZ8ubuE4HfAOcAtwCLgLPdPa9oHzPLBM4CXolumCIiIiLlyOwGKekJcd1bufO8uftd7t7a3TPcfbi7zy2xPdvdm7t7WUtn7cfMJpjZRjObV6L8KjNbZGbzzeyeiPJbzGxpuG10RPlAM5sbbnvALAFm5BMREZGyJacGS2UlevIWBROBMZEFZjaSoPWuj7v3BO4Ly3sAFwA9w+c8bGbJ4dP+BlxOMN9c55LHFBERkQRUNGihsCDWkURVlSZv7j6ZAwc3XAn8sWh9VHffGJafBTzj7rnuvgxYCgwxsxZAfXef6u4OPA6MrZI3ICIiItVX6yGQtws21OxL8au65a00XYDjzWyamX1kZoPD8lZA5DoXq8OyVuH9kuUiIiKSyNodE/xc8Uls44iy6pC8pQCNgKHAjcBz4TVspV3H5uWUl8rMLjezGWY2Izs7uzLiFRERkeqoQWto2A6WT4l1JFFVHZK31cCLHpgOFAJNw/I2Efu1BtaG5a1LKS+Vu49390HuPigzM7PSgxcREZFqpN2xQctbDZ6stzokby8DJwKYWRegFrAJeBW4wMzSzKwDwcCE6e6+Dsgxs6FhC90laKoSERERAWg3DPZsgexFsY4kasqbpLfSmdnTBKs2NDWz1cBtwARgQjh9yD7g0nAgwnwzew5YAOQD49y9aPjIlQQjV2sDb4Y3ERERSXTtw+XWV3wMzbrFNpYoMa/BzYolDRo0yGfMmBHrMERERCRa3OFP3YPu03P/GetojoiZzXT3QSXLq0O3qYiIiEjlMAu6Tld8XGOve1PyJiIiIjVLu2GQsw62Lot1JFGh5E1ERERqlnZF173VzPnelLyJiIhIzZLZDeo0UfImIiIiEhfMoO0xwXVvNZCSNxEREal52h0LW5fD9jWxjqTSKXkTERGRmqfdsOBnDew6VfImIiIiNU/z3pBWv0Z2nSp5ExERkZonKRnaDlXLm4iIiEjcaDcMNi2CndmxjqRSKXkTERGRmqlovreVNav1TcmbiIiI1Ewt+kFK7RrXdarkTURERGqmlFrQZkiNG7Sg5E1ERERqrnbHwvp5sGdbrCOpNEreREREpOZqNwxwWPlprCOpNEreREREpOZqPQiSUmtU16mSNxEREam5UmtDq4E1atCCkjcRERGp2dofC+vmQO7OWEdSKZS8iYiISM3WbhgU5sOqmnHdm5I3ERERqdnaDoOUdFjyXqwjqRRK3kRERKRmq1UHOgyHxW+Ce6yjOWJK3kRERKTm6zIati6HTUtiHckRU/ImIiIiNV/n0cHPxW/FNo5KoORNREREar6GbSCrFyx+O9aRHDElbyIiIpIYuoyGlVNhz9ZYR3JElLyJiIhIYugyBrwAvvog1pEcESVvIiIikhhaDYQ6TeK+61TJm4iIiCSGpGTofAoseQcKC2IdzWGr0uTNzCaY2UYzm1fKthvMzM2saUTZLWa21MwWmdnoiPKBZjY33PaAmVlVvQcRERGJY11GB9e8rf4s1pEctqpueZsIjClZaGZtgFHAyoiyHsAFQM/wOQ+bWXK4+W/A5UDn8HbAMUVEREQO0OlESEqJ6ylDqjR5c/fJwJZSNt0P3ARETnt8FvCMu+e6+zJgKTDEzFoA9d19qrs78DgwNrqRi4iISI2Q3gDaHhPX173F/Jo3MzsTWOPun5fY1ApYFfF4dVjWKrxfslxERETk4LqMgY0LYOuKWEdyWGKavJlZHeBXwG9K21xKmZdTXtZrXG5mM8xsRnZ29uEFKiIiIjVHl/BqqyXvxDaOwxTrlrdOQAfgczNbDrQGZplZc4IWtTYR+7YG1oblrUspL5W7j3f3Qe4+KDMzs5LDFxERkbjT9Cho3Clur3uLafLm7nPdvZm7t3f39gSJ2QB3Xw+8ClxgZmlm1oFgYMJ0d18H5JjZ0HCU6SXAK7F6DyIiIhKHuoyBZZMhd2esIzlkVT1VyNPAVKCrma02sx+Wta+7zweeAxYAbwHj3L1oUpYrgUcJBjF8BbwZ1cBFRESkZukyGgr2wbKPYh3JIUupyhdz9wsPsr19icd3AneWst8MoFelBiciIiKJo+0xkFY/GHXa7fRYR3NIYn3Nm4iIiEjVS6kVzPm2+G3wMsc9VktK3kRERCQxdRkDO9fDupKzlVVvSt5EREQkMXUeBRgsiq9L55W8iYiISGKq2xTaHwfzXoirrlMlbyIiIpK4ep8Hm5fA2tmxjqTClLyJiIhI4upxFiTXgi+ei3UkFabkTURERBJX7YbBwIV5z0NBfqyjqRAlbyIiIpLY+pwPu7Lh60mxjqRClLyJiIhIYus8CtIbwhfPxjqSClHyJiIiIoktJQ16joWFr8fFWqdK3kRERET6nA95u2HRG7GO5KCUvImIiIi0GQoN2sZF16mSNxEREZGkJOhzHnz1AezcGOtoyqXkTURERASg93fAC4MVF6oxJW8iIiIiAM26QfM+1b7rVMmbiIiISJE+5wdLZW1aEutIyqTkTURERKRIr2+DJVXr5bKUvImIiIgUqd8COpwQdJ26xzqaUil5ExEREYnU53zYtgJWTY91JKVS8iYiIiISqfsZkFK72g5cUPImIiIiEimtHnQ7HeY+D/t2xTqaAyh5ExERESlp8I8gdzvM/U+sIzmAkjcRERGRktoOhazeMP0f1W7ggpI3ERERkZLMYMiPYMM8WPlprKPZj5I3ERERkdL0Pg/SG8D08bGOZD9K3kRERERKU6su9L8YvnwVctbHOppiSt5EREREyjLoMijMh5kTYx1JMSVvIiIiImVp0gmOGgUzJkD+vlhHA1Rx8mZmE8xso5nNiyi718wWmtkXZvaSmTWM2HaLmS01s0VmNjqifKCZzQ23PWBmVpXvQ0RERBLIkMth5wZY+FqsIwGqvuVtIjCmRNm7QC937wMsBm4BMLMewAVAz/A5D5tZcvicvwGXA53DW8ljioiIiFSOo06GRu1h+qOxjgSo4uTN3ScDW0qUvePu+eHDT4HW4f2zgGfcPdfdlwFLgSFm1gKo7+5T3d2Bx4GxVfIGREREJPEkJQWT9q78BNbPjXU01e6at8uAN8P7rYBVEdtWh2Wtwvsly0tlZpeb2Qwzm5GdnV3J4YqIiEhC6HdRsN7p9H/EOpLqk7yZ2a+AfODJoqJSdvNyykvl7uPdfZC7D8rMzDzyQEVERCTx1GkMvc8NlsvaszWmoVSL5M3MLgXOAC4Ku0IhaFFrE7Fba2BtWN66lHIRERGR6BnyY8jbDXOeimkYMU/ezGwMcDNwprvvjtj0KnCBmaWZWQeCgQnT3X0dkGNmQ8NRppcAr1R54CIiIpJYWvSFNkODrtPCwpiFUdVThTwNTAW6mtlqM/sh8BBQD3jXzOaY2SMA7j4feA5YALwFjHP3gvBQVwKPEgxi+IpvrpMTERERiZ4hP4Z9O2Hb8piFYN/0UtZ8gwYN8hkzZsQ6DBEREYlXBXnghZCSFvWXMrOZ7j6oZHlK1F9ZREREpKZITo11BLG/5k1EREREKk7Jm4iIiEgcUfImIiIiEkeUvImIiIjEESVvIiIiInFEyZuIiIhIHFHyJiIiIhJHlLyJiIiIxBElbyIiIiJxRMmbiIiISBxJqLVNzSwbWBHll2kKbIryayQy1W/1ovMRXarf+KDzFF2JXL/t3D2zZGFCJW9VwcxmlLaIrFQO1W/1ovMRXarf+KDzFF2q3wOp21REREQkjih5ExEREYkjSt4q3/hYB1DDqX6rF52P6FL9xgedp+hS/Zaga95ERERE4oha3kRERETiiJI3ERERkTii5E1EREQkjih5qybMLCPWMdRkZjbazK6NdRwS0Oc9uvR5jw/6HkRXTf4eKHmrBszsdOBlMzsh1rHURGZ2CvAH4PNYxyL6vEebPu/xQd+D6Krp3wONNo0xM+sLvAO8BDQH7nf3j2IbVc1hZscDHwK93f1LM2sIpAOb3T0vpsElIH3eo0uf9/ig70F0JcL3ICXWAQjLgJuB/wLnADeaGfoiV5rFQA5wvJktBV4EdgJpZvYX4E3XfzBVSZ/36NLnPT7oexBdNf57oJa3GDIzc3c3s2R3LzCzxsC5wFnAve4+ycxaARvcPT+20cavsA5nAo2Bn7n7eDP7OXAy8B133xnTABOEPu9VQ5/36k3fg6pR078HSt5iILzW4WxgDfChu0+K2NaU4D+xE4EtBE3qF7v7rhiEGpfMrA9Q4O7zI8paEHxh/xJR9gZwg7sviEGYCUOf9+jS5z0+6HsQXYn2PdCAhSpmZkOAPwGTgHXA82b2naLt7r7J3ccDRvBlvl1f4Iozs1OBOcCVZjagqNzd15X4Ap8PtACyqzzIBKLPe3Tp8x4f9D2IrkT8Huiat6qXBUxz9ycAzOwr4C9mVujuz4dlo4FhwMnuPi92ocYXM6sNDAZ+CTQAvhNeRzIrYp9k4ELgV8C57h73X+JqTp/3KNHnPa7oexAlifo9UPJW9VYCeWbW2t1Xu/u7ZnYN8JSZrXP3j4FpwHB3XxbbUOOLu+8xs8fdfbmZNQN+A5xnZknuPiPcp8DMtgBj3X1RTANODPq8R4k+73FF34MoSdTvga55q2JmlgL8i2AkzNUEffRuZlcDqe7+fzENsAYxsyzg1wSjjP5CcKHqPHefHdPAEog+71VHn/fqS9+DqpMo3wNd81aFwv8E8oEfAZ2BB4EO4eZ6QLtYxVbThCO5NgB3APnAUwTXnOyLaWAJRJ/3qqPPe/VjZhb+1PcgCorqt0RZwnwPlLxFkZmlRz5290Izq+XuucDpBB+u35jZSwT98f+IQZhxq2T9hmVJEDSThz83ALuBnsAJkSORpHKZWRszq1P0WJ/3ylWyfsMyfd6rGTMbYGZZHnZr6XtQuUrWb1iWcN8DJW9RYmYjgA/NrFuJ/8D2mdko4EaC5vM/AH8FznD3ubGKN96UU7+FZjbSzO4Jy+oBGcDoeB8aXp2Z2WnAAwQXDBeVmT7vlaOc+tXnvRoxs28RJGPdIsqKfu+fgr4HR6Sc+k2874G76xaFG8Fw7xzgHqAr31xf2BOYDpwf6xjj+VaB+j03Yt+UWMdbk2/AacBsYFgp2/oCn+nzHvX61ec99uepDTAXODZ8bBG/l3rp936V1G/CfA80YCFKzKwTwbBkgNrAFUAq0A/Y6e6fhv856wQchgrWb5K7F8YoxIRgweSizwCr3f37Fqwh+G2Cc/EuUAC0dvcp+rwfukOsX33eY8jMOgMPuvsYM2sJXAdkAo8T9HLp9/4ROIT6TYjvgbpNK5kFkoC9wFbgSmApwQLEk4AF+gIfvkOs3xr/Ba4GdhJ0/2wwsz8CbxMk0H2Bj4FaStyOyKHUrz7vsbWU4DydTDCydAXBxLG/AJL0e/+IVbR+E+J7oHneKomZdSS4EHWju+8F1oQX1NcH3gHGAfMIkg70BT40qt/qJWz5LABWuPtLZraPYKHtp939z+E+WwkuyP6dzsehUf3Gh4jfS5vcfbeZLSM4J1+7+0PhPtuBn5nZh+6eF8Nw447qt2xK3iqBmZ0N3ApsB2aa2UJ3/ycwH7gfGAD8ABgN3GVmV7l7jRu6HC2q3+qlxPmYbWafufszZrbE3RdHtC7kEiQgcghUv/GhlPM0BbiLYCqQAWY20t0/JGg93RK7SOOT6rd8uubtCJlZUcvPz4GvCZY3+S7wFvAq8D5wh7s/F+7fwt3XxSjcuKP6rV7KOB8XESy0/VDEfhcC1wPfc/eFsYg1Hql+40MZ5+l7BL+TngR+S9Ar0AToAVzq7p/HJNg4pPo9OLW8Hbl8YA2w1t3Xm9nbwCbgGmADMMDd88ws1d3zlFgcMtVv9VLW+RhnZlvd/UkzO4mgJfT7SiwOmeo3PpR2njYDVxGcr18DTQlGvy9191UxizQ+qX4PQgMWjpC77wYWAxPMrJ677yIY1v88cCwUz8eUMH3xlUn1W72Ucz5eJhiuD8HUFd9zLa59yFS/8aGM8zQLeAE4Ptxno7t/mIiJxZFS/R6ckrcjYFa8PMetBKNeHgw/aDnA/4DBQGNdTHx4VL/VSwXOxxAza+nuO9x9Y6zijFeq3/hQgfM0iGAKCzkMqt+KUfJ2BIqSBg+W5LgfyAbeNLMuwIlAHXRB8WFT/VYvFTwfGihymFS/8UG/l6JL9VsxGrBwGCxY/LYg8r6ZtSeY8f8qoCPQFrjW3efELNA4pfqtXnQ+okv1Gx90nqJL9XtolLxVkJmdCZzo7teGjyM/aCOAW4CrwqH8yQRLc+TGKNy4o/qtXnQ+okv1Gx90nqJL9Xv4NNq0AsxsCMEs5xlm1szdvxv+V5AKpAF/BO5x98VQ3Nyb8M26FaX6rV50PqJL9RsfdJ6iS/V7ZNTyVgFmNgao7cFM57OBhe5+YcT2Ru6+1RJkTbXKpvqtXnQ+okv1Gx90nqJL9XtklLxVkJk192C+GQNmEMwtc364TRPDHiHVb/Wi8xFdqt/4oPMUXarfw6fkrQxhf3tngv8MHgjLarn7vrDvfTrBvDNvA8OBmzxYc1MqQPVbveh8RJfqNz7oPEWX6rcSubtuJW7AacAC4KcES3M8HLEtNeL+DoJZn3vHOuZ4uql+q9dN50P1q5vOk+o3vm4xD6C63QiGIn8CnBQ+bkAwMWBXwpbKsHwEsAzoGeuY4+mm+q1eN50P1a9uOk+q3/i7abTpgXKB37v7+2ZWC9gN7OXAmfxrA6PcfWksgoxjqt/qRecjulS/8UHnKbpUv5VMKyyEzKxtOER5q7u/AeDu+zxYM/NroDDcb2i47U19wCpO9Vu96HxEl+o3Pug8RZfqN3qUvAFmdjrwBvAw8G8z6xaW1wp3aQDUMbMLgSfMrEVsIo1Pqt/qRecjulS/8UHnKbpUv9GV0N2m4fDk1gSTAf4M+BL4HvCBmY1y9/nhrmuAXwK1gLNcw5crRPVbveh8RJfqNz7oPEWX6rdqJHTy5u5uZmuBqcASYKO7/5+Z5QHvmNmJ7r4IWA+cC4x294UxDDmuqH6rF52P6FL9xgedp+hS/VaNhJ3nzcyOAhoRDlkGZrr7PRHbbwJ6Aj8G+gLr3X1VLGKNR6rf6kXnI7pUv/FB5ym6VL9VJyFb3szsDOAPwFZgLvAk8IAFi+LeFe72HPArd98HfBabSOOT6rd60fmILtVvfNB5ii7Vb9VKuOTNzIYB9wEXuvtsMxsPDAGGAZ+Gszw/AxwH9Dezxu6+JXYRxxfVb/Wi8xFdqt/4oPMUXarfqpdw3abhh6yLu08MH2cCE939dDPrCNxKMP/MEOAH7j43ZsHGIdVv9aLzEV2q3/ig8xRdqt+ql4jJWzJQ1913hPdbAK8Bp7n7OjNrRzAKpq67b49lrPFI9Vu96HxEl+o3Pug8RZfqt+ol3Dxv7l7g7jvChwZsA7aEH7DvEQxdTtUH7PCofqsXnY/oUv3GB52n6FL9Vr2Ea3krjZlNBNYBpwDfV5Nu5VL9Vi86H9Gl+o0POk/RpfqNroRO3sLJBFMJJhFMJVg0d0lso6o5VL/Vi85HdKl+44POU3SpfqtGQidvRczs+8Bn/s3Mz1KJVL/Vi85HdKl+44POU3SpfqNLyRvBfwquioga1W/1ovMRXarf+KDzFF2q3+hS8iYiIiISRxJutKmIiIhIPFPyJiIiIhJHlLyJiIiIxBElbyKS0Mzst2bm4a3QzLaa2WdmdqeZNT+M491kZiMqP1IRkYCSNxER2A4cQ7CQ9gXAi8DFwFwzG3iIx7oJGFGp0YmIREiJdQAiItVAvrt/GvH4bTP7GzAZeNbMurp7QYxiExHZj1reRERK4e7bCFrROgGjAMzsj2Y218x2mtlqM3sysmvVzJYDTYDbIrpiR4TbkszsF2a21MxyzWyxmV1axW9LRGoAJW8iImX7EMgHhoaPmwF/AE4HrgU6Ah+YWXK4/WyCLth/EnTDHgPMCrc9CNwKjA+f/xIwwczOiPq7EJEaRd2mIiJlcPdcM9sEZIWPLyvaFiZsU4HVwLHAZHefbWb5wOrIblgzOwq4EviBuz8WFr9nZi2A24DXq+QNiUiNoJY3EZHyWfEds1PN7BMz207QIrc63NTlIMc4CSgEXjKzlKIb8D7QL6LlTkTkoNTyJiJSBjNLJ7iGbYOZDQZeJeju/COwEXDgUyD9IIdqCiQTdKmWpgXfJIIiIuVS8iYiUraRBL8npxJcz5YNnF+04LaZtavgcbYQtNQdS9ACV9LGIw9VRBKFkjcRkVKYWUPgbmAp8B4wBsgrStxCF5Xy1H0c2BL3AUHLWwN3f7fyoxWRRKLkTUQEUsysaERpPWAgwQCDOsAYdy8ws3eBa83sz8BrBBP6fq+UYy0ETjezt4CdwCJ3X2RmjwDPmNk9wAyCBK8n0MXdfxTF9yYiNYySNxERaEDQNerADoLWtieAB919PYC7v2FmNwNXAT8O9z8DWFziWDcCfwX+S5D8jQQmAePCfX8M3B6+zgKCaUVERCrM9u8BEBEREZHqTFOFiIiIiMQRJW8iIiIicUTJm4iIiEgcUfImIiIiEkeUvImIiIjEESVvIiIiInFEyZuIiIhIHFHyJiIiIhJHlLyJiIiIxJH/B2fZKPLtArfVAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 720x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# plotting Upcoming Open price on date index\n",
    "fig,ax=plt.subplots(figsize=(10,5))\n",
    "ax.plot(df_merge.loc['2021-04-01':,'open'],label='Current Open Price')\n",
    "ax.plot(upcoming_prediction.loc['2021-04-01':,'open'],label='Upcoming Open Price')\n",
    "plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)\n",
    "ax.set_xlabel('Date',size=15)\n",
    "ax.set_ylabel('Stock Price',size=15)\n",
    "ax.set_title('Upcoming Open price prediction',size=15)\n",
    "ax.legend()\n",
    "fig.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "id": "4d17464d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAm8AAAF6CAYAAABPxb89AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8QVMy6AAAACXBIWXMAAAsTAAALEwEAmpwYAABeC0lEQVR4nO3dd5xU1fnH8c+zyy69s8DSe5GlFylKsWHsGI0aY401xpJfLDEaNZYklthLNLaQ2LuJHRUVBREQpBfpvbddYNvz++PexWHZXRbY2dnZ+b5fr3nNzLllnjn3Djx77j3nmLsjIiIiIvEhKdYBiIiIiEjpKXkTERERiSNK3kRERETiiJI3ERERkTii5E1EREQkjih5ExEREYkjSt5EoszMxprZ68Usm2Rmz5dzSAfNzNzMfhvrOAqY2W1mtj7WcZQFMxse1m9GrGOJFjO7z8wWR7w/P/zOtfZjH9eb2fAiyivUuSkSDUreRORADAJei3UQldQUgvr9MdaBlKP3CL5z1n5scz0wvIhynZtS6VWJdQAiEn/cfUKsY6hszMyAqu6+FajQ9Wtm1d19R1ntz93XAevKaF8Vuu5EyoJa3kQqkIJLrGZ2iZktNrMdZvaemTUvtF51M7vHzJaY2S4zW2Rmf41YnhxeSlwaLp9pZr8stI/nw8u2x5vZLDPLCj+rgZl1MLPPzSwzXKdHoW33uDQVEfcvzWyBmW01sw/MrEWh7VqF5TvCmM8PtxtbiroZZWYTw203mNn7Zta6hPXbmtnbYSzbzOy/Ztah0Dq/Dutmh5mtN7MvzKxbxPJqYT0vC+txmpkdt48424T180sz+3f42WvN7NZC690WfuZhZvYdsBM4vajLpuHxvNHM5oVxLC98ud3MTg6P1U4zWx3GnbKPWPd5vkV8n7PNbLSZbQb+Gy5rYGZPmtma8HO/MbNDC31GPTN7MTyXVpnZTUXEsddl05LOcQsuuTYEbg23cwsvoRY+N8Oy35rZ/HA/C8zsd8Uci95mNiH8LXxvZoeXVH8isaKWN5GKZxDQGfg/oBpwN/A20B92t9C8E653BzAZaA5E/kdzO8FlpT8D3wE/B14wM3f3lyLWaxWuezNQA3gEeApoA/wTuAf4K/CymXXzkufTOxRoBvweqA48FO7ruIi43wXqARcSJCt/AtLYxyVCMzsHGA28HH5nA44It11SxPpVgU+BHOBiIDesiy/MrLu7bzSzocA/gFuA8UAdgjqtG7Gr14EBwK1hjL8A3jWzfu4+taSYgXuB/wGnAUMJEo317v5YxDo1gH8R1PM8YCWQXsS+ngTODdf7AmgQ7rfg+/4CeClc749Ae4LjlgRcu484SzzfItwHvAmcDuSFdTyG4HheB6wFLgfGmFlHd18dbvccweXNa4DVYTztCY5JkUpxjo8CPic4Pk+HZbOK2dfFBOf1/cBHwAjg72ZW1d3/FrFqwbF4IIzzVuAtM2vl7vtzOVck+txdDz30iOIDGAu8XsyyScDzhdbNAVpHlA0BHDg2fD8yfH9SMftsAGQCtxYqfx+YG/H+eYL/QNtHlN0T7vvciLLjwrKuEWUO/LZQ3FuA+hFl14TrVQ/fHx++HxCxTvPw+44tof6SgBXAmyWscxuwPuL9ZeF3axdR1gLIBm4M318LTC5hn0eG8Q4rVP4l8FoJ27UJt/u4UPk/w++RFBGzAycXWm94WJ4Rvu8Svr+qmM8zggT2uULlFwI7gIb7ODf3db4VfJ+3Cm3767A+O0aUVSFIcu8N33cLtz0jYp1awEZgcUTZ+eF6tUpzjofrrAduK6J897kZce4UrpvHw/O1WqFjcUTEOr0i60EPPSrSQ5dNRSqeKe6+uzXJ3b8maNUYEBYdAWx093eL2T6DoBWh8E3brwCdzKxxRNlid49s9VoQPn9WRNkel26L8J27b4p4X9ASUrBdf2C1u08sWMHdVxC0qpSkM0GL3nP7WC/SAIJ6XBjxWcuBr4HDwqKpQG8ze8DMhppZaqF9HEXQAvO1mVUpeBC06PUrRQxvFXr/Zvg9Ii8lO/DBPvYzInx+vpjlnQhaUF8tFOdnBC1p++q1uq/zrcB7hd4fRXDsFkV8JgQtgwX1U9B6t/tcdfftwCf7iGlf53hptSCo86J+C3WA7hFlOQTJbIGC83ePS/8iFYEum4pEXy6QXMyyZPa+fLS2iPXW8tPltIbAqhI+r2C9NYXKC97Xj/iMzYXWyS6ivKCsWgmfWdK+CrZrStE3pa8Dapew34bhc0nfubB09v7+hGWtAdx9jJldAFwFXA1sN7P/ANe5eybQKIw5p4j95JUihsLHseB9OrA0fL3J3bMpWUMg04OODEVpFD6/X8zylvvY/77OtwKF67MRMJCi66fgD4KmwDbfu3NDUZ8ZaV/neGnt67fQIKJsq7vnF7xx9+zg6u0+z3uRcqfkTST61hFceipKOnv/R9a4iPUa89N/Zhso+r6oAgXrNQ7XLdAkfN5YwrbRtJrgHrXC0gjufytOwXco6TsXtorgkl1hTYj4/u7+L+BfZpYGnEpwv9NW4A/heiuAU/bjcyMVPo4F7yOTkpLuISywAahpZnWKSeAKvs8lwPdFLF+0j/3v63wrUDjWjQSX/S8vYvtd4fNqoLbt3Tu1qM+MtK9zvLQifwuRYv1bEDkoumwqEn1fAX1t7x6jhxL8J/JVofX7mFmriPWGEPznU3C58VOggZmdUMznzSAYL+v0QuW/AOZ5MCxDLHwHNDWz3Zfjwjrpu4/t5hIkUeftx2d9S1DnbQt91mBgXOGV3X2duz9JcCwOCYs/JWg52u7ukwo/ShHDqELvTyVIJpbvx/eAny5hn1vM8oL6aVNUnO6+oZjtCuzrfCvOp0AHYGkRnzk9XOe78PmkiP3XAo4uxb5LOschaNndV6vYcoJOIEX9FrYC0/faQiQOqOVNJPpGE/Tk+9LM7iS4ubwrQW+2bwh6wEVaC/zPzG7jp95/U9z9w3D5J+E2L5rZ7QSDuqYDQ939Ug96Uj4I3GxmuQStI6cSdDw4K2rfct/eB6YR3Jt1I8HN9LcSXMLKL24jd883s+sJesu+QNCr0gnui3qpmETqeeAG4AMzu4XgMudtBDe5PwlgZn8muGw2NizvDQwjaHWDn+r5EzO7G5hJcJ9UL4Ib3W/cx/ftZmZPAm8Q9Db9NXB15KW50nD3uWb2FEEPycYEHSbqAae5+5lh/fwe+LeZ1SG4hy4baEfQanial9xbcl/nW3FGE3QMGWtm9wELCS53DiC4t/EBd59pZu8CT4SxrSLombqv3pslnuPhOnOA483sQ2A7QWecbZE7CevmNuBJM9sQ7ncYQWvhH929pBZfkQpLyZtIlLn79nBYir8AfyNIGNYQ3DR9UxH/mY8nGILhQYJLimMJLokV7M/NbBTBEArXhOusBF6M2MctBPfSXU7QurcA+JW7v1y23670wrhPJkieniOog7sIhrwo8T9zd3/RzHYCNxEMD5FJMJBtka2I7r7LzI4iGB7iGYIemWOBU9294FLZd8DvgDMJ7rlbQpDgPRQR76kEQ29cQ9ApYCNBR4dHSvGVrwdOIEjedhIcr0dLsV1RfhPGdxFBcrmWiJv+3f0VM9saxnohQbK6kGCokn3dU1fi+VYcd99pZiMIhpr5M8F5tpagxS6yo8H5wBPh/rcDjxHU/WkUo5Tn+HXhvt4j6KAzgj07HBTs65/hsCbXENzbuBz4vbs/sK/vKFJRmXtpbrkQkfJgwWC169292P/YKhMzq0uQZDzq7rfua/14YGZtCO4zO9Hd/xfjcEqUaOebSGWhljcRKTdmdhnBJdL5BK0p/wdUBZ6NZVwiIvFEyZuIlKddBPeitSK4b20icFTkOGMiIlIyXTYVERERiSMaKkREREQkjih5ExEREYkjCXXPW6NGjbxNmzaxDkNERERknyZPnrze3feamSahkrc2bdowaVJpBkYXERERiS0zK7Izly6bioiIiMQRJW8iIiIicUTJm4iIiEgcSah73oqSk5PD8uXL2blT8xNL6VWrVo0WLVqQkpIS61BERCTBJHzytnz5cmrXrk2bNm0ws1iHI3HA3dmwYQPLly+nbdu2sQ5HREQSTMJfNt25cycNGzZU4ialZmY0bNhQrbUiIhITCZ+8AUrcZL/pnBERkVhR8lYBrF69mjPPPJP27dtzyCGHcNxxxzFv3rxyjWHs2LF88803+7VNmzZtWL9+fVTiue2222jevDm9evUiIyODd999t8j1/vGPfzB69OioxCAiIlIRJfw9b7Hm7owaNYrzzjuPl19+GYCpU6eyZs0aOnXqVKp95OXlkZycXOz70hg7diy1atVi8ODB+7VdNP3ud7/j2muvZfbs2Rx++OGsXbuWpKSf/t7Izc3lsssui2GEIiIi5U8tbzH2+eefk5KSskcS0qtXLw4//HDGjh3LCSecsLv8t7/9Lc8//zwQtHrdfvvtHHbYYbz22mt7vf/4448ZNGgQffr04fTTT2f79u27t7v11lvp06cP3bt3Z86cOSxevJh//OMfPPDAA/Tq1Yuvvvpqjxi3b9/OBRdcQPfu3enRowdvvPHGXt/j/vvvJyMjg4yMDB588EEAMjMzOf744+nZsycZGRm88sorAEyePJlhw4bRt29fRo4cyapVq0qso65du1KlShXWr1/P8OHD+eMf/8iwYcN46KGHuO2227jvvvsAWLBgAUcddRQ9e/akT58+/PjjjwDce++99O/fnx49enDrrbfux9EREUlsC9dt59Z3ZvD0VwuZsnQTu3LzYh2SoJa3Pfz5vzOZtXJrme7zkGZ1uPXEbsUunzFjBn379j2gfVerVo1x48YB8Ic//GH3+/Xr13PqqacyZswYatasyd13383999/PLbfcAkCjRo2YMmUKjz/+OPfddx9PP/00l112GbVq1eLaa6/d63PuuOMO6taty/Tp0wHYtGnTHssnT57Mc889x7fffou7c+ihhzJs2DAWLlxIs2bNeO+99wDYsmULOTk5XHnllbzzzjukpaXxyiuvcNNNN/Hss88W+z2//fZbkpKSSEsLpnfbvHkzX3zxBRBcXi1w9tln84c//IFRo0axc+dO8vPz+fjjj5k/fz4TJ07E3TnppJP48ssvGTp06AHVuYhIopi0eCMXjZ7E9p255OY7AKlVkshoVoc+rerTp3V9+rdpQFrtqjGONPEoeYtjZ5xxRpHvJ0yYwKxZsxgyZAgA2dnZDBo0aPd6p556KgB9+/blzTff3OfnjBkzZvclXYD69evvsXzcuHGMGjWKmjVr7t7/V199xbHHHsu1117LDTfcwAknnMDhhx/OjBkzmDFjBkcffTQQXOJNT08v8nMfeOAB/vOf/1C7dm1eeeWV3Z0ECn9vgG3btrFixQpGjRoFBIktwMcff8zHH39M7969gaAVcf78+UreRERK8P70VVzzylSa16vOO1cMoXpKMlOWbmLK0s1MWbKJ0ROW8PS4RVw3sjNXjOgQ63ATjpK3CCW1kEVLt27deP3114tcVqVKFfLz83e/Lzw0RUGyVPi9u3P00Ufz0ksvFbnfqlWDv5KSk5PJzc3dZ4zuXmLvSncvsrxTp05MnjyZ999/nxtvvJFjjjmGUaNG0a1bN8aPH7/Pzy24562wwt+7pBjcnRtvvJFLL710n58nIiLw9FcLuev92fRpVZ9/ntuPBjVTATg2I51jM4I/trNz85m5cguN61SLZagJS/e8xdgRRxzBrl27+Oc//7m77LvvvuOLL76gdevWzJo1i127drFlyxY+/fTTUu1z4MCBfP311yxYsACArKysffZerV27Ntu2bSty2THHHMOjjz66+33hy6ZDhw7l7bffJisri8zMTN566y0OP/xwVq5cSY0aNfjVr37Ftddey5QpU+jcuTPr1q3bnbzl5OQwc+bMUn2vktSpU4cWLVrw9ttvA7Br1y6ysrIYOXIkzz777O57/lasWMHatWsP+vNERCqbvHznz/+dyZ3vzebYbk154aJDdyduhaVWSaJ3q/o0r1e9nKMUUPIWc2bGW2+9xSeffEL79u3p1q0bt912G82aNaNly5b84he/oEePHpx99tm7L/3tS1paGs8//zxnnXUWPXr0YODAgcyZM6fEbU488UTeeuutIjss3HzzzWzatImMjAx69uzJ559/vsfyPn36cP755zNgwAAOPfRQLrroInr37s306dMZMGAAvXr14q677uLmm28mNTWV119/nRtuuIGePXvSq1ev/R6ipDj//ve/efjhh+nRoweDBw9m9erVHHPMMfzyl79k0KBBdO/endNOO63YJFVEJFFl7srlihem8NzXi7lwSFse/WUfqqXs36gFUn6suMtNlVG/fv180qRJe5TNnj2brl27xigiiWc6d0QkXu3IzmPykk18u2gDExZuYOqyzeTmOzcffwi/PkzT/lUUZjbZ3fsVLtc9byIiIglg284cXvh2KWNmrWHa8s3k5DnJSUZG87pcOKQtx3RrQt/WDWIdppSCkjcREZFKbHNWNs9+vZjnv17E1p259GxRlwsPa8vAdg3p17o+taulxDpE2U9K3kRERCqh9dt38cy4RYz+ZjGZ2XmM7NaEK4/oSEbzurEOTQ6SkjcREZE4l5/vrN66k8UbMlm6IYsZK7fw+uTl7MrN54QezbhiRHu6NK0T6zCljCh5ExERiQM5efms2LQjSNA2ZrF4fRZLNmSyZGMWSzdmkZ3707igqclJnNAznStGdKB9Wq0YRi3RoORNRESkgtiZkxcmZmGCtiGTJRuyWLIhixWbd5CX/9MIEdVTkmndsAbt02pyZJfGtGpYgzYNa9KqQQ2a1atOclLxg6tLfFPyFmOLFy/mhBNOYMaMGbvLbrvttmLnGY2mSZMmMXr0aB5++OGofUatWrV2D5hb1s4//3y++OIL6tatS1JSEo899tge04IVuOWWWxg6dChHHXVUVOIQEdlfu3LzuOhfk/hq/vo9yutWT6FNwxr0bFmPk3s1o1WDGrRpVJPWDWuQVqtqibPfSOWl5E1269evH/367TWcTFy59957Oe200/j444+59NJL+eGHH/ZYnpeXx+233x6j6EREinb3B3P5av56Lh3ajkOa1aFNwyBBq1ej6BkOJLFphoUKbvjw4VxzzTUMHjyYjIwMJk6cCAQTrF9wwQV0796dHj168MYbbwDw0ksv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\n",
      "text/plain": [
       "<Figure size 720x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# plotting Upcoming Close price on date index\n",
    "fig,ax=plt.subplots(figsize=(10,5))\n",
    "ax.plot(df_merge.loc['2021-04-01':,'close'],label='Current close Price')\n",
    "ax.plot(upcoming_prediction.loc['2021-04-01':,'close'],label='Upcoming close Price')\n",
    "plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)\n",
    "ax.set_xlabel('Date',size=15)\n",
    "ax.set_ylabel('Stock Price',size=15)\n",
    "ax.set_title('Upcoming close price prediction',size=15)\n",
    "ax.legend()\n",
    "fig.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1a008259",
   "metadata": {},
   "source": [
    "# THANK YOU!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d60878c1",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
