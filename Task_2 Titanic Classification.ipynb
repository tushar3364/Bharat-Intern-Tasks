{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# DATA SCIENCE INTERN @BHARAT INTERN"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### AUTHOR : TUSHAR KUMAR"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# TASK 2 : TITANIC CLASSIFICATION"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Algorithm which tells whether the person will be save from sinking or not"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Purpose\n",
    "**Titanic** is one of the most infamous shipwrecks in history. On April 15, 1912, during her maiden voyage, the **Titanic** sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.\n",
    "\n",
    "While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.\n",
    "\n",
    "In this challenge, we ask you to build a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc).\n",
    "\n",
    "The dataset is available at Kaggle : https://www.kaggle.com/datasets/rahulsah06/titanic"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## STEPS INVOLVED : \n",
    "### 1. Problem understanding and definition\n",
    "### 2. Data Loading and Importing the necessary libraries\n",
    "### 3. Data understanding using Exploratory Data Analysis (EDA)\n",
    "### 4. Feature Engineering and Data Processing\n",
    "### 5. Feature Engineering and Data Processing\n",
    "### 6. Model Evaluation"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id=\"section1\"></a>\n",
    "## 1. Problem understanding and definition\n",
    "\n",
    "\n",
    "In this challenge, we need to complete the __analysis__ of what sorts of people were most likely to __survive__. In particular,  we apply the tools of __machine learning__ to predict which passengers survived the tragedy\n",
    "\n",
    "- Predict whether passenger will __survive or not__."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id=\"section2\"></a>\n",
    "## 2. Data Loading and Importing the necessary libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Linear algebra\n",
    "import numpy as np\n",
    "\n",
    "# Data manipulation and analysis\n",
    "import pandas as pd\n",
    "\n",
    "# Data visualization\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline\n",
    "from matplotlib import style\n",
    "\n",
    "# Algorithms\n",
    "from sklearn import linear_model\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.linear_model import Perceptron\n",
    "from sklearn.linear_model import SGDClassifier\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn.naive_bayes import GaussianNB"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id=\"section201\"></a>\n",
    "### 2.1  Loading the data files \n",
    "\n",
    "Here we import the data. For this analysis, we will be exclusively working with the Training set. We will be validating based on data from the training set as well. For our final submissions, we will make predictions based on the test set."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',\n",
       "       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'train_test'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_df = pd.read_csv('task_2 Train.csv')\n",
    "test_df = pd.read_csv('task_2 Test.csv')\n",
    "\n",
    "train_df['train_test'] = 1\n",
    "test_df['train_test'] = 0\n",
    "# test_df['Survived'] = np.NaN\n",
    "all_data = pd.concat([train_df,test_df])\n",
    "\n",
    "%matplotlib inline\n",
    "all_data.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
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
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "      <th>train_test</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Braund, Mr. Owen Harris</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>A/5 21171</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>PC 17599</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C85</td>\n",
       "      <td>C</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>Heikkinen, Miss. Laina</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>STON/O2. 3101282</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113803</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>C123</td>\n",
       "      <td>S</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Allen, Mr. William Henry</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>373450</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>6</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Moran, Mr. James</td>\n",
       "      <td>male</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>330877</td>\n",
       "      <td>8.4583</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Q</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>7</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>McCarthy, Mr. Timothy J</td>\n",
       "      <td>male</td>\n",
       "      <td>54.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>17463</td>\n",
       "      <td>51.8625</td>\n",
       "      <td>E46</td>\n",
       "      <td>S</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>8</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Palsson, Master. Gosta Leonard</td>\n",
       "      <td>male</td>\n",
       "      <td>2.0</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>349909</td>\n",
       "      <td>21.0750</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>9</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)</td>\n",
       "      <td>female</td>\n",
       "      <td>27.0</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>347742</td>\n",
       "      <td>11.1333</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>10</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>Nasser, Mrs. Nicholas (Adele Achem)</td>\n",
       "      <td>female</td>\n",
       "      <td>14.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>237736</td>\n",
       "      <td>30.0708</td>\n",
       "      <td>NaN</td>\n",
       "      <td>C</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Survived  Pclass  \\\n",
       "0            1         0       3   \n",
       "1            2         1       1   \n",
       "2            3         1       3   \n",
       "3            4         1       1   \n",
       "4            5         0       3   \n",
       "5            6         0       3   \n",
       "6            7         0       1   \n",
       "7            8         0       3   \n",
       "8            9         1       3   \n",
       "9           10         1       2   \n",
       "\n",
       "                                                Name     Sex   Age  SibSp  \\\n",
       "0                            Braund, Mr. Owen Harris    male  22.0      1   \n",
       "1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   \n",
       "2                             Heikkinen, Miss. Laina  female  26.0      0   \n",
       "3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   \n",
       "4                           Allen, Mr. William Henry    male  35.0      0   \n",
       "5                                   Moran, Mr. James    male   NaN      0   \n",
       "6                            McCarthy, Mr. Timothy J    male  54.0      0   \n",
       "7                     Palsson, Master. Gosta Leonard    male   2.0      3   \n",
       "8  Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)  female  27.0      0   \n",
       "9                Nasser, Mrs. Nicholas (Adele Achem)  female  14.0      1   \n",
       "\n",
       "   Parch            Ticket     Fare Cabin Embarked  train_test  \n",
       "0      0         A/5 21171   7.2500   NaN        S           1  \n",
       "1      0          PC 17599  71.2833   C85        C           1  \n",
       "2      0  STON/O2. 3101282   7.9250   NaN        S           1  \n",
       "3      0            113803  53.1000  C123        S           1  \n",
       "4      0            373450   8.0500   NaN        S           1  \n",
       "5      0            330877   8.4583   NaN        Q           1  \n",
       "6      0             17463  51.8625   E46        S           1  \n",
       "7      1            349909  21.0750   NaN        S           1  \n",
       "8      2            347742  11.1333   NaN        S           1  \n",
       "9      0            237736  30.0708   NaN        C           1  "
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_df.head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
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
       "      <th>PassengerId</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "      <th>train_test</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>892</td>\n",
       "      <td>3</td>\n",
       "      <td>Kelly, Mr. James</td>\n",
       "      <td>male</td>\n",
       "      <td>34.5</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>330911</td>\n",
       "      <td>7.8292</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Q</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>893</td>\n",
       "      <td>3</td>\n",
       "      <td>Wilkes, Mrs. James (Ellen Needs)</td>\n",
       "      <td>female</td>\n",
       "      <td>47.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>363272</td>\n",
       "      <td>7.0000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>894</td>\n",
       "      <td>2</td>\n",
       "      <td>Myles, Mr. Thomas Francis</td>\n",
       "      <td>male</td>\n",
       "      <td>62.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>240276</td>\n",
       "      <td>9.6875</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Q</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>895</td>\n",
       "      <td>3</td>\n",
       "      <td>Wirz, Mr. Albert</td>\n",
       "      <td>male</td>\n",
       "      <td>27.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>315154</td>\n",
       "      <td>8.6625</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>896</td>\n",
       "      <td>3</td>\n",
       "      <td>Hirvonen, Mrs. Alexander (Helga E Lindqvist)</td>\n",
       "      <td>female</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>3101298</td>\n",
       "      <td>12.2875</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>897</td>\n",
       "      <td>3</td>\n",
       "      <td>Svensson, Mr. Johan Cervin</td>\n",
       "      <td>male</td>\n",
       "      <td>14.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7538</td>\n",
       "      <td>9.2250</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>898</td>\n",
       "      <td>3</td>\n",
       "      <td>Connolly, Miss. Kate</td>\n",
       "      <td>female</td>\n",
       "      <td>30.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>330972</td>\n",
       "      <td>7.6292</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Q</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>899</td>\n",
       "      <td>2</td>\n",
       "      <td>Caldwell, Mr. Albert Francis</td>\n",
       "      <td>male</td>\n",
       "      <td>26.0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>248738</td>\n",
       "      <td>29.0000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>900</td>\n",
       "      <td>3</td>\n",
       "      <td>Abrahim, Mrs. Joseph (Sophie Halaut Easu)</td>\n",
       "      <td>female</td>\n",
       "      <td>18.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2657</td>\n",
       "      <td>7.2292</td>\n",
       "      <td>NaN</td>\n",
       "      <td>C</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>901</td>\n",
       "      <td>3</td>\n",
       "      <td>Davies, Mr. John Samuel</td>\n",
       "      <td>male</td>\n",
       "      <td>21.0</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>A/4 48871</td>\n",
       "      <td>24.1500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Pclass                                          Name     Sex  \\\n",
       "0          892       3                              Kelly, Mr. James    male   \n",
       "1          893       3              Wilkes, Mrs. James (Ellen Needs)  female   \n",
       "2          894       2                     Myles, Mr. Thomas Francis    male   \n",
       "3          895       3                              Wirz, Mr. Albert    male   \n",
       "4          896       3  Hirvonen, Mrs. Alexander (Helga E Lindqvist)  female   \n",
       "5          897       3                    Svensson, Mr. Johan Cervin    male   \n",
       "6          898       3                          Connolly, Miss. Kate  female   \n",
       "7          899       2                  Caldwell, Mr. Albert Francis    male   \n",
       "8          900       3     Abrahim, Mrs. Joseph (Sophie Halaut Easu)  female   \n",
       "9          901       3                       Davies, Mr. John Samuel    male   \n",
       "\n",
       "    Age  SibSp  Parch     Ticket     Fare Cabin Embarked  train_test  \n",
       "0  34.5      0      0     330911   7.8292   NaN        Q           0  \n",
       "1  47.0      1      0     363272   7.0000   NaN        S           0  \n",
       "2  62.0      0      0     240276   9.6875   NaN        Q           0  \n",
       "3  27.0      0      0     315154   8.6625   NaN        S           0  \n",
       "4  22.0      1      1    3101298  12.2875   NaN        S           0  \n",
       "5  14.0      0      0       7538   9.2250   NaN        S           0  \n",
       "6  30.0      0      0     330972   7.6292   NaN        Q           0  \n",
       "7  26.0      1      1     248738  29.0000   NaN        S           0  \n",
       "8  18.0      0      0       2657   7.2292   NaN        C           0  \n",
       "9  21.0      2      0  A/4 48871  24.1500   NaN        S           0  "
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test_df.head(10)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id=\"section202\"></a>\n",
    "### 2.2 About The Dataset"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The data has been split into two groups:\n",
    "- training set (train.csv)\n",
    "- test set (test.csv)\n",
    "\n",
    "The training set includes passengers survival status (also know as the ground truth from the titanic tragedy) which along with other features like gender, class, fare and pclass is used to create the machine learning model.\n",
    "\n",
    "The test set should be used to see how well the model performs on unseen data. The test set does not provide passengers survival status. We are going to use our model to predict passenger survival status.\n",
    "\n",
    "This is clearly a <font color='red'>__Classification problem__.</font> In predictive analytics, when the <font color='red'>__target__</font> is a categorical variable, we are in a category of tasks known as <font color='red'>__classification tasks.__</font>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "| Column Name          | Description                                                | Key                    |\n",
    "| ---------------------| ---------------------------------------------------------- | ---------------------- |\n",
    "| __PassengerId__      | Passenger Identity                                         |                        | \n",
    "| __Survived__         | Whether passenger survived or not                          | 0 = No, 1 = Yes        | \n",
    "| __Pclass__           | Class of ticket, a proxy for socio-economic status (SES)| 1 = 1st, 2 = 2nd, 3 = 3rd | \n",
    "| __Name__             | Name of passenger                                          |                        | \n",
    "| __Sex__              | Sex of passenger                                           |                        |\n",
    "| __Age__              | Age of passenger in years                                  |                        |\n",
    "| __SibSp__            | Number of sibling and/or spouse travelling with passenger  |                        |\n",
    "| __Parch__            | Number of parent and/or children travelling with passenger |                        |\n",
    "| __Ticket__           | Ticket number                                              |                        |\n",
    "| __Fare__             | Price of ticket                                            |                        |\n",
    "| __Cabin__            | Cabin number                                               |                        |\n",
    "| __Embarked__         | Port of embarkation                                        | C = Cherbourg, Q = Queenstown, S = Southampton |"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id=\"section3\"></a>\n",
    "## 3. Data understanding using Exploratory Data Analysis (EDA)\n",
    "__Exploratory Data Analysis__ refers to the critical process of performing initial investigations on data so as to discover patterns, to spot anomalies, to test hypothesis and to check assumptions with the help of summary statistics and graphical representations.\n",
    "\n",
    "In summary, it's an approach to analyzing data sets to summarize their main characteristics, often with visual methods."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 891 entries, 0 to 890\n",
      "Data columns (total 13 columns):\n",
      " #   Column       Non-Null Count  Dtype  \n",
      "---  ------       --------------  -----  \n",
      " 0   PassengerId  891 non-null    int64  \n",
      " 1   Survived     891 non-null    int64  \n",
      " 2   Pclass       891 non-null    int64  \n",
      " 3   Name         891 non-null    object \n",
      " 4   Sex          891 non-null    object \n",
      " 5   Age          714 non-null    float64\n",
      " 6   SibSp        891 non-null    int64  \n",
      " 7   Parch        891 non-null    int64  \n",
      " 8   Ticket       891 non-null    object \n",
      " 9   Fare         891 non-null    float64\n",
      " 10  Cabin        204 non-null    object \n",
      " 11  Embarked     889 non-null    object \n",
      " 12  train_test   891 non-null    int64  \n",
      "dtypes: float64(2), int64(6), object(5)\n",
      "memory usage: 90.6+ KB\n"
     ]
    }
   ],
   "source": [
    "train_df.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The training-set has 891 rows and 11 features + the __target variable (survived).__ 2 of the features are floats, 5 are integers and 5 are objects."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "scrolled": false
   },
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
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>train_test</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>714.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>446.000000</td>\n",
       "      <td>0.383838</td>\n",
       "      <td>2.308642</td>\n",
       "      <td>29.699118</td>\n",
       "      <td>0.523008</td>\n",
       "      <td>0.381594</td>\n",
       "      <td>32.204208</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>257.353842</td>\n",
       "      <td>0.486592</td>\n",
       "      <td>0.836071</td>\n",
       "      <td>14.526497</td>\n",
       "      <td>1.102743</td>\n",
       "      <td>0.806057</td>\n",
       "      <td>49.693429</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.420000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>223.500000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>20.125000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>7.910400</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>446.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>28.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>14.454200</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>668.500000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>38.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>31.000000</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>891.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>80.000000</td>\n",
       "      <td>8.000000</td>\n",
       "      <td>6.000000</td>\n",
       "      <td>512.329200</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       PassengerId    Survived      Pclass         Age       SibSp  \\\n",
       "count   891.000000  891.000000  891.000000  714.000000  891.000000   \n",
       "mean    446.000000    0.383838    2.308642   29.699118    0.523008   \n",
       "std     257.353842    0.486592    0.836071   14.526497    1.102743   \n",
       "min       1.000000    0.000000    1.000000    0.420000    0.000000   \n",
       "25%     223.500000    0.000000    2.000000   20.125000    0.000000   \n",
       "50%     446.000000    0.000000    3.000000   28.000000    0.000000   \n",
       "75%     668.500000    1.000000    3.000000   38.000000    1.000000   \n",
       "max     891.000000    1.000000    3.000000   80.000000    8.000000   \n",
       "\n",
       "            Parch        Fare  train_test  \n",
       "count  891.000000  891.000000       891.0  \n",
       "mean     0.381594   32.204208         1.0  \n",
       "std      0.806057   49.693429         0.0  \n",
       "min      0.000000    0.000000         1.0  \n",
       "25%      0.000000    7.910400         1.0  \n",
       "50%      0.000000   14.454200         1.0  \n",
       "75%      0.000000   31.000000         1.0  \n",
       "max      6.000000  512.329200         1.0  "
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_df.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Conclusions from .describe() method\n",
    "__.describe()__ gives an understanding of the central tendencies of the numeric data.\n",
    "\n",
    "- Above we can see that __38% out of the training-set survived the Titanic.__ \n",
    "- We can also see that the passenger age range from __0.4 to 80 years old.__\n",
    "- We can already detect some features that contain __missing values__, like the ‘Age’ feature (714 out of 891 total).\n",
    "- There's an __outlier__ for the 'Fare' price because of the differences between the 75th percentile, standard deviation, and the max value (512). We might want to drop that value."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id=\"section301\"></a>\n",
    "### 3.1 Exploring missing data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
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
       "      <th>Total</th>\n",
       "      <th>%</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Cabin</th>\n",
       "      <td>687</td>\n",
       "      <td>77.1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Age</th>\n",
       "      <td>177</td>\n",
       "      <td>19.9</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Embarked</th>\n",
       "      <td>2</td>\n",
       "      <td>0.2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>PassengerId</th>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Survived</th>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Pclass</th>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Name</th>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Sex</th>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>SibSp</th>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Parch</th>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Ticket</th>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Fare</th>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>train_test</th>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "             Total     %\n",
       "Cabin          687  77.1\n",
       "Age            177  19.9\n",
       "Embarked         2   0.2\n",
       "PassengerId      0   0.0\n",
       "Survived         0   0.0\n",
       "Pclass           0   0.0\n",
       "Name             0   0.0\n",
       "Sex              0   0.0\n",
       "SibSp            0   0.0\n",
       "Parch            0   0.0\n",
       "Ticket           0   0.0\n",
       "Fare             0   0.0\n",
       "train_test       0   0.0"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "total = train_df.isnull().sum().sort_values(ascending=False)\n",
    "percent_1 = train_df.isnull().sum()/train_df.isnull().count()*100\n",
    "percent_2 = (round(percent_1, 1)).sort_values(ascending=False)\n",
    "missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])\n",
    "missing_data.head(13)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The __'Embarked'__ feature has only 2 missing values, which can easily be filled or dropped. It will be much more tricky to deal with the __‘Age’__ feature, which has 177 missing values. The __‘Cabin’__ feature needs further investigation, but it looks like we might want to drop it from the dataset since 77% is missing."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',\n",
       "       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'train_test'],\n",
       "      dtype=object)"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_df.columns.values"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Above we can see the 11 features and the target variable (survived). __What features could contribute to a high survival rate ?__\n",
    "\n",
    "I believe it would make sense if everything except ‘PassengerId’, ‘Name’ and ‘Ticket’ would be high correlated with survival rate."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id=\"section302\"></a>\n",
    "### 3.2 Dealing with the outlier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.boxplot(x='Survived',y='Fare',data=train_df);"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Passengers who paid over 300"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
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
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "      <th>train_test</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>258</th>\n",
       "      <td>259</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Ward, Miss. Anna</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>PC 17755</td>\n",
       "      <td>512.3292</td>\n",
       "      <td>NaN</td>\n",
       "      <td>C</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>679</th>\n",
       "      <td>680</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Cardeza, Mr. Thomas Drake Martinez</td>\n",
       "      <td>male</td>\n",
       "      <td>36.0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>PC 17755</td>\n",
       "      <td>512.3292</td>\n",
       "      <td>B51 B53 B55</td>\n",
       "      <td>C</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>737</th>\n",
       "      <td>738</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Lesurer, Mr. Gustave J</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>PC 17755</td>\n",
       "      <td>512.3292</td>\n",
       "      <td>B101</td>\n",
       "      <td>C</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     PassengerId  Survived  Pclass                                Name  \\\n",
       "258          259         1       1                    Ward, Miss. Anna   \n",
       "679          680         1       1  Cardeza, Mr. Thomas Drake Martinez   \n",
       "737          738         1       1              Lesurer, Mr. Gustave J   \n",
       "\n",
       "        Sex   Age  SibSp  Parch    Ticket      Fare        Cabin Embarked  \\\n",
       "258  female  35.0      0      0  PC 17755  512.3292          NaN        C   \n",
       "679    male  36.0      0      1  PC 17755  512.3292  B51 B53 B55        C   \n",
       "737    male  35.0      0      0  PC 17755  512.3292         B101        C   \n",
       "\n",
       "     train_test  \n",
       "258           1  \n",
       "679           1  \n",
       "737           1  "
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_df[train_df['Fare']>300]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Drop the outliers\n",
    "\n",
    "It might be beneficial to drop those outliers for the model. Further investigation needs to be done."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "# train_df = train_df[train_df['Fare']<300]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### The Captain went down with the ship\n",
    "__\"The captain goes down with the ship\"__ is a maritime tradition that a sea captain holds ultimate responsibility for both his/her ship and everyone embarked on it, and that in an emergency, he/she will either save them or die trying.\n",
    "\n",
    "In this case, __Captain Edward Gifford Crosby__ went down with Titanic in a heroic gesture trying to save the passengers."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
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
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "      <th>train_test</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>745</th>\n",
       "      <td>746</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>Crosby, Capt. Edward Gifford</td>\n",
       "      <td>male</td>\n",
       "      <td>70.0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>WE/P 5735</td>\n",
       "      <td>71.0</td>\n",
       "      <td>B22</td>\n",
       "      <td>S</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     PassengerId  Survived  Pclass                          Name   Sex   Age  \\\n",
       "745          746         0       1  Crosby, Capt. Edward Gifford  male  70.0   \n",
       "\n",
       "     SibSp  Parch     Ticket  Fare Cabin Embarked  train_test  \n",
       "745      1      1  WE/P 5735  71.0   B22        S           1  "
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_df[train_df['Name'].str.contains(\"Capt\")]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id=\"section303\"></a>\n",
    "### 3.3 Embarked, Pclass and Sex:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1107.3x288 with 3 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "FacetGrid = sns.FacetGrid(train_df, col='Embarked', height=4, aspect=1.2)\n",
    "FacetGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', ci=95.0, palette='deep', order=None, hue_order=None)\n",
    "FacetGrid.add_legend();"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id=\"section304\"></a>\n",
    "### 3.4 Distribution of Pclass and Survived"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 576x432 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.set(style='darkgrid')\n",
    "plt.subplots(figsize = (8,6))\n",
    "ax=sns.countplot(x='Sex', data = train_df, hue='Survived', edgecolor=(0,0,0), linewidth=2)\n",
    "\n",
    "# Fixing title, xlabel and ylabel\n",
    "plt.title('Passenger distribution of survived vs not-survived', fontsize=25)\n",
    "plt.xlabel('Gender', fontsize=15)\n",
    "plt.ylabel(\"# of Passenger Survived\", fontsize = 15)\n",
    "labels = ['Female', 'Male']\n",
    "\n",
    "# Fixing xticks.\n",
    "plt.xticks(sorted(train_df.Survived.unique()),labels);"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
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
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>train_test</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Sex</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>female</th>\n",
       "      <td>431.028662</td>\n",
       "      <td>0.742038</td>\n",
       "      <td>2.159236</td>\n",
       "      <td>27.915709</td>\n",
       "      <td>0.694268</td>\n",
       "      <td>0.649682</td>\n",
       "      <td>44.479818</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>male</th>\n",
       "      <td>454.147314</td>\n",
       "      <td>0.188908</td>\n",
       "      <td>2.389948</td>\n",
       "      <td>30.726645</td>\n",
       "      <td>0.429809</td>\n",
       "      <td>0.235702</td>\n",
       "      <td>25.523893</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        PassengerId  Survived    Pclass        Age     SibSp     Parch  \\\n",
       "Sex                                                                      \n",
       "female   431.028662  0.742038  2.159236  27.915709  0.694268  0.649682   \n",
       "male     454.147314  0.188908  2.389948  30.726645  0.429809  0.235702   \n",
       "\n",
       "             Fare  train_test  \n",
       "Sex                            \n",
       "female  44.479818         1.0  \n",
       "male    25.523893         1.0  "
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_df.groupby(['Sex']).mean()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "As previously mentioned, women are much more likely to survive than men. __74% of the women survived, while only 18% of men survived.__"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Looking deeper into differences between females and males statistics"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
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
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>train_test</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Sex</th>\n",
       "      <th>Pclass</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th rowspan=\"3\" valign=\"top\">female</th>\n",
       "      <th>1</th>\n",
       "      <td>469.212766</td>\n",
       "      <td>0.968085</td>\n",
       "      <td>34.611765</td>\n",
       "      <td>0.553191</td>\n",
       "      <td>0.457447</td>\n",
       "      <td>106.125798</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>443.105263</td>\n",
       "      <td>0.921053</td>\n",
       "      <td>28.722973</td>\n",
       "      <td>0.486842</td>\n",
       "      <td>0.605263</td>\n",
       "      <td>21.970121</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>399.729167</td>\n",
       "      <td>0.500000</td>\n",
       "      <td>21.750000</td>\n",
       "      <td>0.895833</td>\n",
       "      <td>0.798611</td>\n",
       "      <td>16.118810</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th rowspan=\"3\" valign=\"top\">male</th>\n",
       "      <th>1</th>\n",
       "      <td>455.729508</td>\n",
       "      <td>0.368852</td>\n",
       "      <td>41.281386</td>\n",
       "      <td>0.311475</td>\n",
       "      <td>0.278689</td>\n",
       "      <td>67.226127</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>447.962963</td>\n",
       "      <td>0.157407</td>\n",
       "      <td>30.740707</td>\n",
       "      <td>0.342593</td>\n",
       "      <td>0.222222</td>\n",
       "      <td>19.741782</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>455.515850</td>\n",
       "      <td>0.135447</td>\n",
       "      <td>26.507589</td>\n",
       "      <td>0.498559</td>\n",
       "      <td>0.224784</td>\n",
       "      <td>12.661633</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "               PassengerId  Survived        Age     SibSp     Parch  \\\n",
       "Sex    Pclass                                                         \n",
       "female 1        469.212766  0.968085  34.611765  0.553191  0.457447   \n",
       "       2        443.105263  0.921053  28.722973  0.486842  0.605263   \n",
       "       3        399.729167  0.500000  21.750000  0.895833  0.798611   \n",
       "male   1        455.729508  0.368852  41.281386  0.311475  0.278689   \n",
       "       2        447.962963  0.157407  30.740707  0.342593  0.222222   \n",
       "       3        455.515850  0.135447  26.507589  0.498559  0.224784   \n",
       "\n",
       "                     Fare  train_test  \n",
       "Sex    Pclass                          \n",
       "female 1       106.125798         1.0  \n",
       "       2        21.970121         1.0  \n",
       "       3        16.118810         1.0  \n",
       "male   1        67.226127         1.0  \n",
       "       2        19.741782         1.0  \n",
       "       3        12.661633         1.0  "
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_df.groupby(['Sex','Pclass']).mean()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We are grouping passengers based on Sex and Ticket class (Pclass). Notice the difference between survival rates between men and women.\n",
    "\n",
    "Women are much more likely to survive than men, **specially women in the first and second class.** It also shows that men in the first class are almost **3-times more likely to survive** than men in the third class."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id=\"section305\"></a>\n",
    "### 3.5 Age and Sex distributions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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uzXNj/X7h9bjdmTS32PTz2PT322T67bIJEKu2yLXZeFK8gvXZ2X7cLvMbm0jFru05xmqABQREZE+y+EwujQZTTGfTmJz9ajCHo5K5OK6O4FST437traRdfZnBGB7oAkHNZab4tkz2PTyJurDLo42OdljO4niBD7dc5jptMhyWmQ6bbJczb8zzy77TtYRjEGaI3H3IrpdJv+5untfst4776oeiubC4lIArl27lt/97ndEo1G+9rWvcccdd5x3/549e/jBD35AY2MjEyZM4Kc//SlOp2pVERERaD+PiogkCsMALxZes4nSKZcTee3llvtsG664opCjNVGqIg7OhB2ciTiojRp8EnFQETAJWq12Cf7vN4EsnIZNumnjNW185qe3000L39m/pTls3A5wO2yyDINY2EFBXRNNMXA7oIs7QFNCr1dVp06dYvny5axevRq3282CBQu49tprGTZsWMs63/3ud3nkkUcYN24cS5cuZdWqVdx+++29HaqIiEjC6UgeFRFJBoYBmS6aLzNB7ILrhC2ojzqoixpkfOGLfLB+E/VRB4GY0fJzOuSg8exti7YqOw/89DWa902C07BxGuAwbEwDTMA8e9thnL0NtN6cb8U/iUQ/jdNofefZmy6nyamqRqD54NiLsj9do/W6Y4blccWArl3vsCN6vQAsLy9n0qRJZGdnA3DzzTezbt06vvWtbwFw7NgxmpqaGDduHADz5s1jxYoVnSoAu3JM82fb+H2uNtf1ZXfsBTHTfRdct6PtL7SNzrRtvY2utr1YLJ1xrl1XtvHZ9Vtvw+l04opGO9y2re10pX172+jOOOlo/9raxsXGb3u6M0582Rmdeo1b9691m54YJ53ZTnfGSVv3te5bW9vo7Ov02fW7837QnTECkJ7mwrC7d45CV88/6an2yaq9PNqe7jxv59q2Hj+Jmpc68v7Z2fesC7VvL46OtD8Xa2e30533vbbad3Qb3cmPHWnf3nY6k2culmc7qic+j53bzsXadzTvJ+tnutbtW8fiMtouSVxAOlAIDBtfxIC9F8pfNmBj280FY9AyCJ/7sQ0Ml5PGphjZV4/n2JadhG2IWAYxGywbYhjNv22wAMs2Wm63fgRfv3SikWjLcst9rRZcbhPLunB+bO+w1XMFZUa6q0feq9t8HNu22y1Oe9ITTzxBIBBgyZIlALzwwgvs3r2bhx9+GIAdO3awbNkyVq5cCcChQ4f4+te/zmuvvdabYYqIiCSk9vKoiIjIxfT6PDuWZWG0Kn9t2z5vub37RURE+jLlSRER6Y5eLwALCwuprKxsWa6srCQ/P7/N+z/55JPz7hcREenL2sujIiIiF9PrBeB1113H5s2bOXPmDMFgkPXr13P99de33F9UVITH42H79u0ArFmz5rz7RURE+rL28qiIiMjF9Po5gNA8ffUTTzxBJBJh/vz53Hvvvdx7770sXryYsWPH8uGHH/Lggw/S0NDAmDFj+MUvfoHb7W5/wyIiIn3AhfKoiIhIR8SlABQREREREZHe1+uHgIqIiIiIiEh8qAAUERERERHpI1QAioiIiIiI9BEqAEVERERERPqIPl8Arl27ltmzZzNjxgyee+65eIfTIxoaGrjllls4evQoAOXl5dx6663MmDGD5cuXxzm67vnNb35DaWkppaWlLFu2DEid/v36179m9uzZlJaW8vTTTwOp07fWHnvsMR544AEgdfp35513Ulpaypw5c5gzZw67du1Kmb4BlJWVMW/ePGbNmsUjjzwCpM5rJx2TDLkyWXJfsuWxZMxNyZJnkil3JFMeeOGFF1qe0zlz5vD5z3+ehx56KGHjXbNmTct7wmOPPQb0wnNr92EnT560p0+fbldXV9uNjY32rbfeau/fvz/eYXXLzp077VtuucUeM2aMfeTIETsYDNrTpk2zDx8+bEciEfvuu++2N27cGO8wu+Ttt9+2b7vtNjsUCtnhcNhetGiRvXbt2pTo37vvvmsvWLDAjkQidjAYtKdPn27v2bMnJfrWWnl5uX3ttdfa3//+91NmbFqWZU+dOtWORCItf0uVvtm2bR8+fNieOnWqfeLECTscDtsLFy60N27cmDL9k/YlQ65MltyXbHksGXNTsuSZZModyZwH9u3bZ99000328ePHEzLeQCBgT5w40a6qqrIjkYg9f/58e8OGDZc81j69B7C8vJxJkyaRnZ2Nz+fj5ptvZt26dfEOq1tWrVrFj3/8Y/Lz8wHYvXs3gwcPZtCgQTidTm699dak7WNeXh4PPPAAbrcbl8vF0KFDOXjwYEr075prruEPf/gDTqeTqqoqYrEYdXV1KdG3c2pqali+fDn33XcfkDpjs6KiAoC7776bL3/5yzz77LMp0zeA119/ndmzZ1NYWIjL5WL58uV4vd6U6Z+0LxlyZbLkvmTLY8mWm5IpzyRT7kjmPPCTn/yEJUuWcOTIkYSMNxaLYVkWwWCQaDRKNBrF7/df8lj7dAF4+vRp8vLyWpbz8/M5depUHCPqvp/97GdMmDChZTmV+njFFVcwbtw4AA4ePMjf//53DMNImf65XC5WrFhBaWkpkydPTqnXDuBHP/oRS5YsITMzE0idsVlXV8fkyZP57W9/yzPPPMPzzz/P8ePHU6JvAIcOHSIWi3HfffcxZ84c/vSnP6XMaycdkwyvd7LkvmTMY8mUm5IpzyRT7kjWPFBeXk5TUxOzZs1K2Hj9fj/f/va3mTVrFtOmTaOoqKhXYu3TBaBlWRiG0bJs2/Z5y6kgFfu4f/9+7r77br73ve8xaNCglOrf4sWL2bx5MydOnODgwYMp07cXXniBAQMGMHny5Ja/pcrYHD9+PMuWLSMjI4Pc3Fzmz5/PihUrUqJv0Pzt5ObNm/n5z3/On//8Z3bv3s2RI0dSpn/SvmT8X030mJMtjyVDbkq2PJNMuSNZ88Dzzz/PXXfdBSTuWPjwww956aWXePPNN9m0aRMOh6NX/secPbq1JFNYWMi2bdtalisrK1sOH0kVhYWFVFZWtiwnex+3b9/O4sWLWbp0KaWlpWzZsiUl+nfgwAHC4TCjRo3C6/UyY8YM1q1bh2maLeska98A/va3v1FZWcmcOXOora0lEAhw7NixlOjftm3biEQiLR86bNumqKgoJcYlQP/+/Zk8eTK5ubkAfOlLX0qpsSntS8Zcmci5L5nyWDLlpmTLM8mUO5IxD4TDYbZu3cqjjz4KJO57wltvvcXkyZPp168fAPPmzeOpp5665M9tn94DeN1117F582bOnDlDMBhk/fr1XH/99fEOq0ddddVVfPzxxy2771955ZWk7eOJEyf45je/yeOPP05paSmQOv07evQoDz74IOFwmHA4zIYNG1iwYEFK9A3g6aef5pVXXmHNmjUsXryYG264gd///vcp0b/6+nqWLVtGKBSioaGBv/zlL3znO99Jib4BTJ8+nbfeeou6ujpisRibNm1i5syZKdM/aV8y5spEzQ3JlseSKTclW55JptyRjHlg7969DBkyBJ/PByTu/9nIkSMpLy8nEAhg2zZlZWW9Emuf3gNYUFDAkiVLWLRoEZFIhPnz53PllVfGO6we5fF4ePTRR7n//vsJhUJMmzaNmTNnxjusLnnqqacIhUIt3+YALFiwICX6N23aNHbv3s3cuXMxTZMZM2ZQWlpKbm5u0vetLakyNqdPn86uXbuYO3culmVx++23M378+JToGzQnzXvuuYfbb7+dSCTClClTWLhwISUlJSnRP2lfMubKRH1/SbY8luy5KVHHASRX7kjGPHDkyBEKCwtblhN1LEydOpUPPviAefPm4XK5GDt2LPfffz9Tpky5pLEatm3bPbpFERERERERSUh9+hBQERERERGRvkQFoIiIiIiISB+hAlBERERERKSPUAEoIiIiIiLSR6gAFBERERER6SNUAIokiUgkwtSpU7nnnnviHYqIiEjCUH4U6RwVgCJJ4vXXX2fkyJG8//77HDhwIN7hiIiIJATlR5HO0XUARZLEnXfeyezZs9m/fz/RaJSHHnoIgCeffJIXX3yR9PR0JkyYwIYNGygrKyMcDvP444+zdetWYrEYo0eP5sEHH8Tv98e5JyIiIj1H+VGkc7QHUCQJfPTRR+zYsYOZM2cyd+5c1qxZQ3V1NZs2bWL16tW8+OKLrF69msbGxpY2Tz75JKZpsnr1al5++WXy8/N5/PHH49gLERGRnqX8KNJ5zngHICLtW7lyJdOnTycnJ4ecnBwGDhzIqlWrqKysZObMmWRmZgJwxx138M477wCwceNG6uvrKS8vB5rPkejXr1/c+iAiItLTlB9FOk8FoEiCCwQCrFmzBrfbzQ033ABAQ0MDzz77LKWlpbQ+its0zZbblmWxdOlSpk2bBkBjYyOhUKh3gxcREblElB9FukaHgIokuLVr15Kdnc2mTZsoKyujrKyMN954g0AgwJgxY1i/fj319fUAvPjiiy3tpk6dynPPPUc4HMayLH74wx/yy1/+Ml7dEBER6VHKjyJdowJQJMGtXLmSu+6667xvLzMzM7nzzjt55pln+OpXv8ptt93GvHnzqK+vx+v1AvCNb3yDoqIivvKVrzB79mxs2+aBBx6IVzdERER6lPKjSNdoFlCRJPbee++xY8cOFi1aBMDTTz/Nrl27+NWvfhXfwEREROJI+VGkbSoARZJYQ0MDS5cupaKiAsMwGDBgAA8//DAFBQXxDk1ERCRulB9F2qYCUEREREREpI/QOYAiIiIiIiJ9hApAERERERGRPkIFoIiIiIiISB+hAlBERERERKSPUAEoIiIiIiLSR6gAFBERERER6SP+P2/5tIpMR+81AAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 1080x360 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "survived = 'survived'\n",
    "not_survived = 'not survived'\n",
    "\n",
    "fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))\n",
    "\n",
    "women = train_df[train_df['Sex']=='female']\n",
    "men = train_df[train_df['Sex']=='male']\n",
    "\n",
    "# Plot Female Survived vs Not-Survived distribution\n",
    "ax = sns.histplot(women[women['Survived']==1].Age.dropna(), bins=20, label = survived, ax = axes[0],color='b', kde=True)\n",
    "ax = sns.histplot(women[women['Survived']==0].Age.dropna(), bins=20, label = not_survived, ax = axes[0],color='r', kde=True)\n",
    "ax.legend()\n",
    "ax.set_title('Female')\n",
    "\n",
    "# Plot Male Survived vs Not-Survived distribution\n",
    "ax = sns.histplot(men[men['Survived']==1].Age.dropna(), bins=20, label = survived, ax = axes[1],color='b', kde=True)\n",
    "ax = sns.histplot(men[men['Survived']==0].Age.dropna(), bins=20, label = not_survived, ax = axes[1],color='r', kde=True)\n",
    "ax.legend()\n",
    "ax.set_title('Male');"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We can see that __men__ have a higher probability of survival when they are between __18 and 35 years old.__ For __women,__ the survival chances are higher between __15 and 40 years old.__\n",
    "\n",
    "For men the probability of survival is very low between the __ages of 5 and 18__, and __after 35__, but that isn’t true for women. Another thing to note is that __infants have a higher probability of survival.__"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Saving children first"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
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
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>train_test</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Sex</th>\n",
       "      <th>Pclass</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th rowspan=\"3\" valign=\"top\">female</th>\n",
       "      <th>1</th>\n",
       "      <td>525.375000</td>\n",
       "      <td>0.875000</td>\n",
       "      <td>14.125000</td>\n",
       "      <td>0.500000</td>\n",
       "      <td>0.875000</td>\n",
       "      <td>104.083337</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>369.250000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>8.333333</td>\n",
       "      <td>0.583333</td>\n",
       "      <td>1.083333</td>\n",
       "      <td>26.241667</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>374.942857</td>\n",
       "      <td>0.542857</td>\n",
       "      <td>8.428571</td>\n",
       "      <td>1.571429</td>\n",
       "      <td>1.057143</td>\n",
       "      <td>18.727977</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th rowspan=\"3\" valign=\"top\">male</th>\n",
       "      <th>1</th>\n",
       "      <td>526.500000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>8.230000</td>\n",
       "      <td>0.500000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>116.072900</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>527.818182</td>\n",
       "      <td>0.818182</td>\n",
       "      <td>4.757273</td>\n",
       "      <td>0.727273</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>25.659473</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>437.953488</td>\n",
       "      <td>0.232558</td>\n",
       "      <td>9.963256</td>\n",
       "      <td>2.069767</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>22.752523</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "               PassengerId  Survived        Age     SibSp     Parch  \\\n",
       "Sex    Pclass                                                         \n",
       "female 1        525.375000  0.875000  14.125000  0.500000  0.875000   \n",
       "       2        369.250000  1.000000   8.333333  0.583333  1.083333   \n",
       "       3        374.942857  0.542857   8.428571  1.571429  1.057143   \n",
       "male   1        526.500000  1.000000   8.230000  0.500000  2.000000   \n",
       "       2        527.818182  0.818182   4.757273  0.727273  1.000000   \n",
       "       3        437.953488  0.232558   9.963256  2.069767  1.000000   \n",
       "\n",
       "                     Fare  train_test  \n",
       "Sex    Pclass                          \n",
       "female 1       104.083337         1.0  \n",
       "       2        26.241667         1.0  \n",
       "       3        18.727977         1.0  \n",
       "male   1       116.072900         1.0  \n",
       "       2        25.659473         1.0  \n",
       "       3        22.752523         1.0  "
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_df[train_df['Age']<18].groupby(['Sex','Pclass']).mean()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "__Children below 18 years of age__ have higher chances of surviving, proven they saved childen first"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id=\"section306\"></a>\n",
    "### 3.6 Passenger class distribution; Survived vs Non-Survived"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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TNDly5c1evRoeXt7a9WqVXctFx4erunTp6tVq1aqVq2aatSooY4dO2rBggWKiYm5a/45c+bI29tbM2fO1NatW+Xn56cqVaqoadOm+v7771N8L6Tb3VaBgYGSpE6dOqWr2X7AgAGaPXu2Vq5cqRw50rdp7dmzRyNHjlSzZs1UvXp1870fNGiQfv/9d7vLnDhxQmPGjFHr1q1VvXp11apVSwEBAZo1a5YiIiLumj8hIUFLlixR9+7d1bBhQ1WpUkUNGzbUgAED9PPPP6erzsx68cUXzZqTdvGuWrVK3t7eaty48V3LbN++Xf3791ezZs1UtWpV1a1bV927d9fixYsVGxt71zr27Nkj6fYvb29vb40ePVrS7SEd3t7eeuGFF3TixAl16dJFVatWVb169cwB602bNpW3t7eWL19ut/7Y2Fh9/PHH8vX1VdWqVdW4cWONGTPGbnd1aq9Juj1o3va9sX0Xbdu7zauvvpps+7dtw126dLG7zs2bN6t379569tlnzc81tW3nXr6394Otq3X//v2KjIxMcT5PT0+tWbNGW7duVZUqVczpSd/T06dP213W9hkn3aektW3UqFFD3t7e+uGHH1KsyfZZffjhh5Lu/vwNw9Dzzz8vb29vff311ymuZ/z48fL29tbIkSOTTY+KitLcuXMVEBCgGjVqqHr16vL399fs2bN1/fr1FNf3zz//aMSIEfLx8dHTTz8tf39/LV68WIZhpLiMPbZtr23btinOExQUJG9vb1WvXj1ZC25Gj1vpYdtWJWnixIm6evVqhtcRGhqq9957L9kxpF27dvr444/tvqdJjyFXrlzR5MmT1bRpU1WpUkX169fXsGHDMnVCS9LhRWntgwcNGqTvv/9e77//frLptn3Hf/7zH7vL2bbHpk2bJpvevXt3eXt7a9u2bZo7d67q16+vatWqqU2bNho2bFimPvM796O2v/3ss8+muD+5ePGiKlasaPe7u3fvXg0ePNg8XtWvX19vvPFGivs16fYx+5tvvlH79u1Vo0YN1a1bV8OGDVNISEiKy9iTpaEw6QsrWrSo+e8rV66oc+fOmjRpkn7//XflyZNH3t7ecnd317Fjx/TVV1+pffv2unTpUrL1TZ06VcOHD9eOHTtksVjk7e0tJycn7dixQyNHjjQPfjZnzpxR+/bt9e233+rs2bPy8vJSiRIldPbsWX37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VGoHfNtzSVtxHxTbdyKt98D2XbLXfZyR721qUvoxWalSJb311lvpWkdKSpcurdKlS+ull17Snj171L9/f0VFRWnJkiVZGgpT2jaKFi2q+vXra8eOHVq/fr369OmjhIQEs8Uvvd2xuXLlUqtWrbR06VKtW7fODIUpdUHbPt+QkJA0z5y07e+SDv9Iaf/q4uIiNzc38wS69OrQoYOWLl2qzZs3a8KECXJxcdGGDRvsBlop48etzOrevbu2bNmivXv36v/+7//M49qdkm7/qQVS23P//vuvDMNItj/MSPfjihUrzO7MO7311lvJujftcXNzU4MGDdSgQQMNGTJEY8eO1ffff699+/bp0KFDWXZnk9SOixn9zFMSEBCgDz/8ULt27dLly5fl6emp3377TRERESpRokSylj3bthwbG5uhPGL7f2q5IumQs7Tc99tOxMTE6JVXXtGJEydUoEABdenSRVWqVDEvcpozZ07NmjXLbii0WCzq1KmTOnXqpCtXrmj37t3as2ePtm3bpvPnz5tjH8aPH28uU7JkSU2ePFmTJk3SwYMHtWfPHv3+++/avXu3goOD1bt3b23cuFHFihUzu0j9/Pw0e/bs+/1WyM3NTXFxcalefPbOA1jSbtx169al+msgK1WsWFFeXl46f/68fvvtt3TdWmn//v166aWXlD9/fn311Vepfnnnzp1rjp9o1aqVGjdurHLlyumpp55S7ty5FRISYjcU2mr74IMPFBcXp/3792v37t3auXOn/vzzTwUFBalnz57asmWL+SVxcXHRq6++qldffVWXLl3Srl27tHv3bv36668KDw/Xhx9+qFy5cunVV1/NxDuVuqTbddKzbFPTvHlzNW/eXFFRUdqzZ4+5zZ88eVKBgYG6cePGfelGvJO9MGVj24HZ29mkFKiyalyQLZCm1WVl21nej1Zum5R23um5o8+hQ4c0ZswYXbt2TVu2bEn1IFWnTh1169Yt1bvXpPS+ZzQEJdWxY0ft2LFD69atU58+fbRz506FhYWpRIkSKd7XO6X1LF26VBs3btS4ceN09uxZ/f3333J3dze7gm3c3NzMbTy9t3RLemZqSvtXwzAyNZavevXqKlu2rE6cOKFt27apefPmZpe6vWCcmeNWZti6kdu2bauIiAhNmjTJ7o+UpNt/asceW7B2d3e/px/IFy9eTPF7YfvOTpgw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9f1Ix4wZIzc3NwUHB6tz587aunXrXQftEydO6I033lBQUJCefPLJZN/xXLlymWO05syZk+wktR07dmjSpEn3XKMtzH355ZeKiopS3bp1k132Kb3atWsnJycnBQYG6uTJkypVqpTdcXDdunWTh4eHTp8+rf79+yfbJq5cuaKhQ4fqxIkTcnZ21muvvWY+16tXLxUoUMDslreFbMMwtGTJknu6d66Hh4caNWqkGzdumL0c9roRM3rcygq2bmR3d/cUf1T16dNHuXPnVnBwsIYMGZLsR9vZs2fVr18/hYeHq0iRInrllVeyrLaUNGjQwGxxGz9+vN5999277jgWExOjlStXaujQoZJun0iatCvcNjb12LFjWrhwofnar127prFjx97z7SDT+5mnxfbDyjAMc5iavR9V7u7u6tu3ryTp3Xff1cqVK5M1Am3dulUTJ06UdPtYYbvHeIECBczvwbhx45L1Oly4cEFvvPFGhoaQ3PcxhU2bNlWNGjX0119/acqUKfriiy9UpEgRhYWFKTQ0VBaLRfXr19fOnTt1+fJlc4BrlSpVNHToUM2aNUszZ87U559/rhIlSujWrVs6e/as4uPjVapUqWRBatKkSXrxxRcVHBysNm3aqESJEsqbN68uXLhgXstp+PDhyc6QHjt2rEJDQ/Xzzz+rf//+8vT0VJEiRXT+/HkzfPn5+Zkb5r3IkyePPvroI7366qvaunWrfv31V5UvX143b95USEiIvLy89MQTTyg8PDzZpRhKly6tDz/8UMOGDdP69eu1efNmlStXTnFxcWZTs7u7uz7//PM07wSRUU2aNNH8+fM1ePBg/f333+rdu7fy5cunEiVKKD4+XqdOnTJ3fG3atNG7775r9zISSTk5OWnIkCHmXUZ8fHz05JNPKjY2VqdPn1Z8fLwqVaqkixcvKjIyUpcuXTJbFGfNmmUOvG/WrJlKlCghNzc3nTt3TtevX1fOnDk1adIks5vB19dXL7zwgpYtW6Y333xT7733nooVK6aoqCidOXNGhmHo6aefTldL3p2GDBmSrEU0NjZWV69e1fnz580d1AsvvKAJEyaka32enp6aMmWKRo4cqYULF2rFihUqVaqUEhISdObMGcXExKhgwYJ69913ky1XqVIl/fzzz1q3bp2OHj2q2rVrmzuPe+Hl5aVChQppyJAhKl68uAoWLKhjx44pNjZWHh4emjt3brJfzfnz51evXr302Wefaf369dq+fbtKlCih8+fP6+rVq+rSpYt++uknu7e0rFSpkvbu3atJkybpu+++U9euXZMFzjuNGTPG/GU8ZMgQeXp6qmjRojp79qz5XX/55ZfVp0+fe34f7qdq1arp008/1ejRo3Xy5EkNGDBA7u7u8vLyUq5cuXT58mXz/apYsaLmzJlz14/ToUOHqn///jp+/Lh5BYhr167p/Pnzqlq1qmrWrHlP98qtUqWKrFareZDNzIFRun3prEaNGpmXkUmptTF//vz69NNP1b9/f+3cuVPPP/+8ypUrJ4vFolOnTik2NlZOTk764IMPkl1KysPDw7xTzdq1a/XDDz+obNmyunTpksLCwtS0aVNt27Yt0610HTp00M8//6ybN2+qdOnSdgNtZo5bWaFkyZIaMWJEipenKVmypGbPnq0hQ4bop59+ko+Pj8qVK6eEhAQdP35ciYmJKl68uD7++OP7cl1ae2bOnCl3d3etWbNGCxcu1MKFC1W8eHEVLlxYMTExZheps7OzRowYoa5duyZb3sfHR7Vr19Yff/yhd999VwsWLFDBggV18uRJxcXFadCgQeaPw8xKz2ee3vUsWrRIN2/eVO7cuVPsgu7Tp4/Onj2rZcuWaezYsZoxY4ZKlCih0NBQszu8Zs2amjx5crLlBg4cqFOnTpl3c3ryySfl7u6u4OBg5ciRQz4+Puk++/i+txTmzJlTX3/9tf7zn/+oYsWKunXrloKDg80zKRctWqRPPvlErq6uunr1arIBqq+//rrmzp0rHx8fubi4KDg4WGFhYapYsaKGDx+utWvXJjtV3tPTUytWrFCvXr1Urlw5hYWFKTg4WK6urmrdurW+++47M4nbuLq66tNPP9WsWbPUqFEjxcXF6fDhw0pISFDdunX13nvv6cMPP0wz6KRXtWrVFBgYqE6dOqlw4cIKDg7WrVu31LVrV61YscLc4d95HUMfHx99//336tmzp0qVKqVTp07p9OnT8vLyUteuXRUYGJjuExkyqm7dutqyZYtGjRqlunXrysXFRceOHdOZM2dUvHhxdejQQYsXL9b777+f7rNcu3btqq+//loNGjRQ3rx5dezYMUVERKhatWqaMGGCli1bZv6aTnotsnLlymn16tXq0qWLvLy8dOHCBR0/flz58uVTx44dtXbtWvOSHjZvv/22pk6dqrp165pnSF+9elW1atXShAkTtGTJkky1Ah88eFB//vmn+d/hw4d1/fp1VahQQd26ddOKFSvMltb0atu2rb799lv5+fkpX758OnHihM6fP6/SpUurX79+2rBhw13jNfv06aPOnTurQIECCgkJydS9SO1xcXHRN998o9dee02GYSg4OFgeHh565ZVXFBgYaPfajsOGDdPMmTNVq1Ytc2B4mTJlNGPGDP3f//1fin9rypQpatCggZycnHTq1Kk0zzp1cXHR3LlzNWvWLDVs2FCxsbE6fPiw3Nzc1Lp1ay1cuFBvvfXWfetaz0r16tXTpk2b9Pbbb6tZs2YqVKiQLl68qCNHjshisahJkyaaNm2aVq5caXdcauPGjbVkyRI1a9ZM7u7uOn78uFxdXTV48GAtWbIkw91d9ti6eXPnzp2p62TeuZ4cOXKkelmPqlWrat26dRowYIB5r+yTJ0/qiSeeUEBAgFauXGm3jnr16mn16tV68cUXVbBgQR09elRubm4aNGjQPZ9M2KRJE7OFPrXaM3rcyirdunVLdahEw4YNzWNIiRIldOrUKfP+uyNGjNDatWuz7Mze9HBxcdG0adO0fPlyvfbaa6pcubJiY2N15MgRXbp0SWXKlDHvF33ncVu6vQ3Nnz9fQ4cOVfny5RUREaELFy6oXr16+u677+46DmRGej/ztFSuXNm8jFfLli3tXqdYut3q+84772j+/Plq3ry5nJycdPjwYd28eVPVq1fX+PHj9c0339z1nXZxcdFHH32kqVOnqkaNGgoPD9fZs2fN96J69erprtViZOSiTbjvnn32WUVGRuq77767byEPAADgTve9pRD/M2fOHLVu3Vqff/653ecPHDigyMhIOTs7P7BLzwAAAEiEwgeqUqVKOn78uD799FPzkik2R48eNe9r2LZt2yy56j0AAEB60X38ABmGoYEDB5rXZipatKg8PDwUGRlpnnlVq1Ytff7554RCAADwQBEKH7DExET9+OOP+u9//6tTp07p8uXLyp8/v5566in5+/urQ4cO6bobAgAAQFYiFAIAAOD+X6cQGRcZeVOJiWR1AMCjLUcOiwoWvH/3J0fGEAqzocREg1AIAAAeKM4+BgAAAKEQAAAAhEIAAACIMYUAAGQbCQnxiowMU3x8rKNLyVI5cuSUm1se5cmT/6G4J/njilAI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\n",
      "text/plain": [
       "<Figure size 576x576 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.subplots(figsize = (8,8))\n",
    "ax=sns.countplot(x='Pclass',hue='Survived',data=train_df)\n",
    "plt.title(\"Passenger Class Distribution - Survived vs Non-Survived\", fontsize = 25);"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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+P3331vtp1Kp8J///AdTpkxBv379MGDAANxyyy347rvvoNPpWm2/ePFiJCUl4cMPP8SOHTswefJkpKWlYfz48di4cWO7vwugqWtr3bp1AIBZs2bZ1LT/+OOP47PPPsNvv/0GDse2S+zIkSN47rnnMGHCBPTv39/6u3/yySdx8ODBNvfJzs7GSy+9hKlTp6J///645pprMGPGDCxatAhqtbrV9iaTCStXrsScOXMwatQopKWlYdSoUXj88cfx559/2hRnZ91+++3WmJt3A//+++9ISkrCmDFjWu2zb98+PProo5gwYQLS09MxdOhQzJkzBytWrIBer291jCNHjgBo+kaelJSEF198EUDTsI+kpCTcdtttyM7Oxh133IH09HQMHz7cOth9/PjxSEpKwq+//tpm/Hq9Hp9//jkmTZqE9PR0jBkzBi+99FKbXdod/UxA04B7y+fG8lm0XO8W9913X4vr33IN33HHHW0ec+vWrXjwwQcxbNgw69+1o2unK59bZ7B0x54+fRpVVVXtbhcWFoY1a9Zgx44dSEtLs77e/Hean5/f5r6Wv3Hze8rVro0BAwYgKSkJ27dvbzcmy9/qk08+AdD678+yLK677jokJSXhhx9+aPc4r776KpKSkvDcc8+1eF2j0eCLL77AjBkzMGDAAPTv3x/Tpk3DZ599htra2naPd/78eTz77LMYO3Ys+vbti2nTpmHFihVgWbbdfdpiufamT5/e7jbHjx9HUlIS+vfv36Jl197nli0s1yoALFiwANXV1XYfo7y8HO+//36LZ8hNN92Ezz//vM3fafNnSGVlJd5++22MHz8eaWlpGDFiBObPn9+pyTDNhyBd7R785JNPYuPGjfjoo49avG65d/z73/9ucz/L9Th+/PgWr8+ZMwdJSUnYs2cPvvjiC4wYMQL9+vXDjTfeiPnz53fqb37lfdRy7mHDhrV7PyktLUVKSkqbn92jR49i3rx51ufViBEj8Nhjj7V7XwOantk//vgjZs6ciQEDBmDo0KGYP38+8vLy2t2nPU5JEJv/kBEREdb/rqysxK233oo333wTBw8ehFgsRlJSEkQiES5duoTvv/8eM2fORFlZWYvjLVy4EM888wz2798PhmGQlJQEHo+H/fv347nnnrM+CC0KCgowc+ZM/PTTTygsLER0dDRiYmJQWFiIn376CTfddBPOnz/f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Hn30UZw+fRr33nsvRCIRBg8ejBEjRuDaa69t1UI9fvx4DBkyBEeOHMHHH3+Mjz/+GD179sSIESMwevRoDB8+3ObWtM6y3DyvVrssODgYDz74IL7++musWbMGa9asQWhoKIYNG4ZRo0ZhzJgxnW7N7Uzx6dDQ0HavHcs13NY4MGezXOMpKSntToYSiURISEhAVlYWcnNzWxUXtudz6yoBAQGYNm0apk2bBqBpTPfBgwexfft27N+/HyzLYtGiRUhNTW13IlBntHdtzJw5E5999hmOHj2K0tJSREZGAvin1a/5w60jkyZNwptvvomioiIcP37cmpBu3LgRBoMBQ4YMafEF1nKf/PXXX9udratSqQD8cy1otVrrl7/evXu3uU9ycjIYhrFrsoqfnx+mTp2KVatWYd26ddYHcFVVFfbu3QuGYTBjxgzr9p15bnVGXFwcnnnmGbzzzjvYuHEjrr/++na/OGo0GuszpfkEpytZEsC2JqBd7fNiac3qrJSUFKSkpODJJ59EY2MjTpw4gf3792Pt2rVQq9UoKCjAU0891WpMele0d93b+zfvyM0334yPP/4Y69ats34JMhqN1kmizb8YWfIRg8HQ5hdzCy6XC7PZjJycHPTt29f691IoFO3uk5KSYteXeYcXyu6IUqnE+++/jy1btrRo8fH390d6ejpMJlOb3ScffPABhg0bhl9//RWnT59GcXExVq9ejdWrV0MoFOK2227D888/b/1GlJKSgnXr1uGbb77B9u3bUV1djcOHD+Pw4cP4+OOPoVAosGDBAutMMMuDW6PR2NRE3larwtWabpvfjCxdjB2tNHLlzaN507QtA03b+ibXmQSo+ZJtly5darMg9JXMZjMyMzORlJR01VnMLMvixx9/xNKlS1FRUWF9nWEYJCQk4KabbrI+iJobM2YMVq9ejW+//Ra7d+9GfX099uzZgz179mDhwoW45ppr8Oabb1q/qPB4PCxduhQrVqzA77//jqysLOTk5CAnJwfLly+HWCzGgw8+iEceecQpK1QUFhZar3nLt+2OzJ8/H2lpaVi+fDmOHTsGpVKJ9evXY/369eDxeJgyZQpef/11u+ttdWapqo4e/pb3mrfuuorlM3G134Hls9RWF7M9n9uOtPfFMjU1Fa+99ppNx2hPfHw84uPj8a9//QtHjhzBo48+Co1Gg5UrVzo0QWzv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IYSIVpIgCiFi2r59+2hsbMTj8bB8+XKWLl2K3W5n8eLFvPLKK2RmZl597UsvvcR3v/tdpk6disvlYt68eRw6dAiv18uuXbtYvXo1iqKwZMkSjh07RigUuubjEyZMiOB3LIQQAycJohAipni9XhYuXAhAMBgkOTmZJUuWsHXrVubPn4/dbgfgRz/6EXBlD+L7nnrqKbZs2cJvf/tbTp8+TW9vL263m7Fjx6LX67n77ruZNWsWt912G5WVlTgcjms+LoQQ0U4SRCFETPnwHsT31dbWoijK1T87HA4cDkef13z961+npKSE2bNnc/vtt7N//35UVSUxMZGVK1dSV1fHjh07+P73v883v/lNvva1r33s40IIEc0kQRRCDAszZszgmWee4YEHHiA+Pp5f/epXqKpKaWkpcCVhPHjwIL///e+x2+3s3LmTxsZGQqEQGzdu5I9//CMvvPACkydPRlVVDh069LGPCyFEtJMEUQgxLMydO5eTJ09y7733AlBUVMQTTzzBunXrAEhMTOShhx7iS1/6ElarlczMTCZMmMC5c+e4++672bJlCwsWLMBqtWK323niiSfIzs6+5uNCCBHtFFVV1UgHIYQQQgghhg7ppCKEEEIIIfqQBFEIIYQQQvQhCaIQQgghhOhDEkQhhBBCCNGHJIhCCCGEEKIPSRCFEEIIIUQfkiAKIYQQQog+/j90hA74mtRtPAAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 720x576 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.subplots(figsize=(10,8))\n",
    "ax=sns.kdeplot(train_df.loc[(train_df['Survived'] == 0),'Pclass'],shade=True,color='r',label='Not Survived')\n",
    "ax.legend()\n",
    "ax=sns.kdeplot(train_df.loc[(train_df['Survived'] == 1),'Pclass'],shade=True,color='b',label='Survived')\n",
    "ax.legend()\n",
    "\n",
    "plt.title(\"Passenger Class Distribution - Survived vs Non-Survived\", fontsize = 25)\n",
    "labels = ['First', 'Second', 'Third']\n",
    "plt.xticks(sorted(train_df.Pclass.unique()),labels);"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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z89PlStX1rFjx8x9p3v37hoxYkS229/o+/Z2adCggaZPn64zZ86oY8eO6tmzp1q0aJHt+/h671krcto3WrZsqfnz5ys6OtqppcUhLi5Ov/32myTpySefzPaxbTab2rdvr0mTJmnx4sXq0aNHlnVSU1MVExOT5XEOHDig8PBwHT9+XB4eHrrnnntUqFAh7d+/X5MnT9bChQs1depUPfDAA06Pt3jxYo0YMUJpaWny9fXVv/71Lx0/flyjR49W7dq1rb5MucL57846/6WmpqpHjx7asmWLPDw8VKFCBZUtW1ZHjx7VkiVLtHTpUo0ZM8a8mvmajOs4evSoYbfbDbvdbvz222/XXPeVV14x7Ha78eCDDxoJCQnm/FdffdWw2+1G8+bNjUOHDjlts3r1aqN69eqG3W433nvvPXP+/v37DbvdbgQHB2d53gULFhiVK1c27Ha7sWXLFnP+oEGDDLvdbvz73/82kpKSzPkJCQlGp06dDLvdbjz//PNOj/XRRx8ZdrvdaNiwobF69Wpzfnp6uvH1118bVatWNQIDA401a9Y4bed4Tex2uzFu3DgjJSXF3M7xmHa73fjjjz/MbVJSUoymTZsadrvd6NWrl3HmzBlz2U8//WTUrFnT3G7evHnmslOnThm1a9c27Ha78frrrxvnz583lx0+fNjo2LGjYbfbjWeeecapxokTJ5qPN2DAAOPSpUuGYRhOz5uTyMhIc9uTJ09ed/1rcTzOr7/+as777bffDLvdblSuXNmYO3eucfnyZXPZyZMnjWeffdaw2+1GSEiI0zLH7zpmzBjz9zEMwzhy5IjRokULw263G2+88YY5f9WqVYbdbjfq1q1r7N6925yfnp5ufP7554bdbjeqVq2a69/RUbfdbjeOHj2aq20cf9fx48eb8+bNm2fY7XajQYMG5rzz588bwcHBht1uN5YsWeL0GL/88otRrVq1bJc5XqvMj391rc2aNTNOnTplGIZhJCUlmftr48aNDbvdbsyePTvb7apUqWLMnDnTyMjIMGt88cUXDbvdbtSuXdtpX8rud8os87Hk6tcuu33EMP7ehzt37uw0f/To0YbdbjceeughY9myZeb89PR0Y+bMmUbVqlUNu91ufPnll9k+z428b2+3l19+2amuwMBAo3Xr1sbIkSONJUuWXPP9mvk1jY2NzXYdx9848zHlevvG77//btjtdiMoKMjpeOMwderULMec7P7+x44dMwIDAw273W7s378/y+N89913ht1uNxo3bmzuY3/++afRvHlzw263G/369TNrMwzDiI+PN/r06WPY7XajRYsWRnJysrnsyJEj5vtn7Nix5vEhPT3d+OKLL5xe45xeq2u9vpz//uYO57+oqChzPzp+/LhTnaNGjTLsdrtRq1Yt83e7lpvu2ktJSdEff/yhkSNHauHChZKkHj16qHTp0pKuJN6NGzfKZrNp+PDhWT4JN2jQQK1atZIkp77/PXv2SJLuu+++LFdKtGvXTl26dFGbNm2cujocl+c/8cQTKlKkiDm/dOnSeu2119SgQQOnFpMzZ86Yif/TTz91+nTt4eGh5557Tj169JBhGGbz9dUaN26sIUOGmOOHPDw8NGjQIPn6+kqS+UlbkubNm6ejR4/q7rvv1qRJk5y6yx577DGNGTMm2+eYNm2azp07pyZNmmjMmDEqXry4uezee+/Vp59+qqJFi2rjxo1atWpVto/x6quvmmMwctNNd/78efPn3Kx/o3755Rd5e3urefPmeuqpp5y6AsuUKWO2EJ0+fdqpu87xN37qqaecxpTcc889evXVV9W4cWOVK1cuy/qOrgoHDw8P9e3bVy1btlTr1q2dft9bzbEvnjt37prrHTp0SJcuXZKvr6/5nnCoX7+++vTpo9DQUEtXRfXt29ccyFykSJFrjnfLLDw8XF27djWvOCxevLjGjRune++9V+fOnXPJ1UmnTp0yn3fMmDFq2bKluczDw0Ndu3Y1959PPvkkSzeLdGPv29vt/fff18svv2zuJ4ZhaN++ffr22281ZMgQPfroo3ruueduqKvhRmS3bzz88MOqUKGCUlNTzdbczBYtWiQp59Yoh3Llyqlu3bqSZA4VyO5x2rVrZ+5jc+bM0eHDh/Xggw9q0qRJTgPw/fz89PHHH6tcuXKKjY116gL6z3/+o0uXLql27doaNmyYeXzw8PBQnz59rlurFZz/7tzzn+P1atiwodPFLAULFtSwYcNUv359NW/e/LrHbekGx0h169bN7Bt1/Ktevbrat29vHtg6duzo1E3i6emplStXauvWrXrssceyPKZhGOZ9h1JSUsz5jvtw7N69W++//75iY2OdtnvzzTc1btw4p+ZaxzaOPtHMjxccHKz//Oc/Gj58uDlv1apVSk1NVaVK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ev755827BWd23333mSPzHQffRo0aycPDQwcPHjR3wKvNmDFDzz33nNq2bXvTn6gd92OJjo7O9gZiv/zyi9nXm5ljkOLMmTOzPXFcvHhR3bt3V7t27fTVV1/dVI2Z9e3bV35+fkpKStJrr72Wbc0OK1euNJv8+/bte827NJ89e9b8lJxTq1XmG0Q6Dn779u1TixYt1L17dyUkJGTZpkaNGmYQc7xOI0aMUKdOnTR16tQs63t5eZmDNW/HmJiFCxeaoe+FF1647vqzZs1S27ZtNXTo0GxbF0JCQsyfM58QbteX2BqGke2N8ZKSkrRgwQJJzoNNHa0C58+fzzZ4X+vmjo7fITddag0bNpSnp6cSEhL03XffZbvOzJkzlZ6ersKFC9/2+wdZdc8996hGjRqSrtxT7Vqf1DMyMsyvGco8aLlEiRLma5fd+JAff/zxpsKDt7e32rRpI0n67rvv9MMPP2QZHJ5bji6877//3rx31FNPPZVlPceHm7lz52bbdZyenq7+/furQ4cOTucHR6vBrFmzsn0/53TT2duJ89/f8tv5r1evXuratat5LMusSJEieuihhyTl7txw24OU49Pj+fPn9dVXXzkdKP/3v/+pZ8+eZt9p5j/UE088IZvNpp9//llTp0516ls+ceKE2bXh+AqOkiVLmpdvjhgxwmkQbEZGhiIjI80bojnWK1eunHmzrSFDhujHH3902mbOnDnmV3t07do1S//tjerQoYPKli2rY8eOaciQIU6XVW7cuNG8I/XV+vTpIx8fH23atElDhw51urrg+PHj6tOnj86ePatixYple5WVVcWKFdOYMWPk5eWln376SV27dtUvv/zi9DdMSkrSp59+qkGDBskwDIWEhKhbt27XfNy77rpLJUqUkHTljZr5U0RiYqLeeustpyt8HAfTypUry2636/LlyxoyZIjTJ7nU1FRFREQoKSlJPj4+5jgAx1Uns2bN0sKFC51q37dvn9lycPV3CN6Ms2fP6tNPPzWb3Nu3b5/t1S1Xe/zxx+Xl5aW9e/fq3XffdeqSSUxMNG9uV716daeuDMcn+uwOQjdr/Pjx5klPuvIp7t///rfi4uJ0zz33qEOHDuYyxxfWGoahd9991/y7paWl6auvvtLs2bNzfB7H73DixInr1lS2bFlz3MIbb7zhVF9GRoaioqLMbuP+/fvn2+8RlKThw4ercOHC2rt3rzp27KiVK1dmaVk6cOCA+vfvr02bNqlixYpO7/FChQqZN1icNGmS00DdNWvWaPTo0TddoyMA/ec//1FSUpLq1KnjdIuR3Grbtq08PT21ePFiHTx4UPfee6/5Ps2sa9eu8vPz0+HDh9WvXz+nfSIxMVGDBg3SgQMH5OXlpeeff95c1qtXL5UoUcLsMnUEU8MwFBUVleNA5duJ89/f8tv5z3Fu+OSTT7R69WqnZRs3bjRbonJzbrjtY6SaNGmiGjVqaMuWLXr33Xc1depUBQQEKCEhQXFxcbLZbAoJCdHatWsVHx8vwzBks9kUFBSkQYMGKSIiQh999JGmTJmi8uXLKzk5WUePHlV6erruvfdepxffcRfsvXv3qk2bNipfvryKFSumEydOmF+/MGTIEKcrK0aMGKG4uDj99NNP6tevn/z9/RUQEKDjx4+bf7DQ0FANGjTopl+LokWL6uOPP1bPnj21cuVKrV69Wv/617/0559/KjY2VuXKlVPp0qV1+vRpp8t+K1SooAkTJmjw4MFasmSJli9frkqVKiktLU2xsbFKT0+Xj4+PpkyZct07Qt+oxo0ba9q0aRo4cKC2b9+u3r17q3jx4ipfvrzS09N16NAh803epk0bvfPOO9lespyZp6enXnrpJY0aNUobNmxQo0aNVLFiRaWmpurw4cNKT09X1apVdfLkSZ09e1anTp0yW64iIiLMwbfNmjVT+fLlVbhwYR07dkwXLlyQh4eHRo8ebXZ/tGjRQk8//bRmz56tV199Ve+//77Kli2rpKQkHTlyRIZhqFq1arlqMbraSy+95NTylpqaqnPnzun48ePmAfPpp5/Wm2++mavH8/f317vvvquhQ4fq66+/1ty5c3Xvvffq8uXLOnLkiC5duqSSJUuadz53qFq1qn766SdFR0drz549evjhhzVy5Mgb/n2uVq5cOd1111166aWXdPfdd6tkyZLat2+fUlNT5efnp8mTJzt1y/j6+qpXr176/PPPtWTJEv3yyy8qX768jh8/rnPnzqlLly768ccfs/06jKpVq+r333/X6NGj9e233+qZZ55xCmlXGz58uOLi4vTDDz/opZdekr+/v8qUKaOjR4+a7/Vnn31W4eHhN/063E7Vq1fXZ599pmHDhungwYMaMGCAfHx8VK5cORUqVEjx8fHm61WlShVNmjQpywlt0KBB6tevn/bv329eOXb+/HkdP35cwcHBqlmzZo5Xv+VGUFCQ7Ha7OdjX6v2YSpcurQYNGpiXqOfUquXr66vPPvtM/fr109q1a9W0aVNVqlRJNptNhw4dUmpqqjw9PTV+/Hin25b4+flpwoQJevHFF7Vo0SJ9//33euCBB3Tq1CklJCSoSZMmWrVqVZ5ekcn572/57fzXrVs3rV27VqtXr1Z4eLj8/f3l7++vs2fPmh9KmzRpkqs7m9/2FikPDw/NmDFD//d//6cqVaooOTlZe/fuNa9Amjlzpj799FMVLFhQ586dcxq8+cILL2jy5Mlq1KiRvL29tXfvXiUkJKhKlSoaMmSIFi1a5HT5pb+/v+bOnatevXqpUqVKSkhI0N69e1WwYEG1bt1a3377rfr06eNUX8GCBfXZZ58pIiJCDRo0UFpamnbt2qXLly+rTp06ev/99zVhwoTrhoPcql69uhYvXqwOHTqoVKlS2rt3r5KTk/XMM89o7ty55kHy6vtsNGrUSEuXLlWPHj1077336tChQzp8+LDKlSunZ555RosXL871YOYbVadOHa1YsUKvvPKK6tSpI29vb+3bt09HjhzR3XffrSeffFKRkZEaN25crq8Oe+aZZzRjxgw9+uijKlasmPbt26czZ86oevXqevPNNzV79mzzk0Dme4NUqlRJCxYsUJcuXVSuXDmdOHFC+/fvV/HixfXUU09p0aJF5uXjDqNGjdLYsWNVp04d88rCc+fOqVatWnrzzTcVFRVl6dPWjh07tHnzZvPfrl27dOHCBVWuXFldu3bV3LlzzRa93HriiSf0zTffKDQ0VMWLF9eBAwd0/PhxVahQQX379tV3332XZfxZeHi4OnbsqBIlSig2NvaaYxBuhLe3t7766is9//zzMgxDe/fulZ+fn7p3767Fixdne++twYMH66OPPlKtWrWUlpamQ4cO6b777tOHH36ot956K8fnevfdd/Xoo4/K09NThw4duu7VWt7e3po8ebIiIiJUv359paamateuXSpcuLBat26tr7/+Wm+88cZt6/a8lerVq6eYmBiNGjVKzZo101133aWTJ09q9+7dstlsaty4sd577z3Nmzcv23F2DRs2VFRUlJo1ayYfHx/t379fBQsW1MCBAxUVFXXNq6Ryy9EFV6RIEUv3Mbv6cQoUKKB27drluF5wcLCio6M1YMAA8ytLDh48qNKlS6tdu3aaN29etnXUq1dPCxYsUKdOnVSyZEnt2bNHhQsX1r///e88GVB9Nc5/zvLT+c/Dw0OTJ0/WiBEjVKNGDXOManJysurXr68PPvhAn376aa4uHLEZVm+qgduibt26Onv2rL799tvbFowAAMhv7tTz321vkcLfJk2apNatW5tf1Hu1bdu26ezZs+aXOQIA4A7c+fxHkMpDVatW1f79+/XZZ5+Zl+c77NmzR0OHDpV0pXvnZgf2AQCQX7jz+Y+uvTxkGIZefPFF87uDypQpIz8/P509e9a8TLZWrVqaMmXKHbcjAQCQE3c+/xGk8lhGRoZ++OEH/fe//9WhQ4cUHx8vX19f3X///QoLC9OTTz6Z6+/3AQDgTuGu5z+CFAAAgEWMkQIAALDozmtDc6GzZ/9URgYNeAAA91aggE0lS96+L/x2JwSpG5CRYRCkAACAia49AAAAiwhSAAAAFhGkAAAALCJIAQAAWESQAgAAsIggBQAAYJFb3f4gOjpan332mdLT09W9e3d17drVXLZr1y4NGzbMnE5MTJSvr6+WLFniilIBAIAbcJsgFRcXp4iICM2fP1/e3t7q3Lmz6tSpo0qVKkmSqlSpokWLFkmSkpOT1bFjR7311lsurBgAANzp3KZrb+3atapbt65KlCghHx8fhYaGKiYmJtt1v/jiCz3yyCN6+OGH87hKAADgTtymRSo+Pl5+fn7mtL+/v7Zt25ZlvYsXL2r27NmKjo7Oy/IAAIAbcpsglZGRIZvNZk4bhuE07bB48WI1a9ZMpUqVuuHnKFWq6E3VCAAA3IvbBKkyZcpo48aN5nRCQoL8/f2zrLdy5Ur17dvX0nOcOZPEd+0BANxegQI2Gg9yyW3GSIWEhGjdunVKTExUcnKyVqxYoYYNGzqtYxiGdu7cqRo1arioSve2efNGjRr1mjZv3nj9lQEAcANuE6QCAgI0ePBgdevWTe3atVObNm1UrVo1hYeHa/v27ZKu3PLAy8tLBQsWdHG17mnOnCjt2rVTc+ZEuboUAADyhM0wDPqqcomuvWsbNKi/Tp06oTJl7taECZ+6uhwAgEV07eWe27RIAQAA5DWCFAAAgEUEKQAAAIsIUgAAABYRpAAAACwiSAEAAFhEkAIAALCIIAUAAGARQQoAAMAighQAAIBFBCkAAACLCFIAAAAWEaQAAAAsIkgBAABYRJACAACwiCAFAABgEUEKAADAIoIUAACARQQpAAAAiwhSAAAAFhGkAAAALCJIAQAAWESQAgAAsIggBQAAYBFBCgAAwCKCFAAAgEUEKQAAAIs8XV3AP0Gx4oVUqKCXq8u47Tw8bOb/fn7FXFxN3ki5lKaLF1JcXQYAwEUIUnmgUEEvPfNKpKvLuO1On74oSTp1+uI/4veVpKgPuuqiCFIA8E9F1x4AAIBFBCkAAACLCFIAAAAWEaQAAAAsIkgBAABYRJACAACwiCAFAABgkVsFqejoaLVq1UotWrRQZGTW+xgdPHhQzz33nJ544gn16tVL58+fd0GVAADAXbhNkIqLi1NERISioqK0cOFCzZo1S/v37zeXG4ahfv36KTw8XIsXL1aVKlU0ZcoUF1YMAADudG4TpNauXau6deuqRIkS8vHxUWhoqGJiYszlO3fulI+Pjxo2bChJeuGFF9S1a1dXlQsAANyA2wSp+Ph4+fn5mdP+/v6Ki4szp48cOaLSpUtrxIgRat++vUaOHCkfHx9XlAoAANyE23zXXkZGhmw2mzltGIbTdHp6ujZs2KCZM2cqODhYEyZM0Hvvvaf33nsv189RqlTRW1oz3MM/5QuaAQBZuU2QKlOmjDZu3GhOJyQkyN/f35z28/NThQoVFBwcLElq06aNBg4ceEPPceZMkjIyjBuujROte0tIuOjqEgDglipQwEbjQS65TddeSEiI1q1bp8TERCUnJ2vFihXmeChJqlGjhhITE7V7925J0o8//qgHH3zQVeUCAAA34DYtUgEBARo8eLC6deumtLQ0dejQQdWqVVN4eLgGDhyo4OBgTZ48Wa+//rqSk5NVpkwZffDBB64uGwAA3MHcJkhJUlhYmMLCwpzmTZ061fy5evXqmjt3bl6XBQAA3JTbdO0BAADkNYIUAACARQQpAAAAiwhSAAAAFhGkAAAALCJIAQAAWESQAgAAsIggBQAAYBFBCgAAwCKCFAAAgEUEKQAAAIsIUgAAABYRpAAAACwiSOGWsXl4Of0PAIC7I0jhlil6d015FS2jonfXdHUpAADkCU9XFwD3UdD3HhX0vcfVZQAAkGdokQIAALCIIAUAAGARQQoAAMAighQAAIBFBCkAAACLCFIAAAAWEaQA5FubN2/UqFGvafPmja4uBQCyxX2kAORbc+ZE6dChg0pJSVbNmg+7uhwAyIIWKQD5VnJyitP/AJDfEKQAAAAsIkgBAABYRJACAACwiCAFAABgEUEKAADAIoIUAACARQQpAAAAiwhSAAAAFhGkAAAALCJIAQAAWESQAgAAsMitglR0dLRatWqlFi1aKDIyMsvyTz75RI0bN1bbtm3Vtm3bbNcBAADILU9XF3CrxMXFKSIiQvPnz5e3t7c6d+6sOnXqqFKlSuY6O3bs0Pjx41WjRg0XVgoAANyF27RIrV27VnXr1lWJEiXk4+Oj0NBQxcTEOK2zY8cOffHFFwoLC9Po0aN16dIlF1ULAADcgdsEqfj4ePn5+ZnT/v7+iouLM6f//PNPValSRUOHDtWCBQt04cIFffrpp64oFQAAuAm36drLyMiQzWYzpw3DcJouUqSIpk6dak4///zzGjFihAYPHpzr5yhVquitKRZuxc+vmKtLcFseHjbzf15nAPmR2wSpMmXKaOPGjeZ0QkKC/P39zekTJ05o7dq16tChg6QrQcvT88Z+/TNnkpSRYdxwbZwA3FtCwkVXl+C2Ll82zP95nYG8U6CAjcaDXHKbrr2QkBCtW7dOiYmJSk5O1ooVK9SwYUNzeaFChfThhx/q6NGjMgxDkZGRat68uQsrBgAAdzq3CVIBAQEaPHiwunXrpnbt2qlNmzaqVq2awsPDtX37dt11110aPXq0+vXrp5YtW8owDPXs2dPVZQMAgDuY23TtSVJYWJjCwsKc5mUeFxUaGqrQ0NC8LgsAALgpt2mRAgAAyGsEKQAAAIsIUgAAABYRpAAAACwiSAEAAFhEkAIAALCIIAUAAGCRW91HCvinKOnrLU/vgq4u47b7J37XXnrqJZ09n+rqMgDkEkEKuAN5ehfUpg96u7qM2+7S2Tjz/3/C7ytJtV75jySCFHCnoGsPAADAIoIUAACARQQpAAAAiwhSAAAAFhGkAAAALCJIAQAAWESQAgAAsIggBQAAYBFBCgAAwCKCFAAAgEUEKQAAAIsIUgAAABYRpAAAACwiSAEAAFhEkAIAALCIIAUAAGARQQoAAMAighQAAIBFBCkAAACLCFIAAAAWEaQAAAAsIkgBAABYRJACAACwiCAFAABgEUEKAADAIoIUAACARZ6uLiCzJk2ayGaz5bj8hx9+uOb20dHR+uyzz5Senq7u3bura9eu2a73888/a/To0frxxx9vql4AAPDPlq+C1MSJEyVJUVFR8vLyUqdOneTh4aH58+crLS3tmtvGxcUpIiJC8+fPl7e3tzp37qw6deqoUqVKTuudPn1a77///m37HQDcOgU9Czj9DwD5Tb46OgUFBSkoKEj79u3TqFGjVLVqVQUGBmr48OHatm3bNbddu3at6tatqxIlSsjHx0ehoaGKiYnJst7rr7+uF1988Xb9CgBuoRaVSur+koXUolJJV5cCANnKVy1SDhcuXFBiYqLuuusuSVdam5KSkq65TXx8vPz8/Mxpf3//LOHr66+/VtWqVVW9evVbXzSAW66Kn4+q+Pm4ugwAyFG+DFLdu3dXWFiY6tevL8Mw9Ouvv2ro0KHX3CYjI8NpfJVhGE7Te/fu1YoVKzRjxgydOnXKUl2lShW1tB3cm59fMVeXADfDPgXcOfJlkHrmmWdUs2ZNrVu3TpLUu3dv2e32a25TpkwZbdy40ZxOSEiQv7+/OR0TE6OEhAQ99dRTSktLU3x8vJ555hlFRUXluq4zZ5KUkWHc4G/DQdHdJSRczPPnZJ9yb67Yp4DMChSw0XiQS/lqjFRmsbGxOnfunDp16qS9e/ded/2QkBCtW7dOiYmJSk5O1ooVK9SwYUNz+cCBA7V8+XItWrRIU6ZMkb+//w2FKAAAgKvlyyA1ZcoUffvtt4qJidGlS5f0ySefaPLkydfcJiAgQIMHD1a3bt3Url07tWnTRtWqVVN4eLi2b9+eR5UDAIB/knzZtbd06VLNmTNHTz/9tEqWLKnZs2erU6dOGjBgwDW3CwsLU1hYmNO8qVOnZlmvfPny3EMKAADctHzZIuXp6Slvb29zunjx4vL0zJeZDwAA/IPly3RStmxZ/fzzz7LZbEpNTdW0adNUrlw5V5cFAADgJF8GqTfeeEOvvPKK9uzZo4ceekjVq1fXuHHjXF0WAACAk3wZpHx8fPTVV18pOTlZly9fVtGiXIIJAADyn3w5Rqpp06Z65ZVXtHPnTkIUAADIt/JlkPrhhx9Uo0YNvf/++2rZsqWmTZumxMREV5cFAADgJF8GqWLFiqlLly6aM2eOJkyYoOXLl6tRo0auLgsAAMBJvhwjJUk7d+7UggULFBMTo6CgIH388ceuLgkAAMBJvgxSYWFhSk5O1pNPPql58+YpICDA1SUBAABkkS+D1LBhw/Too4+6ugwAAIBryldBaurUqQoPD9ePP/6on376Kcvy119/3QVVAQAAZC9fBalixYpJkkqWLOniSgAAAK4vXwWpzp07S5JKly6tNm3acA8pAACQr+XL2x+sX79ezZo104gRI7RlyxZXlwMAAJCtfNUi5RAREaHz589ryZIleuedd5SSkqKOHTuqe/furi4NAADAlC9bpCTJ19dXnTp1Ut++feXj46OpU6e6uiQAAAAn+bJF6o8//tC8efMUExOjqlWrqnfv3mrSpImrywIAAHCSL4NU//791aFDB82ZM0d33323q8sBAADIVr4MUrVq1dKLL77o6jIAAACuKV+Okdq3b58Mw3B1GQAAANeUL1uk/Pz81Lp1a1WvXl1FihQx53NncwAAkJ/kyyBVo0YN1ahRw9VlAAAAXFO+DFKMjwIAAHeCfBmkwsLCsp0fHR2dx5UAAADkLF8GqTfeeMP8OS0tTUuXLtU999zjwooAAACyypdBqnbt2k7TISEh6ty5s/r16+eiigAAALLKl7c/uNrZs2cVHx/v6jIAAACc5MsWqavHSJ04cUKdOnVyUTUAAADZy3dByjAMDRs2TF5eXrp48aJ2796tZs2aKTAw0NWlAQAAOMlXXXv79+9X06ZNlZqaqmrVqumjjz7SkiVL1Lt3b/3666+uLg8AAMBJvgpSH3zwgQYNGqTGjRtr6dKlkqSlS5dq9uzZmjRpkourAwAAcJavgtTJkyf1xBNPSJLWr1+vpk2bqkCBAipbtqySkpJcXB0AAICzfBWkChT4u5wtW7bokUceMacvXbrkipIAAABylK8Gm/v6+mr37t1KSkpSQkKCGaQ2b96sgIAAF1cHAADgLF8FqSFDhqhHjx5KSkrS//3f/8nHx0fTpk3T559/rsmTJ7u6PAAAACf5Kkg99NBDWr16tVJSUlS8eHFJUo0aNTRnzhxVrFjRtcUBAABcJV+NkZIkb29vM0RJUs2aNXMdoqKjo9WqVSu1aNFCkZGRWZZ///33CgsLU+vWrTVs2DClpqbeqrIBAMA/UL4LUlbFxcUpIiJCUVFRWrhwoWbNmqX9+/eby//66y+NHj1aX375pZYuXapLly5pwYIFLqwYAADc6dwmSK1du1Z169ZViRIl5OPjo9DQUMXExJjLfXx89OOPP6p06dJKTk7WmTNnnFq+AAAAbpTbBKn4+Hj5+fmZ0/7+/oqLi3Nax8vLS6tWrdJjjz2ms2fPqn79+nldJgAAcCP5arD5zcjIyJDNZjOnDcNwmnZo1KiR1q9fr/Hjx+utt97SuHHjcv0cpUoVvSW1wr34+RVzdQlwM+xTwJ3DbYJUmTJltHHjRnM6ISFB/v7+5vS5c+e0Y8cOsxUqLCxMgwcPvqHnOHMmSRkZxg3XxkHRvSUkXMzz52Sfcm+u2KeAzAoUsNF4kEtu07UXEhKidevWKTExUcnJyVqxYoUaNmxoLjcMQ0OHDtWJEyckSTExMapZs6arygUAAG7AbVqkAgICNHjwYHXr1k1paWnq0KGDqlWrpvDwcA0cOFDBwcEaM2aM+vbtK5vNpkqVKmnUqFGuLhsAANzB3CZISVe668LCwpzmTZ061fy5WbNmatasWV6XBQAA3JTbdO0BAADkNYIUAACARQQpAAAAiwhSAAAAFhGkAAD/GJs3b9SoUa9p8+aN118ZyAW3umoPAIBrmTMnSocOHVRKSrJq1nzY1eXADdAiBQD4x0hOTnH6H7hZBCkAAACLCFIAAAAWEaQAAAAsIkgBAABYRJACAACwiCAFAABgEUEKAADAIoIUAACARQQpAAAAiwhSAAAAFhGkAAAALCJIAQAAWESQAgAAsIggBQAAYJGnqwsAALhWcd+CKujt7eoy8oSHh83838+vmIuruf0upabqwvlLri7DrRGkAOAfrqC3t3p8+ZKry8gTcRcSzP//Cb/zjJ4fSyJI3U507QEAAFhEkAIAALCIIAUAAGARQQoAAMAighQAAIBFBCkAAACLCFIAAAAWEaQAAAAsIkgBAABYRJACAACwiCAFAABgEUEKAADAIoIUAACARW4VpKKjo9WqVSu1aNFCkZGRWZavXLlSbdu21RNPPKH+/fvr/PnzLqgSAAC4C7cJUnFxcYqIiFBUVJQWLlyoWbNmaf/+/ebypKQkvfXWW5oyZYoWL16swMBATZo0yYUVAwDyms2rgNP/wM1ymz1p7dq1qlu3rkqUKCEfHx+FhoYqJibGXJ6WlqaRI0cqICBAkhQYGKiTJ0+6qlwAgAv4VgtQwYAi8q0W4OpS4CbcJkjFx8fLz8/PnPb391dcXJw5XbJkSTVv3lySlJKSoilTpqhZs2Z5XicAwHUKly8m/+b3qXD5Yq4uBW7C09UF3CoZGRmy2WzmtGEYTtMOFy9e1IABA1S5cmW1b9/+hp6jVKmiN10n3I+fHwdk3FrsU7iV2J9uL7cJUmXKlNHGjRvN6YSEBPn7+zutEx8fr169eqlu3boaMWLEDT/HmTNJysgwbng7dmL3lpBwMc+fk33KveX1PsX+5N6s7E8FCthoPMglt+naCwkJ0bp165SYmKjk5GStWLFCDRs2NJdfvnxZL7zwgh5//HG99tpr2bZWAQAA3Ai3aZEKCAjQ4MGD1a1bN6WlpalDhw6qVq2awsPDNXDgQJ06dUp//PGHLl++rOXLl0uSgoKC9M4777i4cgAAcKdymyAlSWFhYQoLC3OaN3XqVElScHCwdu/e7YqyAACAm3Kbrj0AAIC8RpACAACwiCAFAABgEUEKAADAIoIUAACARQQpAAAAiwhSAAAAFhGkAAAALCJIAQAAWESQAgAAsIggBQAAYBFBCgAAwCKCFAAAgEUEKQAAAIsIUgAAABYRpAAAACwiSAEAAFhEkAIAALCIIAUAAGARQQoAAMAighQAAIBFBCkAAACLCFIAAAAWEaQAAAAsIkgBAABYRJACAACwiCAFAABgEUEKAADAIoIUAACARQQpAAAAiwhSAAAAFhGkAAAALCJIAQAAWESQAgAAsIggBQAAYBFBCgAAwCK3ClLR0dFq1aqVWrRoocjIyBzXe+WVVzR//vw8rAwAALgjtwlScXFxioiIUFRUlBYuXKhZs2Zp//79WdZ54YUXtHz5chdVCQAA3InbBKm1a9eqbt26KlGihHx8fBQaGqqYmBindaKjo9W0aVM9/vjjLqoSAAC4E09XF3CrxMfHy8/Pz5z29/fXtm3bnNbp3bu3JGnTpk15WhsAAHBPbhOkMjIyZLPZzGnDMJymb4VSpYre0seDe/DzK+bqEuBm2KdwK7E/3V5uE6TKlCmjjRs3mtMJCQny9/e/pc9x5kySMjKMG96Ondi9JSRczPPnZJ9yb3m9T7E/uTcr+1OBAjYaD3LJbcZIhYSEaN26dUpMTFRycrJWrFihhg0burosAADgxtwmSAUEBGjw4MHq1q2b2rVrpzZt2qhatWoKDw/X9u3bXV0eAABwQ27TtSdJYWFhCgsLc5o3derULOu99957eVUSAABwY27TIgUAAJDXCFIAAAAWEaQAAAAsIkgBAABYRJACAACwiCAFAABgEUEKAADAIoIUAACARQQpAAAAiwhSAAAAFhGkAAAALCJIAQAAWESQAgAAsIggBQAAYBFBCgAAwCKCFAAAgEUEKQAAAIsIUgAAABYRpAAAACwiSAEAAFhEkAIAALCIIAUAAGARQQoAAMAighQAAIBFBCkAAACLCFIAAAAWEaQAAAAsIkgBAABYRJACAACwiCAFAABgEUEKAADAIoIUAACARQQpAAAAiwhSAAAAFhGkAAAALHKrIBUdHa1WrVqpRYsWioyMzLJ8165devLJJxUaGqrXXntN6enpLqgSAAC4C7cJUnFxcYqIiFBUVJQWLlyoWbNmaf/+/U7rDB06VG+++aaWL18uwzA0e/ZsF1ULAADcgdsEqbVr16pu3boqUaKEfHx8FBoaqpiYGHP58ePHlZKSooceekiS9OSTTzotBwAAuFGeri7gVomPj5efn5857e/vr23btuW43M/PT3FxcTf0HAUK2CzXV7pkEcvbIn+7mf3iZngXL+WS58Xt54p9qnTRu/L8OZE3rOxPrjqu3YncJkhlZGTIZvv7D28YhtP09ZbnRsmbCEMTh7ezvC3yt1KlirrkeYNfeN8lz4vbzxX71EcdR+b5cyJvuOoY9U/hNl17ZcqUUUJCgjmdkJAgf3//HJefPn3aaTkAAMCNcpsgFRISonXr1ikxMVHJyclasWKFGjZsaC4vV66cChYsqE2bNkmSFi1a5LQcAADgRtkMwzBcXcStEh0drS+++EJpaWnq0KGDwsPDFR4eroEDByo4OFi7d+/W66+/rqSkJD344IMaO3asvL29XV02AAC4Q7lVkAIAAMhLbtO1BwAAkNcIUgAAABYRpAAAACwiSAEAAFhEkAIAALCIIIVbJikpSW3atNGxY8dcXQrcwCeffKLWrVurdevW+uCDD1xdDtzAxx9/rFatWql169b68ssvXV0O3ARBCrfE1q1b1aVLF8XGxrq6FLiBtWvXas2aNVqwYIEWLlyonTt36vvvv3d1WbiDbdiwQb/99psWL16sefPm6ZtvvtHBgwddXRbcAEEKt8Ts2bM1cuRIvnYHt4Sfn5+GDRsmb29veXl56YEHHtCJEydcXRbuYLVr19bXX38tT09PnTlzRpcvX5aPj4+ry4IbcJsvLYZrvfPOO64uAW7kX//6l/lzbGysli1bpm+//daFFcEdeHl5aeLEiZo+fbpatmypgIAAV5cEN0CLFIB8a9++fXr++ef1yiuvqGLFiq4uB25g4MCBWrdunU6ePKnZs2e7uhy4AYIUgHxp06ZN6tGjh15++WW1b9/e1eXgDnfgwAHt2rVLklS4cGG1aNFCe/bscXFVcAcEKQD5zsmTJzVgwAB99NFHat26tavLgRs4duyYXn/9daWmpio1NVU//PCDatWq5eqy4AYYIwUg35k2bZouXbqk9957z5zXuXNndenSxYVV4U7WqFEjbdu2Te3atZOHh4datGhBSMctYTMMw3B1EQAAAHciuvYAAAAsIkgBAABYRJACAACwiCAFAABgEUEKAADAIm5/AOC2OXbsmJo3by673W7OMwxD3bp1U4cOHbLdZv78+Vq+fLm++OKLvCoTACwjSAG4rQoVKqRFixaZ03FxcWrTpo2CgoJUuXJlF1YGADePIAUgTwUEBKhChQqKjY3VqlWrtGDBAnl6eqpChQpON+CUpP/973/68MMPlZqaqoSEBIWEhOjdd99Venq6xowZo82bN8vLy0vly5fX2LFjVbBgwWznFylSxEW/LQB3R5ACkKe2bNmiI0eOKDk5WfPnz9fs2bPl6+ursWPHaubMmQoICDDX/frrrzVw4EDVqVNHf/75p5o2baodO3YoJSVFGzZs0HfffSebzaYPP/xQe/bsUUZGRrbza9as6cLfGIA7I0gBuK1SUlLUtm1bSdLly5dVsmRJffjhh/rll1/UsmVL+fr6SpKGDx8u6coYKYf33ntPq1ev1ueff66DBw/q0qVL+uuvv1S5cmV5eHioY8eOql+/vkJDQ1WtWjVduHAh2/kAcLsQpADcVlePkXJYu3atbDabOX3hwgVduHDBaZ1nn31WgYGBatCggR5//HFt3bpVhmGoePHiWrRokTZv3qzffvtNgwYNUq9evdS1a9cc5wPA7UCQAuASISEh+uCDD9S7d28VLVpUkyZNkmEYqlq1qqQrwWr79u2aOnWqfH19tX79eh05ckQZGRn66aefNH36dH355Zd65JFHZBiGduzYkeN8ALhdCFIAXKJRo0bav3+/unTpIkmqVKmSxowZoxUrVkiSihcvrj59+qh9+/by8fFRQECAatasqcOHD6tjx45avXq12rRpIx8fH/n6+mrMmDEqW7ZstvMB4HaxGYZhuLoIAACAOxF3NgcAALCIIAUAAGARQQoAAMAighQAAIBFBCkAAACLCFIAAAAWEaQAAAAsIkgBAABY9P+2voz9mihqLQAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 576x432 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.subplots(figsize = (8,6))\n",
    "sns.barplot(x='Pclass', y='Survived', data=train_df);\n",
    "plt.title(\"Passenger Class Distribution - Survived Passengers\", fontsize = 25);"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The graphs above clearly shows that __economic status (Pclass)__ played an important role regarding the potential survival of the Titanic passengers. First class passengers had a much higher chance of survival than passengers in the 3rd class. We note that:\n",
    "\n",
    "- 63% of the 1st class passengers survived the Titanic wreck\n",
    "- 48% of the 2nd class passengers survived\n",
    "- Only 24% of the 3rd class passengers survived"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id=\"section307\"></a>\n",
    "### 3.7 Correlation Matrix and Heatmap"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Look at numeric and categorical values separately \n",
    "df_num = train_df[['Age','SibSp','Parch','Fare']]\n",
    "df_cat = train_df[['Survived','Pclass','Sex','Ticket','Cabin','Embarked']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "data": {
      "image/png": "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RJ+mrYkib4ZIREREREqxokhERESkomI5R5GIiIiIPk1bh56ZKBIRERGpiIkiERERESklLcLPa1YFF7MQERERkVKsKBIRERGpiItZiIiIiEgpzlEkIiIiIqWYKBIRERGRUto69MzFLERERESkFCuKRERERCqScuhZdR0C+6nzdPQF0Is6qukQqAjK/H2wpkOgIqhE93WaDoGKoOxrTTUdAgDOUSQiIiKij9DWRJFzFImIiIi+IIcOHUKHDh3Qtm1b7NixQ2F/VFQUPD094e7ujsmTJyMnJyff52KiSERERKQiQSpV6ZVXMTExWLp0KXbu3IkDBw4gKCgIDx8+lOszbtw4TJs2DaGhoRAEAcHBwfm+LiaKRERERCoSJFKVXikpKYiOjlZ4paSkyJ3n/PnzaNq0KczMzGBsbAx3d3ccP35ctv/58+fIzMyEk5MTAMDT01Nu/+fiHEUiIiIiFak6R3HLli0IDAxUaPfz88PIkSNl27GxsbCwsJBtW1pa4ubNmx/db2FhgZiYmHzHxUSRiIiISEVSFW+43a9fP3Tt2lWh3dTUVOE8IpFIti0Igtz2p/Z/LiaKRERERBpmamqqkBQqY21tjStXrsi24+LiYGlpKbc/Li5Oth0fHy+3/3NxjiIRERGRilSdo5hXzs7OuHDhAhISEpCRkYGwsDC4uLjI9tvY2MDQ0BBXr14FAISEhMjt/1ysKBIRERGpSJBI1HIeKysr+Pv7w9fXF2KxGN26dUO9evUwaNAgjBo1Co6Ojli0aBGmTJmC1NRU2Nvbw9fXN9/nY6JIREREpKLPucWNqjw8PODh4SHXtn79etmf7ezssGfPngI5FxNFIiIiIhXxySxEREREVKywokhERESkIm2tKDJRJCIiIlKRlIkiERERESmjzsUs6sQ5ikRERESkFCuKRERERCriHEUiIiIiUkqQCJoOoVAwUSQiIiJSERezEBEREZFSglQ7K4pczEJERERESrGiSERERKQiKecoEhEREZEyXPVMREREREpx1TMRERERKaWtQ89czEJERERESrGiSERERKQizlEkIiIiIqWkWnofRSaKRERERCrS1sUsnKNIREREREqxokhERESkIj7rmYiIiIiU0tahZyaKRERERCpiokhERERESmnr0DMXsxARERGRUqwoEhEREalI4H0USZk/7j3FirAryJZIUcuq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\n",
      "text/plain": [
       "<Figure size 864x432 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.subplots(figsize = (12,6))\n",
    "sns.heatmap(df_num.corr(), annot=True,cmap=\"RdBu\")\n",
    "plt.title(\"Correlations Among Numeric Features\", fontsize = 18);"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We notice from the heatmap above that:\n",
    "- __Parents and sibling like to travel together <font color='blue'>(light blue squares)__</font>\n",
    "- __Age has a high negative correlation with number of siblings__"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id=\"section4\"></a>\n",
    "## 4. Feature Engineering and Data Processing\n",
    "__Feature Engineering__ is the process of using raw data to create features that will be used for predictive modeling. Using, transforming, and combining existing features to define new features are also considered to be feature engineering."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id=\"section401\"></a>\n",
    "### 4.1 Drop 'PassengerId'\n",
    "\n",
    "First, I will drop ‘PassengerId’ from the train set, because it does not contribute to a persons' survival probability. I will not drop it from the test set, since it is required for the submission."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
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
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "      <th>train_test</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Braund, Mr. Owen Harris</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>A/5 21171</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>PC 17599</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C85</td>\n",
       "      <td>C</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>Heikkinen, Miss. Laina</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>STON/O2. 3101282</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113803</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>C123</td>\n",
       "      <td>S</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Allen, Mr. William Henry</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>373450</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Survived  Pclass                                               Name  \\\n",
       "0         0       3                            Braund, Mr. Owen Harris   \n",
       "1         1       1  Cumings, Mrs. John Bradley (Florence Briggs Th...   \n",
       "2         1       3                             Heikkinen, Miss. Laina   \n",
       "3         1       1       Futrelle, Mrs. Jacques Heath (Lily May Peel)   \n",
       "4         0       3                           Allen, Mr. William Henry   \n",
       "\n",
       "      Sex   Age  SibSp  Parch            Ticket     Fare Cabin Embarked  \\\n",
       "0    male  22.0      1      0         A/5 21171   7.2500   NaN        S   \n",
       "1  female  38.0      1      0          PC 17599  71.2833   C85        C   \n",
       "2  female  26.0      0      0  STON/O2. 3101282   7.9250   NaN        S   \n",
       "3  female  35.0      1      0            113803  53.1000  C123        S   \n",
       "4    male  35.0      0      0            373450   8.0500   NaN        S   \n",
       "\n",
       "   train_test  \n",
       "0           1  \n",
       "1           1  \n",
       "2           1  \n",
       "3           1  \n",
       "4           1  "
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_df = train_df.drop(['PassengerId'], axis=1)\n",
    "train_df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id=\"section402\"></a>\n",
    "### 4.2 Combining SibSp and Parch\n",
    "\n",
    "SibSp and Parch would make more sense as a combined feature that shows the total number of relatives a person has on the Titanic. I will create the new feature 'relative' below, and also a value that shows if someone is not alone."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1    537\n",
       "0    354\n",
       "Name: not_alone, dtype: int64"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data = [train_df, test_df]\n",
    "for dataset in data:\n",
    "    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']\n",
    "    dataset.loc[dataset['relatives'] > 0, 'not_alone'] = 0\n",
    "    dataset.loc[dataset['relatives'] == 0, 'not_alone'] = 1\n",
    "    dataset['not_alone'] = dataset['not_alone'].astype(int)\n",
    "train_df['not_alone'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "image/png": "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SKG7UfJkUYDqknJTg20bmWQGmWSm0N0qKfxMPDNBt4xLltVhaXMQHf1xLheYomDQg5GRZKG7QTNc86wQhgbjRTvDQhSFUw5cOhZk6+vrsXPnzvHH4XAYtbW1xz0fDofHHw8MDBz3PBERlQ+3KiPg4e7FRKVMFARUBzVUB7VCd6Xk8YYKFYNSPw4di99r1qzBjh07MDQ0hFQqhS1btmDt2rXjzzc2NkJVVbz11lsAgE2bNh33PBEREREREdGJOBZk6+rqcO+99+LOO+/EzTffjBtvvBErVqzA3Xffjffeew8A8M///M/4p3/6J1x77bVIJpO48847neoOERERERERlQnBLuFFDlwjS3R6PA6pGPA4pGLBY5GKAY9DKgbFfhyebo1sca7sJSIiIiIiIjoJBlkiIiIiIiIqKQyyREREREREVFJKujq9KE6u7mwhlEIfqfzxOKRiwOOQigWPRSoGPA6pGBTzcXi6vpX0Zk9EREREREQ083BqMREREREREZUUBlkiIiIiIiIqKQyyREREREREVFIYZImIiIiIiKikMMgSERERERFRSWGQJSIiIiIiopLCIEtEREREREQlhUGWiIiIiIiISgqDLBEREREREZUUBlkHPPnkk7j++utxzTXX4MEHHyx0d2iG+vd//3fccMMNuOGGG/Dd73630N2hGe473/kOvvrVrxa6GzSDbd26Fbfeeiuuu+46/OM//mOhu0Mz1KZNm8bPzd/5zncK3R2aYeLxOG688UZ0dnYCALZv344NGzbgmmuuwX333Vfg3k0eg+w06+vrw3333YeHHnoIjz/+OB5++GEcOnSo0N2iGWb79u147bXX8Nhjj+Hxxx/H7t278fzzzxe6WzRD7dixA4899lihu0EzWEdHB77xjW/gP/7jP/DEE09gz549ePnllwvdLZphUqkUvvnNb+LnP/85Nm3ahJ07d2L79u2F7hbNEL/73e/wiU98Aq2trQCAdDqNr33ta/iP//gPPPPMM/j9739fcp+LDLLTbPv27Vi1ahVCoRA8Hg/Wr1+PzZs3F7pbNMPU1NTgq1/9KlwuFxRFwfz589Hd3V3obtEMFIlEcN999+GLX/xiobtCM9jzzz+P66+/HvX19VAUBffddx9WrlxZ6G7RDGOaJizLQiqVgmEYMAwDqqoWuls0Q/zqV7/CN77xDdTW1gIAdu3ahTlz5qC5uRmyLGPDhg0ll1nkQneg3PT396Ompmb8cW1tLXbt2lXAHtFMtHDhwvE/t7a24tlnn8Uvf/nLAvaIZqq/+7u/w7333ouenp5Cd4VmsLa2NiiKgi9+8Yvo6enB5Zdfjq985SuF7hbNMD6fD//tv/03XHfdddA0DRdeeCHOO++8QneLZohvfvObxz0+UWbp6+vLd7emhCOy08yyLAiCMP7Ytu3jHhPl08GDB/HZz34Wf/mXf4mWlpZCd4dmmEceeQQNDQ1YvXp1obtCM5xpmtixYwe+9a1v4eGHH8auXbs43Z3ybt++ffj1r3+N3/zmN3j11VchiiIeeOCBQneLZqhyyCwMstOsvr4e4XB4/HE4HB4fwifKp7feeguf+cxn8Od//ue45ZZbCt0dmoGeeeYZbNu2DTfddBP+7d/+DVu3bsW3vvWtQneLZqDq6mqsXr0alZWVcLvduOqqqzhbivLutddew+rVq1FVVQWXy4Vbb70Vb7zxRqG7RTNUOWQWBtlptmbNGuzYsQNDQ0NIpVLYsmUL1q5dW+hu0QzT09ODP/mTP8E///M/44Ybbih0d2iG+vGPf4ynnnoKmzZtwj333IN169bha1/7WqG7RTPQFVdcgddeew3RaBSmaeLVV1/FsmXLCt0tmmGWLFmC7du3I5lMwrZtbN26FcuXLy90t2iGWrlyJY4ePYq2tjaYpomnnnqq5DIL18hOs7q6Otx777248847oes6brvtNqxYsaLQ3aIZ5oEHHkAmk8G3v/3t8a/dfvvt+MQnPlHAXhERFcbKlSvxuc99DnfccQd0XcfFF1+Mj370o4XuFs0wl1xyCfbs2YNbb70ViqJg+fLl+PznP1/obtEMpaoqvv3tb+NP//RPkclkcNlll+Haa68tdLcmRbBt2y50J4iIiIiIiIgmilOLiYiIiIiIqKQwyBIREREREVFJYZAlIiIiIiKiksIgS0RERERERCWFQZaIiIiIiIhKCoMsERFRgdx///34+7//+9O+7rOf/SyGhoYAAHfffTcOHTrkdNeIiIiKGuvIEhERFblt27aN//lHP/pRAXtCRERUHDgiS0RENM1ef/11bNy4Ebfffjs2bNiAF154AR/72Mdw88034/bbb8c777zzoe/5zW9+g9tvvx233norLr/8cvzLv/wLAOCv//qvAQB33XUXenp6sG7dOrz33nv48z//c/znf/7n+Pc/9NBD+MpXvgIA2Lp16wnbO3z48Hgbt9xyCx588EFnfxBEREQO4YgsERGRAw4ePIgXXngBuq7jT//0T/Gzn/0MFRUVOHjwIP7wD/8QW7ZsGX+tbdv4z//8T3z7299GS0sL+vr6cMUVV+DOO+/EP/3TP+HRRx/FT3/6U1RWVo5/z8c+9jF885vfxGc/+1kAwGOPPYZ7770Xra2tuO+++07Y3gMPPIB169bh85//PMLhML71rW/hE5/4BESR97WJiKi0MMgSERE5oKGhAY2NjXjwwQfR39+Pz3zmM+PPCYKA9vb24x7/4Ac/wEsvvYSnnnoKhw8fhm3bSKVSJ33/iy66CJlMBu+99x40TcPQ0BBWr16Nhx566KTtXX311firv/or7Nq1C6tXr8bXv/51hlgiIipJDLJEREQO8Hg8AADLsrB69erxqcIA0NPTg9raWjz//PMAgGQyiVtuuQVXXXUVLrjgAnz0ox/FCy+8ANu2T/r+giDgtttuw6ZNm6AoCm677TYIgnDK9pYsWYLnnnsO27dvx44dO/C9730Pjz76KOrr6x35GRARETmFt2GJiIgctHr1amzbtg2HDx8GALz88svYuHEj0un0+Gva2toQj8fxla98BevWrcPrr7+ObDYLy7IAAJIkwTCMD733Lbfcgq1bt+K5557Drbfeetr2/vzP/xzPPPMMbrjhBnzjG9+Az+c7bmSYiIioVHBEloiIyEELFizA3//93+PP/uzPYNs2ZFnG97//fXi93vHXLF68GJdffjmuu+46uFwuLFq0CAsWLEBbWxtmz56Na6+9Fp/+9Kdx//33H/feNTU1WLp0KQzDQF1d3Wnb+9KXvoS/+Zu/wcMPPwxJknDVVVfhwgsvzOvPg4iIaDoI9qnmLREREREREREVGU4tJiIiIiIiopLCIEtEREREREQlhUGWiIiIiIiISgqDLBEREREREZUUBlkiIiIiIiIqKQyyREREREREVFIYZImIiIiIiKikMMgSERERERFRSfn/AZNq6CKnLG80AAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 1152x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.subplots(figsize = (16,4))\n",
    "ax = sns.lineplot(x='relatives',y='Survived', data=train_df)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id=\"section403\"></a>\n",
    "### 4.3 Missing Data\n",
    "\n",
    "As a reminder, we have to deal with __Cabin (687 missing values), Embarked (2 missing values)__ and __Age (177 missing values).__"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [],
   "source": [
    "import re\n",
    "deck = {\"A\": 1, \"B\": 2, \"C\": 3, \"D\": 4, \"E\": 5, \"F\": 6, \"G\": 7, \"U\": 8}\n",
    "data = [train_df, test_df]\n",
    "\n",
    "for dataset in data:\n",
    "    dataset['Cabin'] = dataset['Cabin'].fillna(\"U0\")\n",
    "    dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile(\"([a-zA-Z]+)\").search(x).group())\n",
    "    dataset['Deck'] = dataset['Deck'].map(deck)\n",
    "    dataset['Deck'] = dataset['Deck'].fillna(0)\n",
    "    dataset['Deck'] = dataset['Deck'].astype(int)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [],
   "source": [
    "# We can now drop the Cabin feature\n",
    "train_df = train_df.drop(['Cabin'], axis=1)\n",
    "test_df = test_df.drop(['Cabin'], axis=1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Age\n",
    "\n",
    "As seen previously on __\"3.1 Dealing with Missing Values\"__, there are a lot of missing 'Age' values (177 data points). We can normalize the 'Age' feature by creating an array that contains random numbers, which are computed based on the mean age value in regards to the standard deviation and is_null."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [],
   "source": [
    "data = [train_df, test_df]\n",
    "\n",
    "for dataset in data:\n",
    "    mean = train_df[\"Age\"].mean()\n",
    "    std = test_df[\"Age\"].std()\n",
    "    is_null = dataset[\"Age\"].isnull().sum()\n",
    "    \n",
    "    # Compute random numbers between the mean, std and is_null\n",
    "    rand_age = np.random.randint(mean - std, mean + std, size = is_null)\n",
    "    \n",
    "    # Fill NaN values in Age column with random values generated\n",
    "    age_slice = dataset[\"Age\"].copy()\n",
    "    age_slice[np.isnan(age_slice)] = rand_age\n",
    "    dataset[\"Age\"] = age_slice\n",
    "    dataset[\"Age\"] = train_df[\"Age\"].astype(int)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_df[\"Age\"].isnull().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Embarked\n",
    "\n",
    "Since the Embarked feature has only 2 missing values, we will fill these with the most common one."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "count     889\n",
       "unique      3\n",
       "top         S\n",
       "freq      644\n",
       "Name: Embarked, dtype: object"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_df['Embarked'].describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We notice the most popular embark location is __Southampton (S).__"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [],
   "source": [
    "common_value = 'S'\n",
    "data = [train_df, test_df]\n",
    "\n",
    "for dataset in data:\n",
    "    dataset['Embarked'] = dataset['Embarked'].fillna(common_value)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_df['Embarked'].isnull().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id=\"section404\"></a>\n",
    "### 4.4 Converting Features"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 891 entries, 0 to 890\n",
      "Data columns (total 14 columns):\n",
      " #   Column      Non-Null Count  Dtype  \n",
      "---  ------      --------------  -----  \n",
      " 0   Survived    891 non-null    int64  \n",
      " 1   Pclass      891 non-null    int64  \n",
      " 2   Name        891 non-null    object \n",
      " 3   Sex         891 non-null    object \n",
      " 4   Age         891 non-null    int32  \n",
      " 5   SibSp       891 non-null    int64  \n",
      " 6   Parch       891 non-null    int64  \n",
      " 7   Ticket      891 non-null    object \n",
      " 8   Fare        891 non-null    float64\n",
      " 9   Embarked    891 non-null    object \n",
      " 10  train_test  891 non-null    int64  \n",
      " 11  relatives   891 non-null    int64  \n",
      " 12  not_alone   891 non-null    int32  \n",
      " 13  Deck        891 non-null    int32  \n",
      "dtypes: float64(1), int32(3), int64(6), object(4)\n",
      "memory usage: 87.1+ KB\n"
     ]
    }
   ],
   "source": [
    "train_df.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We can see that __'Fare'__ is a float data-type. Also, we need to deal with 4 categorical features: __Name, Sex, Ticket, and Embarked__"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Fare\n",
    "\n",
    "Converting 'Fare' from __float64__ to __int64__ using the __astype()__ function provided by pandas"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [],
   "source": [
    "data = [train_df, test_df]\n",
    "\n",
    "for dataset in data:\n",
    "    dataset['Fare'] = dataset['Fare'].fillna(0)\n",
    "    dataset['Fare'] = dataset['Fare'].astype(int)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 891 entries, 0 to 890\n",
      "Data columns (total 14 columns):\n",
      " #   Column      Non-Null Count  Dtype \n",
      "---  ------      --------------  ----- \n",
      " 0   Survived    891 non-null    int64 \n",
      " 1   Pclass      891 non-null    int64 \n",
      " 2   Name        891 non-null    object\n",
      " 3   Sex         891 non-null    object\n",
      " 4   Age         891 non-null    int32 \n",
      " 5   SibSp       891 non-null    int64 \n",
      " 6   Parch       891 non-null    int64 \n",
      " 7   Ticket      891 non-null    object\n",
      " 8   Fare        891 non-null    int32 \n",
      " 9   Embarked    891 non-null    object\n",
      " 10  train_test  891 non-null    int64 \n",
      " 11  relatives   891 non-null    int64 \n",
      " 12  not_alone   891 non-null    int32 \n",
      " 13  Deck        891 non-null    int32 \n",
      "dtypes: int32(4), int64(6), object(4)\n",
      "memory usage: 83.7+ KB\n"
     ]
    }
   ],
   "source": [
    "train_df.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Name\n",
    "\n",
    "Feature Engineering the name of passengers to extract a person's title (Mr, Miss, Master, and Other), so we can build another feature called **'Title'** out of it."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [],
   "source": [
    "data = [train_df, test_df]\n",
    "titles = {\"Mr\": 1, \"Miss\": 2, \"Mrs\": 3, \"Master\": 4, \"Other\": 5}\n",
    "\n",
    "for dataset in data:\n",
    "    # Extract titles\n",
    "    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\\.', expand=False)\n",
    "    \n",
    "    # Replace titles with a more common title or as Other\n",
    "    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr','Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')\n",
    "    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')\n",
    "    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')\n",
    "    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')\n",
    "    \n",
    "    # Convert titles into numbers\n",
    "    dataset['Title'] = dataset['Title'].map(titles)\n",
    "    \n",
    "    # Filling NaN with 0 just to be safe\n",
    "    dataset['Title'] = dataset['Title'].fillna(0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_df = train_df.drop(['Name'], axis=1)\n",
    "test_df = test_df.drop(['Name'], axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
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
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Embarked</th>\n",
       "      <th>train_test</th>\n",
       "      <th>relatives</th>\n",
       "      <th>not_alone</th>\n",
       "      <th>Deck</th>\n",
       "      <th>Title</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>22</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>A/5 21171</td>\n",
       "      <td>7</td>\n",
       "      <td>S</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>8</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>female</td>\n",
       "      <td>38</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>PC 17599</td>\n",
       "      <td>71</td>\n",
       "      <td>C</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>female</td>\n",
       "      <td>26</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>STON/O2. 3101282</td>\n",
       "      <td>7</td>\n",
       "      <td>S</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>8</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>female</td>\n",
       "      <td>35</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113803</td>\n",
       "      <td>53</td>\n",
       "      <td>S</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>35</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>373450</td>\n",
       "      <td>8</td>\n",
       "      <td>S</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>8</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Survived  Pclass     Sex  Age  SibSp  Parch            Ticket  Fare  \\\n",
       "0         0       3    male   22      1      0         A/5 21171     7   \n",
       "1         1       1  female   38      1      0          PC 17599    71   \n",
       "2         1       3  female   26      0      0  STON/O2. 3101282     7   \n",
       "3         1       1  female   35      1      0            113803    53   \n",
       "4         0       3    male   35      0      0            373450     8   \n",
       "\n",
       "  Embarked  train_test  relatives  not_alone  Deck  Title  \n",
       "0        S           1          1          0     8      1  \n",
       "1        C           1          1          0     3      3  \n",
       "2        S           1          0          1     8      2  \n",
       "3        S           1          1          0     3      3  \n",
       "4        S           1          0          1     8      1  "
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Checking results\n",
    "train_df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Sex\n",
    "\n",
    "Convert feature 'Sex' into numeric values\n",
    "- male = 0\n",
    "- female = 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [],
   "source": [
    "genders = {\"male\": 0, \"female\": 1}\n",
    "data = [train_df, test_df]\n",
    "\n",
    "for dataset in data:\n",
    "    dataset['Sex'] = dataset['Sex'].map(genders)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
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
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Embarked</th>\n",
       "      <th>train_test</th>\n",
       "      <th>relatives</th>\n",
       "      <th>not_alone</th>\n",
       "      <th>Deck</th>\n",
       "      <th>Title</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>22</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>A/5 21171</td>\n",
       "      <td>7</td>\n",
       "      <td>S</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>8</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>38</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>PC 17599</td>\n",
       "      <td>71</td>\n",
       "      <td>C</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>26</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>STON/O2. 3101282</td>\n",
       "      <td>7</td>\n",
       "      <td>S</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>8</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>35</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113803</td>\n",
       "      <td>53</td>\n",
       "      <td>S</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>35</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>373450</td>\n",
       "      <td>8</td>\n",
       "      <td>S</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>8</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Survived  Pclass  Sex  Age  SibSp  Parch            Ticket  Fare Embarked  \\\n",
       "0         0       3    0   22      1      0         A/5 21171     7        S   \n",
       "1         1       1    1   38      1      0          PC 17599    71        C   \n",
       "2         1       3    1   26      0      0  STON/O2. 3101282     7        S   \n",
       "3         1       1    1   35      1      0            113803    53        S   \n",
       "4         0       3    0   35      0      0            373450     8        S   \n",
       "\n",
       "   train_test  relatives  not_alone  Deck  Title  \n",
       "0           1          1          0     8      1  \n",
       "1           1          1          0     3      3  \n",
       "2           1          0          1     8      2  \n",
       "3           1          1          0     3      3  \n",
       "4           1          0          1     8      1  "
      ]
     },
     "execution_count": 42,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Ticket"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "count      891\n",
       "unique     681\n",
       "top       1601\n",
       "freq         7\n",
       "Name: Ticket, dtype: object"
      ]
     },
     "execution_count": 43,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_df['Ticket'].describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Since the __'Ticket'__ feature has 681 unique values, it would be very hard to convert them into an useful feature. __Hence, we will drop it from the DataFrame.__"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_df = train_df.drop(['Ticket'], axis=1)\n",
    "test_df = test_df.drop(['Ticket'], axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
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
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Embarked</th>\n",
       "      <th>train_test</th>\n",
       "      <th>relatives</th>\n",
       "      <th>not_alone</th>\n",
       "      <th>Deck</th>\n",
       "      <th>Title</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>22</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "      <td>S</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>8</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>38</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>71</td>\n",
       "      <td>C</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>26</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "      <td>S</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>8</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>35</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>53</td>\n",
       "      <td>S</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>35</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>8</td>\n",
       "      <td>S</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>8</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Survived  Pclass  Sex  Age  SibSp  Parch  Fare Embarked  train_test  \\\n",
       "0         0       3    0   22      1      0     7        S           1   \n",
       "1         1       1    1   38      1      0    71        C           1   \n",
       "2         1       3    1   26      0      0     7        S           1   \n",
       "3         1       1    1   35      1      0    53        S           1   \n",
       "4         0       3    0   35      0      0     8        S           1   \n",
       "\n",
       "   relatives  not_alone  Deck  Title  \n",
       "0          1          0     8      1  \n",
       "1          1          0     3      3  \n",
       "2          0          1     8      2  \n",
       "3          1          0     3      3  \n",
       "4          0          1     8      1  "
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Convert 'Embarked' feature into numeric values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [],
   "source": [
    "ports = {\"S\": 0, \"C\": 1, \"Q\": 2}\n",
    "data = [train_df, test_df]\n",
    "\n",
    "for dataset in data:\n",
    "    dataset['Embarked'] = dataset['Embarked'].map(ports)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
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
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Embarked</th>\n",
       "      <th>train_test</th>\n",
       "      <th>relatives</th>\n",
       "      <th>not_alone</th>\n",
       "      <th>Deck</th>\n",
       "      <th>Title</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>22</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>8</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>38</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>71</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>26</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>8</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>35</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>53</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>35</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>8</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>8</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Survived  Pclass  Sex  Age  SibSp  Parch  Fare  Embarked  train_test  \\\n",
       "0         0       3    0   22      1      0     7         0           1   \n",
       "1         1       1    1   38      1      0    71         1           1   \n",
       "2         1       3    1   26      0      0     7         0           1   \n",
       "3         1       1    1   35      1      0    53         0           1   \n",
       "4         0       3    0   35      0      0     8         0           1   \n",
       "\n",
       "   relatives  not_alone  Deck  Title  \n",
       "0          1          0     8      1  \n",
       "1          1          0     3      3  \n",
       "2          0          1     8      2  \n",
       "3          1          0     3      3  \n",
       "4          0          1     8      1  "
      ]
     },
     "execution_count": 47,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id=\"section405\"></a>\n",
    "### 4.5 Creating new Categories"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [],
   "source": [
    "data = [train_df, test_df]\n",
    "for dataset in data:\n",
    "    dataset['Age'] = dataset['Age'].astype(int)\n",
    "    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0\n",
    "    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1\n",
    "    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2\n",
    "    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3\n",
    "    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4\n",
    "    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5\n",
    "    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6\n",
    "    dataset.loc[ dataset['Age'] > 66, 'Age'] = 6"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "6    163\n",
       "4    156\n",
       "3    147\n",
       "5    135\n",
       "2    126\n",
       "1     96\n",
       "0     68\n",
       "Name: Age, dtype: int64"
      ]
     },
     "execution_count": 49,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Checking the distribution\n",
    "train_df['Age'].value_counts()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Fare\n",
    "\n",
    "For the 'Fare' feature, we need to do the same as with the 'Age' feature. But it isn't that easy, because if we cut the range of the fare values into a few equally big categories, 80% of the values would fall into the first category. Fortunately, we can use pandas \"qcut()\" function, that we can use to see, how we can form the categories."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
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
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Embarked</th>\n",
       "      <th>train_test</th>\n",
       "      <th>relatives</th>\n",
       "      <th>not_alone</th>\n",
       "      <th>Deck</th>\n",
       "      <th>Title</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>8</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>5</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>71</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>8</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>5</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>53</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>8</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>8</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Survived  Pclass  Sex  Age  SibSp  Parch  Fare  Embarked  train_test  \\\n",
       "0         0       3    0    2      1      0     7         0           1   \n",
       "1         1       1    1    5      1      0    71         1           1   \n",
       "2         1       3    1    3      0      0     7         0           1   \n",
       "3         1       1    1    5      1      0    53         0           1   \n",
       "4         0       3    0    5      0      0     8         0           1   \n",
       "\n",
       "   relatives  not_alone  Deck  Title  \n",
       "0          1          0     8      1  \n",
       "1          1          0     3      3  \n",
       "2          0          1     8      2  \n",
       "3          1          0     3      3  \n",
       "4          0          1     8      1  "
      ]
     },
     "execution_count": 50,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0      (-0.001, 7.0]\n",
       "1      (52.0, 512.0]\n",
       "2      (-0.001, 7.0]\n",
       "3      (52.0, 512.0]\n",
       "4         (7.0, 8.0]\n",
       "           ...      \n",
       "886      (8.0, 14.0]\n",
       "887     (26.0, 52.0]\n",
       "888     (14.0, 26.0]\n",
       "889     (26.0, 52.0]\n",
       "890    (-0.001, 7.0]\n",
       "Name: Fare, Length: 891, dtype: category\n",
       "Categories (6, interval[float64]): [(-0.001, 7.0] < (7.0, 8.0] < (8.0, 14.0] < (14.0, 26.0] < (26.0, 52.0] < (52.0, 512.0]]"
      ]
     },
     "execution_count": 51,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pd.qcut(train_df['Fare'], q=6)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Using the values from **pd.qcut()** to create bins for Fare"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [],
   "source": [
    "data = [train_df, test_df]\n",
    "\n",
    "for dataset in data:\n",
    "    dataset.loc[ dataset['Fare'] <= 7, 'Fare'] = 0\n",
    "    dataset.loc[(dataset['Fare'] > 7) & (dataset['Fare'] <= 8), 'Fare'] = 1\n",
    "    dataset.loc[(dataset['Fare'] > 8) & (dataset['Fare'] <= 14), 'Fare']   = 2\n",
    "    dataset.loc[(dataset['Fare'] > 14) & (dataset['Fare'] <= 26), 'Fare']   = 3\n",
    "    dataset.loc[(dataset['Fare'] > 26) & (dataset['Fare'] <= 52), 'Fare']   = 4\n",
    "    dataset.loc[dataset['Fare'] > 52, 'Fare'] = 5\n",
    "    dataset['Fare'] = dataset['Fare'].astype(int)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
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
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Embarked</th>\n",
       "      <th>train_test</th>\n",
       "      <th>relatives</th>\n",
       "      <th>not_alone</th>\n",
       "      <th>Deck</th>\n",
       "      <th>Title</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>8</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>5</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>5</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>8</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>5</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>8</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>8</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>6</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>5</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "      <td>8</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>8</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>8</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Survived  Pclass  Sex  Age  SibSp  Parch  Fare  Embarked  train_test  \\\n",
       "0         0       3    0    2      1      0     0         0           1   \n",
       "1         1       1    1    5      1      0     5         1           1   \n",
       "2         1       3    1    3      0      0     0         0           1   \n",
       "3         1       1    1    5      1      0     5         0           1   \n",
       "4         0       3    0    5      0      0     1         0           1   \n",
       "5         0       3    0    5      0      0     1         2           1   \n",
       "6         0       1    0    6      0      0     4         0           1   \n",
       "7         0       3    0    0      3      1     3         0           1   \n",
       "8         1       3    1    3      0      2     2         0           1   \n",
       "9         1       2    1    1      1      0     4         1           1   \n",
       "\n",
       "   relatives  not_alone  Deck  Title  \n",
       "0          1          0     8      1  \n",
       "1          1          0     3      3  \n",
       "2          0          1     8      2  \n",
       "3          1          0     3      3  \n",
       "4          0          1     8      1  \n",
       "5          0          1     8      1  \n",
       "6          0          1     5      1  \n",
       "7          4          0     8      4  \n",
       "8          2          0     8      3  \n",
       "9          1          0     8      3  "
      ]
     },
     "execution_count": 53,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Checking the dataset\n",
    "train_df.head(10)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id=\"section5\"></a>\n",
    "## 5. Model building"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train = train_df.drop(\"Survived\", axis=1)\n",
    "Y_train = train_df[\"Survived\"]\n",
    "X_test  = test_df.drop(\"PassengerId\", axis=1).copy()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id=\"section501\"></a>\n",
    "### 5.1 Stochastic Gradient Descent (SGD)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "79.69 %\n"
     ]
    }
   ],
   "source": [
    "sgd = linear_model.SGDClassifier(max_iter=5, tol=None)\n",
    "sgd.fit(X_train, Y_train)\n",
    "\n",
    "Y_pred = sgd.predict(X_test)\n",
    "\n",
    "sgd.score(X_train, Y_train)\n",
    "acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)\n",
    "\n",
    "# Print score\n",
    "print(round(acc_sgd,2,), \"%\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id=\"section502\"></a>\n",
    "### 5.2 Decision Tree"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "93.15 %\n"
     ]
    }
   ],
   "source": [
    "decision_tree = DecisionTreeClassifier()\n",
    "decision_tree.fit(X_train, Y_train)\n",
    "\n",
    "Y_pred = decision_tree.predict(X_test)\n",
    "\n",
    "acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)\n",
    "\n",
    "# Print score\n",
    "print(round(acc_decision_tree,2,), \"%\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id=\"section503\"></a>\n",
    "### 5.3 Random Forest"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "93.15 %\n"
     ]
    }
   ],
   "source": [
    "random_forest = RandomForestClassifier(n_estimators=100)\n",
    "random_forest.fit(X_train, Y_train)\n",
    "\n",
    "Y_prediction = random_forest.predict(X_test)\n",
    "\n",
    "random_forest.score(X_train, Y_train)\n",
    "acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)\n",
    "\n",
    "# Print score\n",
    "print(round(acc_random_forest,2,), \"%\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id=\"section504\"></a>\n",
    "### 5.4 Logistic Regression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "81.71 %\n"
     ]
    }
   ],
   "source": [
    "logreg = LogisticRegression()\n",
    "logreg.fit(X_train, Y_train)\n",
    "\n",
    "Y_pred = logreg.predict(X_test)\n",
    "\n",
    "acc_log = round(logreg.score(X_train, Y_train) * 100, 2)\n",
    "\n",
    "# Print score\n",
    "print(round(acc_log,2,), \"%\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id=\"section505\"></a>\n",
    "### 5.5 KNN"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "86.98 %\n"
     ]
    }
   ],
   "source": [
    "knn = KNeighborsClassifier(n_neighbors = 3)\n",
    "knn.fit(X_train, Y_train)\n",
    "\n",
    "Y_pred = knn.predict(X_test)\n",
    "\n",
    "acc_knn = round(knn.score(X_train, Y_train) * 100, 2)\n",
    "\n",
    "# Print score\n",
    "print(round(acc_knn,2,), \"%\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id=\"section506\"></a>\n",
    "### 5.6 Gaussian Naive Bayes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "78.68 %\n"
     ]
    }
   ],
   "source": [
    "gaussian = GaussianNB()\n",
    "gaussian.fit(X_train, Y_train)\n",
    "\n",
    "Y_pred = gaussian.predict(X_test)\n",
    "\n",
    "acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)\n",
    "\n",
    "# Print score\n",
    "print(round(acc_gaussian,2,), \"%\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id=\"section507\"></a>\n",
    "### 5.7 Perceptron"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "69.58 %\n"
     ]
    }
   ],
   "source": [
    "perceptron = Perceptron(max_iter=1000)\n",
    "perceptron.fit(X_train, Y_train)\n",
    "\n",
    "Y_pred = perceptron.predict(X_test)\n",
    "\n",
    "acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)\n",
    "\n",
    "# Print score\n",
    "print(round(acc_perceptron,2,), \"%\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id=\"section6\"></a>\n",
    "## 6. Model evaluation\n",
    "\n",
    "### Which one is the best model?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
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
       "      <th>Model</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Score</th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>93.15</th>\n",
       "      <td>Random Forest</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>93.15</th>\n",
       "      <td>Decision Tree</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>86.98</th>\n",
       "      <td>KNN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>81.71</th>\n",
       "      <td>Logistic Regression</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>79.69</th>\n",
       "      <td>Stochastic Gradient Decent</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>78.68</th>\n",
       "      <td>Naive Bayes</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>69.58</th>\n",
       "      <td>Perceptron</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                            Model\n",
       "Score                            \n",
       "93.15               Random Forest\n",
       "93.15               Decision Tree\n",
       "86.98                         KNN\n",
       "81.71         Logistic Regression\n",
       "79.69  Stochastic Gradient Decent\n",
       "78.68                 Naive Bayes\n",
       "69.58                  Perceptron"
      ]
     },
     "execution_count": 62,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "results = pd.DataFrame({\n",
    "    'Model': ['KNN', 'Logistic Regression', \n",
    "              'Random Forest', 'Naive Bayes', 'Perceptron', \n",
    "              'Stochastic Gradient Decent', \n",
    "              'Decision Tree'],\n",
    "    'Score': [acc_knn, acc_log, \n",
    "              acc_random_forest, acc_gaussian, acc_perceptron, \n",
    "              acc_sgd, acc_decision_tree]})\n",
    "\n",
    "result_df = results.sort_values(by='Score', ascending=False)\n",
    "result_df = result_df.set_index('Score')\n",
    "result_df.head(9)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The __Random Forest classifier__ goes on top of the Machine Learning models, followed by **Decision Tree** and __KNN__ respectfully. Now we need to check how the Random Forest performs by using cross validation."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id=\"section601\"></a>\n",
    "### 6.1 K-Fold Cross Validation\n",
    "K-Fold Cross Validation randomly splits the training data into __K subsets called folds__. Image we split our data into 4 folds (K = 4). The random forest model would be trained and validated 4 times, using a different fold for validation every time, while it would be trained on the remaining 3 folds.\n",
    "\n",
    "The image below shows the process, using 4 folds (K = 4). Every row represents one training + validation process. In the first row, the model is trained on the second, third and fourth subsets and validated on the first subset. In the second row, the model is trained on the first, third and fourth subsets and validated on the second subset. K-Fold Cross Validation repeats this process until every fold acted once as an evaluation fold."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import cross_val_score\n",
    "\n",
    "rf = RandomForestClassifier(n_estimators=100)\n",
    "scores = cross_val_score(rf, X_train, Y_train, cv=10, scoring = \"accuracy\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Scores: [0.78888889 0.83146067 0.75280899 0.85393258 0.87640449 0.83146067\n",
      " 0.79775281 0.7752809  0.85393258 0.83146067]\n",
      "Mean: 0.8193383270911362\n",
      "Standard Deviation: 0.03721725315177691\n"
     ]
    }
   ],
   "source": [
    "print(\"Scores:\", scores)\n",
    "print(\"Mean:\", scores.mean())\n",
    "print(\"Standard Deviation:\", scores.std())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This looks much more realistic than before. The __Random Forest classifier__ model has an average __accuracy of 81%__ with a __standard deviation of 3.9%__. The standard deviation tell us how precise the estimates are.\n",
    "\n",
    "- This means the accuracy of our model can differ __± 3.9%__ \n",
    "\n",
    "I believe the accuracy looks good. Since Random Forest is a model easy to use, we will try to increase its performance even further in the following section."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id=\"section602\"></a>\n",
    "### 6.2 Random Forest"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "metadata": {},
   "outputs": [],
   "source": [
    "importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(random_forest.feature_importances_,3)})\n",
    "importances = importances.sort_values('importance',ascending=False).set_index('feature')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
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
       "      <th>importance</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>feature</th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Title</th>\n",
       "      <td>0.188</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Sex</th>\n",
       "      <td>0.183</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Age</th>\n",
       "      <td>0.159</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Fare</th>\n",
       "      <td>0.105</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Deck</th>\n",
       "      <td>0.094</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Pclass</th>\n",
       "      <td>0.086</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>relatives</th>\n",
       "      <td>0.063</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Embarked</th>\n",
       "      <td>0.048</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>SibSp</th>\n",
       "      <td>0.037</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Parch</th>\n",
       "      <td>0.026</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>not_alone</th>\n",
       "      <td>0.011</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>train_test</th>\n",
       "      <td>0.000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "            importance\n",
       "feature               \n",
       "Title            0.188\n",
       "Sex              0.183\n",
       "Age              0.159\n",
       "Fare             0.105\n",
       "Deck             0.094\n",
       "Pclass           0.086\n",
       "relatives        0.063\n",
       "Embarked         0.048\n",
       "SibSp            0.037\n",
       "Parch            0.026\n",
       "not_alone        0.011\n",
       "train_test       0.000"
      ]
     },
     "execution_count": 66,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "importances.head(12)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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wZYtW9CmTRskJSWhV69eGDJkiHQFJIuUR4iPj0+91hswYIDIz88X2dnZon379kKj0QghhLh8+bKkh29CCHH27FmxcOFC4eLiIvz9/cUXX3whevXqJWmNO/35559CCCF++eUX8cEHH4jy8nLxzjvviC5duojPP/9ctrqrVq0SEydOFL179xaff/658PHx0Z9ukMP27dvFqFGjRGFhoXB3dxeenp6ynfpau3at8PX1FV5eXmL16tUiOztbljqV6rsDv8jISDFkyBCxfv168cUXX4hBgwaJlStXylqzvlXXyaNcR5DVfXZRUVGSrb9BPln83HPPVdvRmxBClq5pU1NTERERAa1Wiz59+mDu3LnYvn07li5dirfffluWvnG0Wi327duHhIQE7N+/Hy4uLggICECvXr0krzVgwAAkJiYCANauXYvAwEDJa1Tnl19+wenTp7F+/Xo0b94cAwcOxCuvvCJrzcrfkVu3buHChQv473//K8sR16JFi+Dt7Y3//ve/kq/7drc/U1JaWoqPPvoIXl5e+rpy9dukVquRkJAAMzMzAMBff/2FgQMH4ptvvpGlXn1e6K+kVquxYcMGNGrUCABQUFCAgIAA/fMMUoiMjMS1a9fw3Xff6W+jBirugDxx4oRk3VA3yFNDLVq0wOrVq+utnru7O55//nlcv35d3+PiE088gQULFqBbt26y1DQ1NUW/fv3Qr18/5OXlISkpCVFRUbIEwe37CikpKbIHwbVr1zBx4kScP38eLVq0gKmpKQ4dOoSSkhJ06dJF8l4m67PP/r1796J3795o06YNzp49i7Nnz1aZLvUX88PqwK9x48YoKiqCnZ0dgIrnXuS8a2jGjBkYPXo0EhMTMXz4cNnvMAMqbm/28/NDnz59IITA3r17MXbsWElruLq64tdff8WhQ4eqPP9kYmIi6V1fDTIIzMzM0Lx583qt6eDgUOUoRI4v5Jo0bdoUgYGBsn1B375HXB8HkFFRUejSpQu++OIL/R5lWVkZli1bhvDwcCxatEjSenI9YFidjIwM9O7dG+np6dVOl/qLuTLEli5diuDgYEnXXZ3KUNXpdPD29kafPn1gYmKC/fv3y9bZI1BxXWLgwIHIyspCo0aNEBERIfuTxQMHDkSHDh1w+PBh6HQ6LF++HG3atJG0RseOHdGxY0f069evxh2gcePGYdWqVQ9Up0EG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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "importances.plot.bar();"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id=\"section604\"></a>\n",
    "### 6.4 Results\n",
    "\n",
    "__'not_alone' and 'Parch' don't play a significant role in the Random Forest classifiers prediction process__. Thus, I will drop them from the DataFrame and train the classifier once again. We could also remove more features, however, this would inquire more investigations of the feature's effect on our model. For now, I will only remove 'not_alone' and 'Parch' from the DataFrame."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Dropping not_alone\n",
    "train_df  = train_df.drop(\"not_alone\", axis=1)\n",
    "test_df  = test_df.drop(\"not_alone\", axis=1)\n",
    "\n",
    "# Dropping Parch\n",
    "train_df  = train_df.drop(\"Parch\", axis=1)\n",
    "test_df  = test_df.drop(\"Parch\", axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "metadata": {},
   "outputs": [],
   "source": [
    "# # Reassigning features\n",
    "X_train = train_df.drop(\"Survived\", axis=1)\n",
    "Y_train = train_df[\"Survived\"]\n",
    "X_test  = test_df.drop(\"PassengerId\", axis=1).copy()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Training the Random Forest classifier once again"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "93.15 %\n"
     ]
    }
   ],
   "source": [
    "random_forest = RandomForestClassifier(n_estimators=100, oob_score = True)\n",
    "random_forest.fit(X_train, Y_train)\n",
    "\n",
    "Y_prediction = random_forest.predict(X_test)\n",
    "\n",
    "random_forest.score(X_train, Y_train)\n",
    "acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)\n",
    "\n",
    "# Print scores\n",
    "print(round(acc_random_forest,2,), \"%\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Feature importance without 'not_alone' and 'Parch' features"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "metadata": {},
   "outputs": [],
   "source": [
    "importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(random_forest.feature_importances_,3)})\n",
    "importances = importances.sort_values('importance',ascending=False).set_index('feature')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
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
       "      <th>importance</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>feature</th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Title</th>\n",
       "      <td>0.216</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Sex</th>\n",
       "      <td>0.167</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Age</th>\n",
       "      <td>0.152</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Fare</th>\n",
       "      <td>0.115</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Deck</th>\n",
       "      <td>0.087</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Pclass</th>\n",
       "      <td>0.084</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>relatives</th>\n",
       "      <td>0.080</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>SibSp</th>\n",
       "      <td>0.052</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Embarked</th>\n",
       "      <td>0.047</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>train_test</th>\n",
       "      <td>0.000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "            importance\n",
       "feature               \n",
       "Title            0.216\n",
       "Sex              0.167\n",
       "Age              0.152\n",
       "Fare             0.115\n",
       "Deck             0.087\n",
       "Pclass           0.084\n",
       "relatives        0.080\n",
       "SibSp            0.052\n",
       "Embarked         0.047\n",
       "train_test       0.000"
      ]
     },
     "execution_count": 72,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "importances.head(12)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The __Random Forest__ model predicts as good as it did before. A general rule is that, the more features you have, the more likely your model will suffer from overfitting and vice versa. But I think our data looks fine for now and hasn't too much features.\n",
    "\n",
    "Moreover, there is another way to validate the Random Forest classifier, which is as accurate as the score used before. We can use something called __Out of Bag (OOB) score__ to estimate the generalization accuracy. __Basically, the OOB score is computed as the number of correctly predicted rows from the out of the bag sample__."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "oob score: 81.71000000000001 %\n"
     ]
    }
   ],
   "source": [
    "print(\"oob score:\", round(random_forest.oob_score_, 4)*100, \"%\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Now we can start tuning the **hyperameters** of random forest."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id=\"section605\"></a>\n",
    "### 6.5 Hyperparameter Tuning"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 85,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Simple performance reporting function\n",
    "def clf_performance(classifier, model_name):\n",
    "    print(model_name)\n",
    "    print('Best Score: ' + str(classifier.best_score_))\n",
    "    print('Best Parameters: ' + str(classifier.best_params_))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 86,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Fitting 5 folds for each of 288 candidates, totalling 1440 fits\n",
      "Random Forest\n",
      "Best Score: 0.8383842822170611\n",
      "Best Parameters: {'bootstrap': True, 'criterion': 'gini', 'max_depth': 15, 'max_features': 'auto', 'min_samples_leaf': 3, 'min_samples_split': 2, 'n_estimators': 400}\n"
     ]
    }
   ],
   "source": [
    "from sklearn.model_selection import GridSearchCV\n",
    "\n",
    "rf = RandomForestClassifier(random_state = 1)\n",
    "param_grid =  {'n_estimators': [400,450,500,550],\n",
    "               'criterion':['gini','entropy'],\n",
    "               'bootstrap': [True],\n",
    "               'max_depth': [15, 20, 25],\n",
    "               'max_features': ['auto','sqrt', 10],\n",
    "               'min_samples_leaf': [2,3],\n",
    "               'min_samples_split': [2,3]}\n",
    "                                  \n",
    "clf_rf = GridSearchCV(rf, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)\n",
    "best_clf_rf = clf_rf.fit(X_train,Y_train)\n",
    "\n",
    "# Print score\n",
    "clf_performance(best_clf_rf,'Random Forest')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id=\"section606\"></a>\n",
    "### 6.6 Testing new parameters"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 87,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "oob score: 82.94 %\n"
     ]
    }
   ],
   "source": [
    "random_forest = RandomForestClassifier(criterion = \"gini\",\n",
    "                                       max_depth = 20,\n",
    "                                       max_features='auto',\n",
    "                                       min_samples_leaf = 3, \n",
    "                                       min_samples_split = 2,\n",
    "                                       n_estimators=450,\n",
    "                                       oob_score=True, \n",
    "                                       random_state=1, \n",
    "                                       n_jobs=-1)\n",
    "\n",
    "random_forest.fit(X_train, Y_train)\n",
    "Y_prediction = random_forest.predict(X_test)\n",
    "\n",
    "random_forest.score(X_train, Y_train)\n",
    "\n",
    "print(\"oob score:\", round(random_forest.oob_score_, 4)*100, \"%\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id=\"section607\"></a>\n",
    "### 6.7 Further evaluation"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Confusion Matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 88,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[494,  55],\n",
       "       [100, 242]], dtype=int64)"
      ]
     },
     "execution_count": 88,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.model_selection import cross_val_predict\n",
    "from sklearn.metrics import confusion_matrix\n",
    "\n",
    "predictions = cross_val_predict(random_forest, X_train, Y_train, cv=3)\n",
    "confusion_matrix(Y_train, predictions)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The first row is about the not-survived-predictions: __494 passengers were correctly classified as not survived__ (called true negatives) and __55 where wrongly classified as not survived__ (false positives).\n",
    "\n",
    "The second row is about the survived-predictions: __98 passengers where wrongly classified as survived__ (false negatives) and __244 where correctly classified as survived__ (true positives).\n",
    "\n",
    "A confusion matrix produces an idea of how accurate the model is."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id=\"section608\"></a>\n",
    "### 6.8 Precision and Recall"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 89,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Precision: 0.8148148148148148\n",
      "Recall: 0.7076023391812866\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import precision_score, recall_score\n",
    "\n",
    "print(\"Precision:\", precision_score(Y_train, predictions))\n",
    "print(\"Recall:\",recall_score(Y_train, predictions))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Our model predicts correctly that __a passenger survived 81% of the time__ (precision). The __recall__ tells us that __71% of the passengers tested actually survived.__"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id=\"section609\"></a>\n",
    "### 6.9 F-score"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "It is possible to combine precision and recall into one score, which is called the F-score. The F-score is computed with the harmonic mean of precision and recall. Note that it assigns more weight to low values. As a result, the classifier will only get a high F-score if both recall and precision are high."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 90,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.7574334898278561"
      ]
     },
     "execution_count": 90,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.metrics import f1_score\n",
    "f1_score(Y_train, predictions)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "There we have it, a __76% F-score.__ The score is not high because we have a recall of 71%. Unfortunately, the F-score is not perfect, because it favors classifiers that have a similar precision and recall. This can be a problem because often times we are searching for a high precision and other times a high recall. An increase of precision can result in a decrease of recall, and vice versa (depending on the threshold). This is called the __precision/recall trade-off.__"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id=\"section610\"></a>\n",
    "### 6.10 Precision Recall Curve"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "For each person the Random Forest algorithm has to classify, it computes a probability based on a function and it classifies the person as __survived__ (when the score is bigger the than threshold) or as __not survived__ (when the score is smaller than the threshold). That’s why the threshold plays an important part in this process.\n",
    "\n",
    "Let's plot the precision and recall with the threshold using matplotlib."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 91,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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yEBDREREREQBQONmL5qSSq5yRkREREREASDNzV40ZdVmP1TSOQw0RERERES9WJjOddlms9WBitrACDUMNEREREREvZhaJUeS3nWls1c/PQopABYHYKAhIiIiIurlBqVFurTllzVg+6ESP1TTMQw0RERERES9XHpKuNv2XUfLfFtIJzDQEBERERH1csP6R2NY/2iX9iKDEc5uvskmAw0RERERUS8nCALun5fl0m6xOVFsMPqhovZrV6BZv349ZsyYgalTp2LVqlUu/SdOnMC8efMwbdo0PPLII3A4HB4vlIiIiIiIvEcmE3BJRoxLu0LRvc+BtFldeXk5li1bhtWrV+PTTz/FmjVrkJub22LM73//ezz++OPYtGkTJEnC2rVrvVYwERERERF5x5zL+0Cr/nmjzXFD4pAQHezHitqmaGvAzp07MXbsWISHhwMApk2bho0bN+L+++8HABQXF8NisWDYsGEAgHnz5uGll17CwoUL212ETCZ0vHLqNTg/yNs4x8jbOMfI2zjHyFMS9cF47t7xOHm+BiE6FQYmhwHw7xxr67XbDDQVFRXQ6/XNj2NiYpCTk9Nqv16vR3l5eYeKjIhwXfea6CdRUd37UwEKfJxj5G2cY+RtnGPkSVEAUpMiWrZ14znW5iVnoihCEH5ORZIktXjcVj8REREREZG3tBlo4uLiYDAYmh8bDAbExMS02l9ZWdmin4iIiIiIyFvaDDTjxo3Drl27UF1dDbPZjM2bN2PChAnN/YmJiVCr1di/fz8AYN26dS36iYiIiIiIvEWQJElqa9D69evx2muvwW63Y8GCBViyZAmWLFmCpUuXIisrCydPnsSjjz4Ko9GIwYMH45lnnoFKpfJF/URERERE1Iu1K9AQERERERF1R917lxwiIiIiIqKLYKAhIiIiIqKAxUBDREREREQBi4GGiIiIiIgCFgMNdQvr16/HjBkzMHXqVKxatcql/6uvvsK1116La665Bvfeey/q6ur8UCUFsrbm2E+2b9+OyZMn+7Ay6inammN5eXlYtGgRrrnmGtxxxx38OUYd0tb8OnbsGObPn49rrrkGd999N+rr6/1QJQU6o9GIWbNmoaioyKXvxIkTmDdvHqZNm4ZHHnkEDofDDxW2QiLys7KyMmnSpElSTU2NZDKZpNmzZ0tnzpxp7m9oaJDGjx8vlZWVSZIkSf/617+kp556yl/lUgBqa479xGAwSNOnT5cmTZrkhyopkLU1x0RRlKZOnSp98803kiRJ0nPPPSf94x//8Fe5FGDa8zPspptukrZv3y5JkiQ988wz0gsvvOCPUimAHTp0SJo1a5Y0ePBgqbCw0KV/5syZ0sGDByVJkqQ//elP0qpVq3xcYet4hob8bufOnRg7dizCw8Oh0+kwbdo0bNy4sbnfbrfjiSeeQGxsLAAgPT0dpaWl/iqXAlBbc+wnjz76KO6//34/VEiBrq05duzYMeh0uuaNp++55x7cfPPN/iqXAkx7foaJogiTyQQAMJvN0Gg0/iiVAtjatWvxxBNPICYmxqWvuLgYFosFw4YNAwDMmzfP7e9Rf2GgIb+rqKiAXq9vfhwTE4Py8vLmxxEREbjqqqsAABaLBcuXL8eUKVN8XicFrrbmGACsXLkSgwYNQnZ2tq/Lox6grTlWUFCA6Oho/PnPf8bcuXPxxBNPQKfT+aNUCkDt+Rn28MMP49FHH8Vll12GnTt34sYbb/R1mRTgnn76aYwaNcpt3y/noF6vd5mD/sRAQ34niiIEQWh+LElSi8c/aWhowF133YWMjAzMnTvXlyVSgGtrjp0+fRqbN2/Gvffe64/yqAdoa445HA7s2bMHN910Ez755BMkJyfj2Wef9UepFIDaml8WiwWPPPII3n77bezYsQMLFy7EH//4R3+USj1Ue9+r+QsDDfldXFwcDAZD82ODweByurOiogILFy5Eeno6nn76aV+XSAGurTm2ceNGGAwGzJ8/H3fddVfzfCNqr7bmmF6vR2pqKrKysgAAs2bNQk5Ojs/rpMDU1vw6ffo01Go1hg4dCgC44YYbsGfPHp/XST3XL+dgZWWl20vT/IWBhvxu3Lhx2LVrF6qrq2E2m7F58+bm68wBwOl04p577sHVV1+NRx55pFt9IkCBoa05tnTpUmzatAnr1q3D8uXLERMTg9WrV/uxYgo0bc2x4cOHo7q6GidPngQAbNu2DYMHD/ZXuRRg2ppfqampKCsrQ15eHgBg69atzeGZyBMSExOhVquxf/9+AMC6detazEF/U/i7AKLY2Fg8+OCDWLx4Mex2OxYsWIChQ4diyZIlWLp0KcrKynD8+HE4nU5s2rQJADBkyBCeqaF2a2uO8Rc/dVV75tgrr7yCRx99FGazGXFxcfjHP/7h77IpQLRnfj3zzDN44IEHIEkSoqKi8Le//c3fZVMPcOEce/755/Hoo4/CaDRi8ODBWLx4sb/LayZIkiT5uwgiIiIiIqLO4CVnREREREQUsBhoiIiIiIgoYDHQEBERERFRwGKgISIiIiKigMVAQ0REREREAYuBhoiol2toaEB1dTUA4OOPP0Z6ejq++OILP1f1s927dyM9PR3Lly/32DGLioqQnp6Oxx9/vM2xDz/8MNLT01tsKkdERN0HAw0RUS+2a9cuTJ06FWfOnPF3KURERJ3CQENE1IsdOXKk+ewMERFRIGKgISIiIiKigMVAQ0TUSz388MP45z//CQBYvHgx0tPTm/tMJhP++te/4rLLLsPQoUMxZ84cbNiwocXzFy1ahHnz5uG9997DmDFjMGLECKxatQoA4HQ6sWLFCsyYMQNZWVkYO3YsHnroIRQUFLQ4hslkwpNPPomrrroKQ4YMwWWXXeZ2HADY7Xa89NJLmDx5MoYMGYLp06c3v96FCgsL8fvf/x7jxo3DkCFDcNVVV2HZsmUwm81tfk+++OILzJs3D9nZ2bjyyivx9ttvt/kcIiLyL4W/CyAiIv+44YYbYLVasWHDBtxzzz3o27cvnE4nAOCZZ55Bv379cM8998BsNuOdd97Bgw8+iKioKIwZM6b5GOfOncO///1v/PrXv0ZDQwPGjh0LSZLwf//3f9i6dStmz56NRYsWoaysDB988AG+++47vP/+++jXrx8A4IEHHsCePXuwaNEipKamoqioCCtXrsTevXuxceNG6HS65tdasWIFYmNjccstt0Aul2P16tV48sknodPpMHfuXADAqVOncMstt8DhcODGG29EUlIS9u7di1dffRW7du3Cu+++C7Va7fb7sXLlSjz99NMYMmQIfvvb36KmpgYvv/wyJEny1n8CIiLyAAYaIqJeavjw4di7dy82bNiAcePGYcyYMfj4448BAP3798f7778PhaLp10RWVhZuvfVWbNiwoUWgaWxsxBNPPIE5c+Y0t33++efYsmULnnzySdxwww3N7QsWLMA111yDv/3tb3jjjTdQXV2Nb7/9FjfffDN+97vfNY9LTk7GO++8g9zcXAwdOrS5PTw8HB999BGCgoIAAJMnT8aUKVPwxRdfNAeaJ598EiaTCR9++CEGDx4MALj55psxYMAAvPTSS3jrrbdwzz33uHwvGhoasGzZMmRnZ+O9996DSqUCAFx99dWYP39+l77PRETkXbzkjIiIXMycObM5zADAsGHDAAAVFRUuY6+44ooWjzds2ACFQoErrrgC1dXVzX+CgoIwevRo7Nq1C0ajEcHBwQgODsaGDRvw4YcfNi9OsGDBAqxfv75FmAGAK6+8sjnMAE3BJyoqqrmm6upq7Nu3DxMnTmwOMz9ZsmQJdDodNm7c6Pbfu3PnTjQ2NuK6665rDjMAkJ6ejgkTJrTx3SIiIn/iGRoiInIRFRXV4rFGowEA2Gy2Fu0KhQLh4eEt2vLz8+FwOC4aBMrLy9GvXz88/fTTeOSRR/Doo4/i8ccfx6BBgzBp0iTMmzcPCQkJLZ4THR3tchyNRtNcU1FREQA0X852IZVK1XxJmzuFhYUAgLS0NJe+/v37Y+vWra3+W4iIyL8YaIiIyIVM1r4T+O7GiaKI8PBwLFu2rNXnxcXFAQCmT5+Oyy67DNu3b8eOHTuwc+dOvPzyy3j99dfx5ptvYuTIke2uSRRFAIAgCG77nU5ni7MvF/rpORaLpdXjEhFR98RAQ0REHpWYmIjz588jOzu7xSViALB7926IogiVSoXGxkacPHkSiYmJmDVrFmbNmgUA+Oqrr3DffffhnXfeaRFo2pKcnAwAyM3NdemzWq0oKipC37593T43JSUFAJCXl4fLL7+8RZ+7FdeIiKj74D00RES92E9nPTx5FmLatGkQRRH/+c9/WrQXFBTg7rvvxlNPPQWFQoHz58/jpptuwuuvv95iXHZ2dova2isqKgrDhw/HN998g2PHjrXoe/PNN9HY2IipU6e6fe748eMRHByMlStXwmQyNbfn5+dj27ZtHaqDiIh8i2doiIh6sZ/uS3nvvfdavb+ko+bNm4fPP/8cK1aswNmzZ3H55Zejrq4Oq1evhsPhwGOPPQZBEJCZmYlJkybhvffeQ0NDA0aMGIHGxkZ8+OGHUCqVuOWWWzr82k888QRuvvlm3HLLLbjpppuQlJSEffv24YsvvsDgwYNx6623un2eTqfDY489hocffhgLFizA9ddfj8bGRrz77rsICQlpXrCAiIi6HwYaIqJebOrUqdiyZUvz/St//vOfu3xMhUKBFStW4PXXX8fnn3+OHTt2IDQ0FFlZWbj33nubz8AAwD//+U+8/vrr+PLLL7Fx40aoVCpkZ2fjySefxKhRozr82pmZmfjf//6Hf//73/j4449hMpmQmJiI3/zmN7jzzjubFzdwZ86cOYiIiMB///tfvPjiiwgJCcEtt9wCu92OV199tVPfCyIi8j5B4o5hREREREQUoHgPDRERERERBSwGGiIiIiIiClgMNEREREREFLAYaIiIiIiIKGAx0BARERERUcBioCEiIiIiooDFQENERERERAGLgYaIiIiIiAIWAw0REREREQUsBhoiIiIiIgpY/x9DaGbELGITBgAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 1008x504 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "from sklearn.metrics import precision_recall_curve\n",
    "\n",
    "# Getting the probabilities of our predictions\n",
    "y_scores = random_forest.predict_proba(X_train)\n",
    "y_scores = y_scores[:,1]\n",
    "\n",
    "precision, recall, threshold = precision_recall_curve(Y_train, y_scores)\n",
    "def plot_precision_and_recall(precision, recall, threshold):\n",
    "    plt.plot(threshold, precision[:-1], \"r\", label=\"precision\", linewidth=5)\n",
    "    plt.plot(threshold, recall[:-1], \"b\", label=\"recall\", linewidth=5)\n",
    "    plt.xlabel(\"threshold\", fontsize=19)\n",
    "    plt.legend(loc=\"upper right\", fontsize=19)\n",
    "    plt.ylim([0, 1])\n",
    "\n",
    "plt.figure(figsize=(14, 7))\n",
    "plot_precision_and_recall(precision, recall, threshold)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We can see in the graph above that the recall is falling of rapidly when the precision reaches around 85%. Thus, we may want to select the precision/recall trade-off before this point (maybe at around 75%).\n",
    "\n",
    "Now we are able to choose a threshold, that gives the best precision/recall trade-off for the current problem. For example, if a precision of 80% is required, we can easily look at the plot and identify the threshold needed, which is around 0.4. Then we could train the model with exactly that threshold and expect the desired accuracy.\n",
    "\n",
    "__Another way is to plot the precision and recall against each other:__"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 92,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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XCUnJ+uFF17Qbbfdpr59+2ro0KHKz8+XJO3Zs0f33HOPBg8erL59+2rkyJF6/vnn5Xa7a7zf4sWLde2116pfv34aPHiwpk+frh9++MF//ET3SC1ZskTXXHONzjnnHPXr108TJkzQe++9F/C6J7pHavfu3br77rv9dQ0fPlxz585VeXm5/5yjX9trr72mV155RSNHjvSfO2/ePDEhBACaN+6RAgA0mPz8fN1888268MILddVVV2nNmjV66aWXtG7dOs2fP99/3vPPP6/U1FQ98MADysvLU2Jionbs2KEJEybIbrdrwoQJatu2rdasWaO5c+fq22+/1XPPPee/d+mpp57SCy+8oLPOOkt33nmnqqqqNH/+fF1//fVauHChunXrVqO2d999V7/73e900UUX6Xe/+53cbrf+/e9/66677lJoaKguu+yyE35NP/zwg66//np5PB5NmDBBHTt21FdffaXnn39eq1ev1muvvabw8HD/+a+88orcbrcmTZqkuLg4LVq0SH/6058UGxurCRMm1PN3HADQWAhSAIAGc+DAAd1www267777JEmTJ0+W0+nUP/7xD7333ntKS0uTJIWFhemvf/1rwEIQf/jDH2QYht566y0lJiZKkiZNmqSUlBQ98cQTevfdd5Wenq6dO3fqxRdf1EUXXaTnnntODodDkjR06FBdddVVevXVV/Xwww/XqG3x4sWKjo7W888/7w9ko0eP1nXXXafNmzefNEg9/PDDKisr0xtvvKGUlBT/19WjRw8988wzeuWVVzRt2jT/+QcPHtSKFSvUtm1bSdKoUaN04YUXavHixQQpAGjGmNoHAGgwdrtdt99+e8C+oyFj+fLl/n39+vULCFEHDx7UZ599pvPPP1+hoaEqLi72/xkxYoRsNpvef/99SdKHH34on8+nqVOn+kOUJPXq1cs/wnQiSUlJKisr0x/+8Adt3rxZkhQXF6fs7GzNmDHjhNcUFxdrzZo1uuiii/wh6qiMjAxFRUUpOzs7YP/AgQP9IUqS2rRpo8TERO3fv/+E7wEAaB4YkQIANBiXy6W4uLiAfW3atFFcXJx2797t33d80JCkXbt2yTAMvffeezXuWTpq7969kqrvRZKkrl271jinb9++J61txowZ+v7775WVlaWsrCy5XC5deOGFSk9P1+DBg094zdH36t69e41jYWFh6ty5s/+ck31tR8+tqqo6aW0AgKaPIAUAaDBhYWEn3O/1egNGj/77OU0+n0+SlJ6ervHjx5/wNaKjoyXJv/CEzWYzVZvL5dJ//vMfffPNN1q1apVycnL01ltv6T//+Y+mTJmiWbNm1bjmaF0ney+v11vjazZbFwCgeSBIAQAazL59+1RZWRmw+EJBQYFKS0tPOIJ0VIcOHSRJHo+nxuhQVVWVPvjgAzmdzoBzd+7cqfbt2wec+9hjjyk8PFwzZ86s8R5bt25VRUWF+vfvr/79++u3v/2t8vPzddNNN2nBggW68847FRMTE3BNp06dJEnbtm2r8XqVlZXKzc094cIWAIDgwz1SAIAG43a79dprrwXse+GFFyRJV1xxxUmvczqd6tevnz744AP//UtHvfrqq7rrrrv00UcfSapenlySsrKyApYU3759u15//XUVFBSc8D3+53/+R9OnT1dZWZl/X2Jiotq1ayebzVZjlEySEhIS1K9fP3388cfasGFDwLG///3vOnLkiEaMGHHSrwsAEDwYkQIANBibzaZnnnlGP/30k1JSUvT5559r+fLlGjVqlC6++OIa9xMdb/bs2Zo8ebImTpyoCRMmqEuXLlq7dq3eeust9e7dW5MmTZIk9ejRQzfeeKNeffVVTZ06VSNGjFBZWZlef/11xcbGKjMz84Svf9ttt+nOO+/UpEmTNHbsWEVGRurLL7/Up59+qgkTJvinDp6sruuvv14TJ05Ux44dtWbNGr3zzjtKSUnRDTfccPrfOABAk0eQAgA0mNDQUL3yyit6+OGHtXTpUrVv31533323brrpplqv7d27t/7973/r2Wef1ZIlS1RSUqKkpCTddNNNuvXWW9WqVSv/ub///e/VvXt3ZWVl6YknnlBcXJwGDBigmTNnql27did8/VGjRunZZ5/V3//+dz333HM6cuSIunTponvvvVdTpkypta6//vWvevPNN1VWVqYOHTrojjvu0K9//WtFRESY/0YBAJodm8Gj1QEADWDKlClau3at1q9fb3UpAADUO+6RAgAAAACTCFIAAAAAYBJBCgAAAABM4h4pAAAAADCJESkAAAAAMIkgBQAAAAAmEaQAAAAAwCSCFAAAAACYRJACAAAAAJMIUgAAAABg0v8PdY88wngDorgAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 1008x504 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "def plot_precision_vs_recall(precision, recall):\n",
    "    plt.plot(recall, precision, \"b--\", linewidth=3)\n",
    "    plt.xlabel(\"precision\", fontsize=19)\n",
    "    plt.ylabel(\"recall\", fontsize=19)\n",
    "    plt.axis([0, 1.2, 0, 1.4])\n",
    "\n",
    "plt.figure(figsize=(14, 7))\n",
    "plot_precision_vs_recall(precision, recall)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id=\"section611\"></a>\n",
    "### 6.11 ROC AUC Curve"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Another way to evaluate and compare binary classifiers is the ROC AUC Curve. This curve plots the true positive rate (also called recall) against the false positive rate (ratio of incorrectly classified negative instances), instead of plotting the precision versus the recall values."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 93,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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WVBFR8uckIiIiIioY2J5aiW8NissxUWoXrMKktvtd0ah08EwbTqM09OgjIiQOSERERERUcfG8hRinqpKWIoKYV+3BpLX63dGER4O44w0GFKnQakPlzkhEREREREBLE8h4y4vh6UwH/aN6wGfz++MMjIKxpmzYJg6FYowncwJiYiIiIjoQixPMnOfOY2qgjzUbN4ESJLfGaXBANOsDERPnAyFVitzQiIiIiIi8oflSSb1J06gqiAPju1bA5YmldkMU3omosZPgEKtkTkhERERERFdCstTC3MdPYqq/G9Q+92OgDPqmFiYMrMQlTIOgopfEiIiIiKi1og/qbcQ5+FDqMrPQ93uXQFnNJ27wJSZhcjRYyAolTKmIyIiIiKiK8Xy1IwkSYLzwH5U5efCuX9fwDlNt+4wZ2UjYsQoCAqFjAmJiIiIiOhqsTw1A0mSULdnNywFeXAeOhhwTturN8xZOQhPSmZpIiIiIiJqY1ieroEkSajd+R2q8nNRf/THgHNh/frDnJUD/ZChEARBxoRERERERNRcWJ6ugiSKcHy7HZaCXNSfOBFwTpcwCOasHOgGJrA0ERERERG1cSxPV0Dy+VCzdTMsBflwnzkdcE4/NBHmzBzo+veXMR0REREREbUklqcgSF4v7Js2wlKYD095WcC58GHDYc7MRljvPjKmIyIiIiIiObA8XYLo8cC+fi0sRQXwVlX5HxIERIwcBXNmNrTde8gbkIiIiIiIZMPy5IdYX4/qtathKS6Ez2bzPyQIiByTAlNGNrRdusiaj4iIiIiI5MfydAHR5YJt1QpYS4rhq7H7H1IqETV2HEzpWdDExckbkIiIiIiIQoblCYCvrg62FctgXVoCsbbW74ygUiFqwiSY0jOgNneSOSEREREREYVahy5PPocD1mUlsC1fBtHp9DsjaDSInjQFxrR0qI1GmRMSEREREVFr0SHLk9duh7W0GLaVKyDVu/zOCNowGKamwjgjDaroaJkTEhERERFRa9OhypPHaoW1pBDVa1ZDcrv9zih0OhimzYBx+kwoIyJkTkhERERERK1VhyhPnqpKWIoKYV+3BpLX63dGER4O44w0GFKnQakPlzkhERERERG1du26PLnLymApyod94wbA5/M7o4yMgjFtFgxTUqEIC5M5IRERERERtRXtsjzVnz4NS2EeajZvAiTJ74zKaIQxLQPREydBodXKnJCIiIiIiNoa2ctTXl4eFi9eDK/XizvuuAO33XZbo+379u3Dk08+idraWowaNQp/+9vfoFIFH3PA90tx7J9bA5cmsxmm9ExEjZ8IhVp9TZ8LERERERF1HAo536ysrAyvvvoqPvvsM3z99df4/PPPcfjw4UYzjz76KJ566imUlJRAkiR88cUXV/Qencp/9Fuc1LFxiLvzLvR+9oVzp+ixOBERERER0RWQ9cjThg0bkJKSAoPBAABIS0tDcXEx5s6dCwA4deoUXC4Xhg0bBgC48cYb8cYbb+DWW28N+j00ZnOjx+rYOBimpiI8eTgEhaxdkToAhUIIdQTqYLjPkZy4v5GcuL+RHK51P5O1PJWXlyMmJqbhcWxsLHbt2hVwe0xMDMrKyq7oPZKef/bagxIFyWzmcvYkL+5zJCfubyQn7m/UFsh6KEYURQjCz21PkqRGjy+3nYiIiIiIKFRkLU/x8fGoqKhoeFxRUYHY2NiA2ysrKxttJyIiIiIiChVZy9O4ceOwceNGWCwWOJ1OlJaWYtKkSQ3bu3btCq1Wi+3btwMAvvnmm0bbiYiIiIiIQkWQpABrereQvLw8vPPOO/B4PLjpppswZ84czJkzB/PmzUNiYiL279+P+fPnw+FwYMiQIViwYAE0Go2cEYmIiIiIiJqQvTwRERERERG1RVy7m4iIiIiIKAgsT0REREREREFgeSIiIiIiIgoCyxMREREREVEQ2mR5ysvLQ0ZGBmbOnIklS5Y02b5v3z7ceOONSEtLw5NPPgmv1xuClNReXG5/W7ZsGa6//nrk5OTggQceQHV1dQhSUntxuf3tvFWrViE1NVXGZNReXW6fO3LkCG6//Xbk5OTgrrvu4vc4uiaX29/27NmDX/7yl8jJycG9994Lu90egpTUnjgcDmRlZeHkyZNNtl1VZ5DamLNnz0pTp06VrFarVFtbK2VnZ0uHDh1qNJOZmSnt2LFDkiRJeuKJJ6QlS5aEICm1B5fb32pqaqTx48dLZ8+elSRJkl577TXpmWeeCVVcauOC+f4mSZJUUVEhzZo1S5o6dWoIUlJ7crl9ThRFaebMmdLq1aslSZKkl156SXrxxRdDFZfauGC+x/3617+WVq1aJUmSJC1YsEBauHBhKKJSO/Hdd99JWVlZ0pAhQ6QTJ0402X41naHNHXnasGEDUlJSYDAYoNfrkZaWhuLi4obtp06dgsvlwrBhwwAAN954Y6PtRFficvubx+PB008/jbi4OADAwIEDcebMmVDFpTbucvvbefPnz8fcuXNDkJDam8vtc3v27IFer2+4Yf19992H2267LVRxqY0L5nucKIqora0FADidToSFhYUiKrUTX3zxBZ5++mnExsY22Xa1naHNlafy8nLExMQ0PI6NjUVZWVnA7TExMY22E12Jy+1vRqMRM2bMAAC4XC68++67mD59uuw5qX243P4GAB9//DEGDx6M5ORkueNRO3S5fe748ePo1KkT/vznP+OGG27A008/Db1eH4qo1A4E8z3u8ccfx/z58zFhwgRs2LABv/rVr+SOSe3Is88+i1GjRvnddrWdoc2VJ1EUIQhCw2NJkho9vtx2oisR7P5UU1ODe+65BwkJCbjhhhvkjEjtyOX2t4MHD6K0tBQPPPBAKOJRO3S5fc7r9WLLli349a9/jf/85z/o3r07nn/++VBEpXbgcvuby+XCk08+iQ8//BDr1q3DrbfeisceeywUUakDuNrO0ObKU3x8PCoqKhoeV1RUNDoUd/H2yspKv4fqiIJxuf0NOPebi1tvvRUDBw7Es88+K3dEakcut78VFxejoqICv/zlL3HPPfc07HtEV+ty+1xMTAx69uyJxMREAEBWVhZ27dole05qHy63vx08eBBarRZJSUkAgFtuuQVbtmyRPSd1DFfbGdpceRo3bhw2btwIi8UCp9OJ0tLShnOxAaBr167QarXYvn07AOCbb75ptJ3oSlxuf/P5fLjvvvuQnp6OJ598kkc56Zpcbn+bN28eSkpK8M033+Ddd99FbGwsPvvssxAmprbucvvc8OHDYbFYsH//fgDAihUrMGTIkFDFpTbucvtbz549cfbsWRw5cgQAsHz58obiTtTcrrYzqFo6WHOLi4vDQw89hNmzZ8Pj8eCmm25CUlIS5syZg3nz5iExMREvv/wy5s+fD4fDgSFDhmD27Nmhjk1t1OX2t7Nnz2Lv3r3w+XwoKSkBAAwdOpRHoOiqBPP9jag5BbPPLVq0CPPnz4fT6UR8fDxefPHFUMemNiqY/W3BggV48MEHIUkSzGYznnvuuVDHpnbmWjuDIEmSJENOIiIiIiKiNq3NnbZHREREREQUCixPREREREREQWB5IiIiIiIiCgLLExERERERURBYnoiIqFVrL+satZfPg4ioI2tzS5UTEbV1t99+e8AbP3bq1Anr168P6nW++uorPPHEE9i4cSNMJlNzRmxw8uRJTJs2rdFzCoUCUVFRSE5Oxh/+8Idmve/P7bffDr1ej3feeQcA8NZbb8FoNOK2227zu70lPP744/jPf/7T6DmFQoHIyEgMHjwYDz74IIYNGxb067ndbrz44otISUnB9OnTrzlfWVkZbr31Vnz11VeIjo5GamoqTp065Xc2OTkZX3zxBTZv3txkCV6lUgmDwYDrrrsODz/8MHr06AEAePPNN/HWW281mhUEAeHh4ejfvz/uu+8+TJkyBcC5m5z+6le/wpdfftli+yARUWvC8kREFAIjRozAY4891uR5tVodgjSX9/DDD2PMmDEAAFEUUVZWhldffRWzZ89GYWEh4uLimuV9nn76aSgUP58U8eabb+JPf/pTwO0tpXv37nj55ZcbHnu9Xvz44494++23cdddd6GoqCioO9EDQHl5OT755BOMGjWqWbI9/fTTuO222xAdHd3wXFpaGn772982mQ0PD2/0eMGCBejTpw+Aczf5Pn78OF566SXccccdKCwshE6nAwCEhYXho48+avhzPp8Pp0+fxrvvvovf/e53+Pe//42EhATExMTgF7/4BZ599lm88sorzfL5ERG1ZixPREQhEBUVdUVHL0KtZ8+eTfLGxsbi1ltvxTfffIN77rmnWd6nX79+17S9uYSFhTX5fEeNGoUuXbrgt7/9LZYuXdpwNExOW7duxdatW/Haa681er5Tp05B7U/9+/dvdLPlkSNHQqlU4tFHH8WKFSuQmZkJ4NyRtotfb+TIkUhMTERaWhpyc3ORkJAAALjjjjswfvx47N27F4MHD76mz4+IqLXjNU9ERK3Url27MGfOHIwaNQpDhw5FWloa/vnPfwacr6iowB/+8AeMGTMGycnJuPXWW5ucHrh+/XrcfPPNSEpKwqRJk/D666/D5/NdVb7zp+tdeMrY1q1bcdttt2HEiBEYN24c/v73v6O2tjbojLfffjvuvfdeAMDAgQMBAC+++CJSU1ObbJ82bRqeeuqpRpmqq6sxdOhQ/Otf/wIA1NXV4ZlnnsG4ceOQlJSE22+/HXv37r2qzxcAIiIimjx3qa/Thac9/uEPf8Dtt9/e8Ofy8/ORnZ2NxMRETJ8+HZ988sll3//9999HamoqwsLCrvpzuJi/r2Mg/j7/qKgojB8/Hv/4xz+aLRMRUWvF8kREFAKSJMHr9Tb557zTp09j9uzZ0Ov1eP3117Fo0SL07t0bTz/9NPbv3+/3NZ944gkcP34cCxYswNtvvw2dTod7770XNpsNALBx40bMmTMH3bp1w1tvvYW77roLH3zwAf7nf/7nqj6Ho0ePAgC6desGAFi9ejVmz56NmJgYvPrqq/j973+PgoIC3HvvvRBFMaiMF/r8888BnCtMF1+DAwCZmZkoLS1tVP6WLl0KAJg5cyYkScL999+PgoICPPjgg3j99deh0Whw++234/jx45f9/C78ujidTnz//fd45plnEBER0VCILvd1io2Nbcj+8MMP4+mnnwYA/Oc//8EjjzyC0aNHY/HixfjFL36BBQsW4H//938D5nE4HFizZg1mzpzZZJu//SnYUnzx19Hf519fX4/Dhw/jiSeegEqlajhCdV5aWhqWLVsGt9sd1HsSEbVVPG2PiCgEVq9e7XehhfOLPxw6dAjDhg3Dyy+/3HAdVHJyMsaMGYNt27Y1nDJ1oW3btuH+++9vOErTv39/fPDBB3A6nTAYDHjttdeQnJyMV199FQAwadIkREdH44knnsBdd93V5IfnC4mi2FDu3G43Dh8+jGeeeQZ6vR45OTkAgNdffx1JSUmNTinr1q0b7r77bqxatQqpqamXzXih86eNde7c2e/pYNnZ2XjnnXewZcsWjB07FgBQVFSESZMmISoqCmvXrsWmTZvwwQcfYNy4cQCAiRMnIjMzE4sXL8aCBQsCfr6HDh1q8vVRq9UYPnw4PvroI8THxzfMXe7rNGjQIADnTn3s168fRFHEwoULkZ2d3XDkbMKECRAEAW+//TZuvfVW6PX6Jpm2bdsGr9fr97/FZ599hs8++6zRc3q9Hjt27Gj03IVfR5fLhT179uDFF19EXFxcwyIQwLkjdhd//gqFAoMHD8Z7773XZNvgwYPhcrmwc+dOjB492v9/VCKidoDliYgoBEaOHIknnniiyfNRUVEAgMmTJ2Py5Mmor6/H/v37cfToUXz//fcAEPC3+8OHD8cbb7yBAwcONPz584tSOJ1O7Nq1Cw899FCjI1yTJk2CKIrYvHnzJcvTQw891OS57t2749VXX0VcXBxqa2uxd+/eJotgTJw4EdHR0di6dStSU1MvmfFK9e/fHwMGDEBRURHGjh0Lm82GzZs348UXXwQAbN68GTqdDqNHj270OU+YMAErVqy45Gv36NEDCxcuBHDudLYXX3wRXbt2xVtvvdVooYar+Tr9+OOPKC8vx5QpU5p8Ld544w3s2rULKSkpTf7c+dPqzhe3C6Wnp+Ouu+5q9JxSqWwy91//9V9Nnhs4cCBefPHFRoUtLCwMn376KQDAYrFg4cKFEEURb7zxBrp27drkNc4/d+rUKZYnImrXWJ6IiEIgMjKy0YX7F/P5fHj++efx+eefw+PxoEePHg2rtQW6X9Crr76KRYsWoaioCAUFBVCr1bjxxhsxf/582O12iKKIV155xe+qaBUVFZfM+8c//rHhB3qVSgWTydRohb2amhpIkgSz2dzkz5pMJjgcjstm1Gg0l8zgT3Z2Nj744AM8/fTTWLp0KdRqNaZOnQoAsNlscDqdGDp0aJM/d7lVDbVabcPXJzExEf3798cNN9yAuXPn4qOPPmpY8e9qvk7nT1F85JFH8MgjjzTZHuhrUVNTA41G47cUmUymS+5P573wwgvo27cvgHP/DWJiYvx+zRQKRaPXGzp0KLKysnD33Xfj3//+d5MjY+evwaqpqblsBiKitozliYioFVq8eDG++OILvPDCC5g8eTL0ej2cTmfDQgj+GAwGPPnkk3jyySexb98+5Obm4oMPPkC3bt1w6623AgDuv//+JvdtAnDZZbe7d+9+yR/OIyMjIQgCqqqqmmyrrKxsOCXvUhmvZsW+zMxMLFy4ENu2bUNxcTGmTZvWsNx2ZGQkzGZzs9wTqm/fvrj//vvx2muvYcmSJQ0LP1zN1ykyMhIA8NRTTyEpKanJ9kBHAA0GA9xuN9xu91UVzfOfRzAl62JmsxlPPPEEHn30Ubzxxht4/PHHG2232+0NGYmI2jMuGEFE1Ap99913GDp0KNLT0xt+y7927VoA/o9oWCwWTJkypWHBhEGDBuGxxx5Dly5dcObMGURERCAhIQEnTpxAYmJiwz9qtRoLFy7E2bNnrylveHg4Bg0ahOLi4kbPr127FjU1NRgxYsRlM/pzuXs6de3aFcOGDUNeXh42bdqE7Ozshm0jR46ExWKBXq9v9Dnn5eUhNzf3ij/H89eFvfnmm7BarQCC+zpdfKSoT58+MBgMKCsra5TLZrPh9ddfbzhKd7HOnTsDwDV/ra5WTk4ORowYgU8//RRHjhxptK2srAzAzxmJiNorHnkiImqFEhMT8d577+HTTz/FgAED8P3332PRokUQBAEul6vJvMlkQs+ePfE///M/qK2tRefOnbFq1SqcOnUKM2bMAADMmzcPv/vd7xAREYEZM2bAarXitddeg0KhwIABA6458+9//3s88MADePDBB3HjjTfizJkzWLhwIYYPH45JkyZBqVReNuPFoqKisH37dowaNQrJycl+Z7Kzs/Hss88iMjKyYWEIAJg6dSoSExNxzz33YO7cuejcuTNKS0uxZMkS/O1vf7viz0+j0eCRRx7BQw89hDfffBNPPfVUUF+n80eaNmzYgF69eiEhIQG///3v8fzzzwMAxo4di5MnT+KVV15Br169Ah55GjlyJNRqNXbs2IEePXpccf7m8Pjjj+OWW27BCy+80OiI3o4dOxARERHwa0RE1F6wPBERtUL33HMPKioq8NZbb6G+vh69evXCX/7yF+Tn5zdZQe28hQsX4sUXX8TLL78Mm82G3r1745VXXmkoFNOmTcPbb7+NRYsW4auvvkJERATGjRuHP/7xjw2nul2L1NRULFq0CIsWLcIDDzwAg8GArKwsPPTQQw1HXy6X8WJz587Fa6+9hm3btmHDhg1+Z9LT0/Hcc88hLS2t0bVMSqUS//jHP/Dyyy/jpZdegsPhQM+ePbFgwQLceOONV/U5ZmRk4OOPP8bnn3+OW2+9NaivU0REBObMmYNPP/0UO3bsQF5eHn7zm98gLCwMH374Id5//30YDAbMmjULDz30EARB8Pve579e69evx/XXX39V+a9VcnIyMjIyUFBQgHXr1mHChAkAzt0/bMqUKZe9loyIqK0TpEBXtBIREVGrsnnzZtx7771Yt26d3xvWhkJVVRUmT56ML774wu8y6kRE7QmveSIiImojxowZg5EjRza5p1Moffzxx5g2bRqLExF1CDzyRERE1IacPn0av/nNb/DVV1+FfHW78vJy3HLLLfjXv/7ld8lzIqL2huWJiIiIiIgoCDxtj4iIiIiIKAgsT0REREREREFgeSIiIiIiIgoCyxMREREREVEQWJ6IiIiIiIiCwPJEREREREQUhP8PnC7JYdryyzAAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 1008x504 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "from sklearn.metrics import roc_curve\n",
    "\n",
    "# Compute true positive rate and false positive rate\n",
    "false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_train, y_scores)\n",
    "\n",
    "# Plotting them against each other\n",
    "def plot_roc_curve(false_positive_rate, true_positive_rate, label=None):\n",
    "    plt.plot(false_positive_rate, true_positive_rate, linewidth=3, label=label)\n",
    "    plt.plot([0, 1], [0, 1], 'r', linewidth=4)\n",
    "    plt.axis([0, 1, 0, 1])\n",
    "    plt.xlabel('False Positive Rate (FPR)', fontsize=16)\n",
    "    plt.ylabel('True Positive Rate (TPR)', fontsize=16)\n",
    "\n",
    "plt.figure(figsize=(14, 7))\n",
    "plot_roc_curve(false_positive_rate, true_positive_rate)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The red line represents a purely random classifier (e.g. a coin flip). Thus, the classifier should be as far away from it as possible. The Random Forest model looks good.\n",
    "\n",
    "There's a tradeoff here because the classifier produces more false positives the higher the true positive rate is."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id=\"section612\"></a>\n",
    "### 6.12 ROC AUC Score"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The ROC AUC Score is the corresponding score to the ROC AUC Curve. It is simply computed by measuring the area under the curve, which is called AUC.\n",
    "\n",
    "A classifier that is 100% correct would have a ROC AUC Score of 1, and a completely random classifier would have a score of 0.5."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 94,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ROC-AUC-Score: 0.9392302857934149\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import roc_auc_score\n",
    "r_a_score = roc_auc_score(Y_train, y_scores)\n",
    "print(\"ROC-AUC-Score:\", r_a_score)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We got a __93% ROC AUC Score__"
   ]
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
 "nbformat_minor": 4
}
