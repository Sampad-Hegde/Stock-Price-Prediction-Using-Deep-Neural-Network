# Stock Price Prediction Using Deep Neural Network i.e, RNN (stacked LSTM).
Indian Stock Market Prediction using Deep Neural Network i.e, Recurrent Neural Network like LSTM. You Can Select any 
Indian Company or that should be listed in NSE for training and testing the Model.

This Project includes Data collection from web (WEB-SCRAPING). That collects the latest data available.

## How to run this project?
On Windows,
```bash
python -m venv stock

stock\Scripts\activate

pip install -r requirements.txt

cd code

cd Models

python Individual_Model.py
```
On Linux / Mac,
```bash
python3 -m venv stock

source stock/bin/activate

pip install -r requirements.txt

cd code

cd Models

python Individual_Model.py
```

### ! When you run the Individual_Model.py for the first time it will train the model and predicts next n days after you run the same file again it uses pre-trained model and predicts next n days price
### If You have the latest Nvidia GPU (any GTX or RTX series Card) then install CUDA drivers that will reduce the training time significantly.
## Data source 
I have written 2 web scraping python script file 
- First file yahoo_search.py : that uses yahoo search api to get official name of the stock from common name that we know

    For example Tata Motors (Common Name) - TATAMOTOS.NS (Trading Symbol assigned by NSE)
  
- Second File yahoofinscraper.py : That collects the data you can tweak these parameters but, I have collected Past 10 years (max data that you can get) Data to train the model.

### Tunable Parameters:

- ticker: Trading Symbol of the stock should correspond with yahoo website
- result_range: Valid Ranges "1d","5d","1mo","3mo","6mo","1y","2y","5y","10y","ytd","max"
- start: Starting Date
- end: End Date
- interval:Valid Intervals Valid intervals: [1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo]

## Model Architecture

|Layer Number| Layer Type | Number of Neurons |
| :------ | ------ | -------: |
| 1 | Input | 60 |
| 2 | LSTM | 50 |
| 3 | LSTM | 50 |
| 4 | LSTM | 50 |
| 5 | Dense and Output | 1 |

## Data Flow

## Collected Dataframe :
Date,Open,High,Low,Close,Volume<br>
2011-04-27 09:15:00,249.13,251.86,246.58,247.25,6580068.0<br>
2011-04-28 09:15:00,248.14,248.98,243.0,243.84,6905431.0<br>
2011-04-29 09:15:00,243.19,256.58,240.22,244.74,6356607.0<br>
2011-05-02 09:15:00,245.66,246.16,241.16,242.47,5867512.0<br>
.<br>
.<br>
.<br>
(2461, 6)


## Selected only Closing Price
| Close |
| ----- |
| 251.86 |
| 248.98 |
| 256.58 |
| . |
| . |
| . |
|(2461, 1) |

## After Passing Through Min Max Scaler (0, 1)
| Close |
| ----- |
| 0.252 |
| 0.853 |
| 0.365 |
| . |
| . |
| . |
|(2461, 1) |

## Splitting of Data into train and test Set (we have used as test set as validation set)
| Close |
| ----- |
| 0.252 |
| 0.853 |
| 0.365 |
| . |
| . |
| . |
| (1969, 1) -> Train set (80% of data) |

| Close |
| ----- |
| 0.252 |
| 0.853 |
| 0.365 |
| . |
| . |
| . |
| (492, 1) -> Test/Validation set (20% of data) |

## forming time series data of X_train and X_Test Y_test and Y_train
| X | Y |
| :----------- | ---------------: |
| [1st to 60th data] |    61th value|
| [2nd to 61th data] |     62nd value|
| . | . |
| . | . |
| . | . |
| (1969, 60) | (462,) |

## Reshaping the data that can be fed into Neural Network

| X | Y |
| ------ | ----- |
| [[1st value ],[2nd value],...,[60th value]]   |       [61th value], |
| [[2nd value ],[2nd value],...,[61th value]]   |       [62nd value], |
| .                                             |            .        |
| .                                             |            .        |
| .                                             |            .        |      
| (1969, 60,1)                                  |       (462,1)       |

## Model Predicted Value will
input : [[0.2], [0.35], [0.89], ....] size = 60   
output : [[0.8965]]

## Inverse Transform
Input :[[0.8965]] output: [[598.25]]


