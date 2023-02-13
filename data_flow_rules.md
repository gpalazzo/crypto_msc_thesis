## Documentation of Data flow and Business Rules
1. pull crypto pairs from Binance in a pre-defined time interval (1 minute, 5 minutes, 15 minutes, ...)
2. define the target coin, volume bar size, amount of bars looking ahead to predict and parameter tau for labeling
3. adjust data types (float, datetime, ...)
4. for the target coin
    - filter only the target coin
    - transform prices into log returns
    - accumulate volume up to the volume bar size threshold
    - define start and end times within the volume bar
    - get target time and log return of target time using the prediction amount of bars ahead
    - create label using the Lopez de Prado method: "top" if $R_{i+1}$ >= $stdev_i$ * tau else "bottom"
5. for the features
    - exclude target from the dataset
    - transform prices into log returns
    - loop over each start and end time windows (defined in the target step) and calculate features
        - features in each window must be summarized into a single data point, so if the window has 50 rows it will become only 1 row per window
6. merge both target and features into master table and drop null data
    - it will reduce the dataset for the least recent coin. example: if we have 3 coins (A, B and C) and A started in 2017, B in 2018 and C in 2019, the dataset will start in 2019
7. build and run model, and generate model reporting
8. build portfolio P&L and generate portfolio reporting
9. compare portfolio with benchmark index
