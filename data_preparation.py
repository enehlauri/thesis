# Editing pandas dataframe to receive desired output array
# Output has 975 columns, 1 column for each trading day
import pandas as pd
import numpy as np
import copy 
from tqdm import tqdm

N_TOP_ACTIVE = 100
TRADING_DAYS_IN_PERIOD = 975
periods = [1, 5, 20]
N = TRADING_DAYS_IN_PERIOD - np.max(periods) - 1
testing_period = 250
    
def prepair_investor_data(stock_data):
    
    ready_data = []
    vol_data = stock_data.drop(['price'],
                                 axis=1).sort_values(by='trading_date')
    price_data = stock_data.drop(['rvolume'],
                               axis=1).sort_values(by='trading_date')
    
    # Drop duplicate activities for same investor in a same day. 
    vol_data = vol_data.groupby(by=['trading_date', 'owner_id'],
                                as_index=False).sum()
    price_data = price_data.groupby(by=['trading_date', 'owner_id'],
                                as_index=False).mean()
    grouped_data = vol_data
    grouped_data['last_price'] = price_data['price']
    
    # Get list of top active traders in trading period
    start_of_trading = grouped_data['trading_date'].unique()[-934]
    end_of_trading = grouped_data['trading_date'].unique()[-testing_period]
    
    grouped_trading_data = grouped_data[
        grouped_data['trading_date']> start_of_trading]
    
    grouped_trading_data = grouped_trading_data[
        grouped_trading_data['trading_date']<= end_of_trading]
        
    grouped_trading_data = grouped_trading_data[
                           grouped_trading_data['rvolume'] != 0]
    
    top_active = grouped_trading_data.groupby(['owner_id'])['owner_id']\
                .count().nlargest(N_TOP_ACTIVE).reset_index(name='count')
    
    active_ids = top_active['owner_id']

    # Clear all but the top active traders' daily net volumes
    to_drop = list()
    for i in tqdm(range(grouped_data.shape[0])):
        if grouped_data['owner_id'][i] not in list(active_ids):
            to_drop.append(i)
    active_data = grouped_data.drop(to_drop).reset_index(drop=True)
    
    # Output matrxi for the prediction
    dmatrix = np.zeros([N_TOP_ACTIVE, TRADING_DAYS_IN_PERIOD])
    
    # Input 4, price matrix
    pricematrix = np.zeros_like(dmatrix)
    
    days = trdates.reset_index(drop=True)
    ids = pd.Series(active_data['owner_id'].unique()).reset_index(drop=True)
    
    for _, row in tqdm(active_data.iterrows()):
        matrix_i = ids[ids == row[1]].index[0] # row i for the owner_id
        matrix_j = days[days == row[0]].index[0] # column j for the day
        if row[2] > 0: 
            dmatrix[matrix_i][matrix_j] = 1
            pricematrix[matrix_i][matrix_j] = row[3]*row[2]
        elif row[2] < 0:
            dmatrix[matrix_i][matrix_j] = -1
            pricematrix[matrix_i][matrix_j] = row[3]*row[2]    
            
    # Making input 4 ready
    last_price = copy.deepcopy(pricematrix.T[0])
    timematrix = np.zeros_like(dmatrix)
    days_now = np.zeros_like(last_price)
    
    for col_i in tqdm(range(len(pricematrix[0]))):    
        for element_i in range(len(pricematrix.T[col_i])):
            if pricematrix[element_i][col_i] == 0:
                
                days_now[element_i] += 1
                pricematrix[element_i][col_i] = last_price[element_i]
            else:
                days_now[element_i] = 0
                last_price[element_i] = pricematrix[element_i][col_i]
                
            timematrix[element_i][col_i] = days_now[element_i]
        
    ready_data.append([])
    ready_data[0].append([dmatrix, pricematrix, timematrix])
        
    return ready_data, ids

def intra_day_vola(data):
# For input 1 intraday vola
    data['log_return'] = np.log(data['price'] /\
                                data['price'].shift(1))  
    data = data[1:]
    n_per_day = 0
    ref_day = data['datetime'][1].date()
    square_sum = 0
    
    # Daily intraday vola
    result = list()
    for _, row in data.iterrows():
        
        if row[0].date() != ref_day:
            result.append( (square_sum/(n_per_day -1 ))**(1/2) )
            n_per_day = 0
            ref_day = row[0].date()
            square_sum = 0
        
        n_per_day += 1
        square_sum += row[2]**2
    
    return np.asarray(result)

def normalize_inputs(iX):
    days = 20
    i = 20
    X = copy.deepcopy(iX)
    
    max_eurovolume = np.max(abs(iX[:-testing_period,100:200]),axis=0)
    while(i < (iX.shape[0])):        
        #X[i,100:200] = (iX[i,100:200] - mean)/std
        X[i,100:200] /= max_eurovolume
        
        # Normalize time since last action by traning period length
        X[i,200:300] /= 684
        
        # Normalize number of active investors
        X[i,300] /= N_TOP_ACTIVE
        
        # Normalize total volume
        mean_totvol = np.mean(iX[i-days:i,-3:],axis=0)
        std_totvol = np.std(iX[i-days:i,-3:], axis=0)
        
        X[i, -3:] = (iX[i, -3:] - mean_totvol)/std_totvol
        
        i += 1
    return X

# Load saved data
loc = 'C:\\Users\\enehl\\OneDrive - TUNI.fi\\Opiskelu\\Kandi\\Code'    

daily_stock_data = pd.read_hdf(loc + '\\daily_stock_data.h5')
investor_data = pd.read_hdf(loc + '\\investor_data.h5')
trdates = pd.read_hdf(loc + '\\trading_dates.h5')
intraday_data = pd.read_hdf(loc + '\\intraday_stock_data.h5')

nokia_data = investor_data[investor_data['isin']=='FI0009000681']   

ready_data, investor_ids = prepair_investor_data(nokia_data)


# ## Input formatting
# Input 1: Daily returns from closing price
return_data = copy.deepcopy(intraday_data)
return_data['datetime'] = [x.date() for x in return_data['datetime']]
return_data = return_data.groupby(by=['datetime'], as_index=False).last()

period_returns = np.zeros([len(periods), N])
for k_ind in range(len(periods)):
    k = periods[k_ind]
    new_row = np.log(return_data['price'] / return_data['price'].shift(k))
    period_returns[k_ind] = new_row[-N:]



# Input 2: Intra-day volatility over 5min periods
intra_day_5min_vola = intra_day_vola(copy.deepcopy(intraday_data))
cum_intraday_vola = intra_day_5min_vola.cumsum()

period_intraday_vola = np.zeros([len(periods), N])
for k_ind in range(len(periods)):
    k = periods[k_ind]
    shifted_cum_intraday_vola = np.concatenate([[np.NaN]*(k-1), [0],  
                                                cum_intraday_vola[:-k]])
    new_row = (cum_intraday_vola - shifted_cum_intraday_vola)/k
    period_intraday_vola[k_ind] = new_row[-N:]



# Input 3: Total traded volume of the asset 
total_volume_data = daily_stock_data['sum(volume)']
cum_volume = total_volume_data.cumsum()

period_total_volume = np.zeros([len(periods), N])
for k_ind in range(len(periods)):
    k = periods[k_ind]
    shifted_cum_volum = np.concatenate([[np.NaN]*(k-1), [0],  cum_volume[:-k]])
    new_row = cum_volume - shifted_cum_volum
    
    period_total_volume[k_ind] = new_row[-N:]


output = ready_data[0][0][0][:,-N:]

# Inputs
i1_log_ret = np.asarray(period_returns)
i2_intraday_vola = np.asarray(period_intraday_vola)
i3_volume = np.asarray(period_total_volume)

i4_eurovolume = ready_data[0][0][1][:,-N:]
i5_time_gone = ready_data[0][0][2][:,-N:]
i6_actvity = np.count_nonzero(output,axis=0)[-N:]
#i7_investor = np.asarray(investor_ids)
i9 = copy.deepcopy(output)

iX = copy.deepcopy(i9)
iX = np.vstack([iX, i4_eurovolume, i5_time_gone, i6_actvity,
                i1_log_ret, i2_intraday_vola, i3_volume])

iX = iX.T
iY = output.T

iX = normalize_inputs(iX)[20:,:]

np.save(loc + '\\data.npy', iX)









