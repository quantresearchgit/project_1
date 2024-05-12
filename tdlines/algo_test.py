#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 13:47:05 2017

@author: PastaTrio
"""

import matplotlib.pyplot as plt
import datetime
import pandas as pd
import numpy as np
from collections import defaultdict

# from matplotlib.dates import DateFormatter, WeekdayLocator,\
#    DayLocator, MONDAY
#from matplotlib.finance import candlestick_ohlc

ANNOTATION_OFFSET_UP = 0.01
ANNOTATION_OFFSET_DOWN = 0.03
PLOT_FIGURE = False

#%%

''' DATA '''

EUREX_FUTURES = True
Skip_Bar_at_9PM = False
FX = False
Daily = False

''' CAPITAL '''

CAPITAL = 1000000
RISK = 0.03

''' FILTERS '''

ATR_LIMIT = True
REF_CLOSE = True
DEMARKER1_EXTREME = True
TF_CONFO = True
HILO = True
REV_HILO = True
TP_OUTERBAR = True

### Filter paramenters 

atr_period =  50
rolling_window_lines = 400
HILO_CONFO_STRICT = True
HILO_CONFO_LOOSE = False # pick either strict or loose

demarker1_period = 13
OB_lvl = 0.60
OS_lvl = 0.40

multiplierTP = 1
multiplierSTOP = 1

PATH = r'C:\Users\manga\Desktop\Trading\nico model\tdlines\\'
MAIN_DF_FILE = 'IKZ7_60.csv'
TF_CONFO_FILE ='IKZ7_30.csv'

# MAIN_DF_FILE = 'RXZ7_60M60D.csv'
# TF_CONFO_FILE ='RXZ7_120M60D.csv'

PRINT_LINES = False
PRINT_RESULTS = True

# if Daily:
#     dateparse = lambda x: pd.datetime.strptime(x, '%d/%m/%y')
# else:
#     dateparse = lambda x: pd.datetime.strptime(x, '%d/%m/%y %H:%M')
    
#%%

dates = []
opens = []
highs = []
lows = []
closes = []
indexes = []
idx = 0

with open(MAIN_DF_FILE , 'r') as infile:

    first_line = infile.readline()      # skip headers (Dates, Open...)

    for line in infile:
        if Skip_Bar_at_9PM:
            if '21:00' not in line:
                line = line.split(',')      # split line where there is a comma
                indexes.append(idx)
                if Daily:
                    dates.append(datetime.datetime.strptime(line[0], "%d/%m/%Y"))
                else:
                    dates.append(datetime.datetime.strptime(line[0], "%d/%m/%Y %H:%M"))
                opens.append(float(line[1]))
                highs.append(float(line[2]))
                lows.append(float(line[3]))
                closes.append(float(line[4]))
                idx += 1
        else:
            line = line.split(',')      # split line where there is a comma
            indexes.append(idx)
            if Daily:
                dates.append(datetime.datetime.strptime(line[0], "%d/%m/%Y"))
            else:
                dates.append(datetime.datetime.strptime(line[0], "%d/%m/%y %H:%M"))            
            opens.append(float(line[1]))
            highs.append(float(line[2]))
            lows.append(float(line[3]))
            closes.append(float(line[4]))
            idx += 1

dates = dates[::-1]
opens = opens[::-1]
highs = highs[::-1]
lows = lows[::-1]
closes = closes[::-1]

data = list(zip(indexes, opens, highs, lows, closes)) # create a list of tuple OHLC with idx

df = pd.read_csv(PATH + MAIN_DF_FILE)
if Daily:
    df.Date  = df.Date.apply(lambda x: datetime.datetime.strptime(x, '%d/%m/%y') )
else:
    df.Date = df.Date.apply(lambda x: datetime.datetime.strptime(x, '%d/%m/%y %H:%M'))
df.rename(columns={'TD Reference Close Up': 'Ref_C_Ups', 'TD Reference Close Down': 'Ref_C_Downs'}, inplace=True)
df.Close.plot()
df.index = df.Date
if Daily == False:
    if Skip_Bar_at_9PM == True:
        df = df.between_time(datetime.time(6, 59), datetime.time(20, 59))  # modify depending on data
    else:
        df = df.between_time(datetime.time(6, 59), datetime.time(21, 00))  # modify depending on data
df = df.iloc[::-1]
del df['Date']
df.reset_index(inplace=True)


df['Close_t-1'] = df['Close'].shift(+1) # previous day close
df['Close_t-2'] = df['Close'].shift(+2)
df['High_t-1'] = df['High'].shift(+1)
df['High_t+1'] = df['High'].shift(-1)
df['Low_t-1'] = df['Low'].shift(+1)
df['Low_t+1'] = df['Low'].shift(-1)
df['High_t-2'] = df['High'].shift(+2)
df['High_t+2'] = df['High'].shift(-2)
df['Low_t-2'] = df['Low'].shift(+2)
df['Low_t+2'] = df['Low'].shift(-2)

'''Change here the way TP is calculated'''
df['BUY_TP'] = df['Close'] - df['Low']
df['SELL_TP'] = - df['Close'] + df['High']
df['BUY_STOP'] = df['Low']
df['SELL_STOP'] = df['High']

TOTAL_BARS = df['Date'].size

''' Data for TF Confo '''

if TF_CONFO:
    df_tf = pd.read_csv(PATH + TF_CONFO_FILE)
    if Daily:
        df_tf.Date = df_tf.Date.apply(lambda x: datetime.datetime.strptime(x, '%d/%m/%y'))
    else:
        df_tf.Date = df_tf.Date.apply(lambda x: datetime.datetime.strptime(x, '%d/%m/%y %H:%M'))
    df_tf.rename(columns={'TD Reference Close Up': 'Ref_C_Ups', 'TD Reference Close Down': 'Ref_C_Downs'}, inplace=True)
    df_tf.Close.plot()
    df_tf.index = df_tf.Date
    if Skip_Bar_at_9PM == True:
            df_tf = df_tf.between_time(datetime.time(6, 59), datetime.time(20, 59))  # modify depending on data
    else:
            df_tf = df_tf.between_time(datetime.time(6, 59), datetime.time(21, 00))  # modify depending on data

    df_tf = df_tf.iloc[::-1]
    del df_tf['Date']
    df_tf.reset_index(inplace=True)
    
    
    

    

    df_tf['Time Stamp'] = df_tf['Date'].dt.time
    
    df_tf['Close_t-1'] = df_tf['Close'].shift(+1) # previous day close
    df_tf['Close_t-2'] = df_tf['Close'].shift(+2)
    df_tf['High_t-1'] = df_tf['High'].shift(+1)
    df_tf['High_t+1'] = df_tf['High'].shift(-1)
    df_tf['High_t-2'] = df_tf['High'].shift(+2)
    df_tf['High_t+2'] = df_tf['High'].shift(-2)
    df_tf['Low_t-1'] = df_tf['Low'].shift(+1)
    df_tf['Low_t+1'] = df_tf['Low'].shift(-1)
    df_tf['Low_t-2'] = df_tf['Low'].shift(+2)
    df_tf['Low_t+2'] = df_tf['Low'].shift(-2)


    
''' FIND TD POINTS '''
# td point = high higher than previous high and following high
# TODO high higher than close 2 bars before
td_highs = []
td_lows = []

for idx in indexes[1:-1]:  # skip first and last datapoints
    if (highs[idx] > highs[idx+1] and highs[idx] > highs[idx-1]) or \
    (highs[idx] > highs[idx+1] and highs[idx] == highs[idx-1] and highs[idx-1] > highs[idx-2]): # if two consecutive tdpoints, take the 2nd
        td_highs.append((idx, dates[idx], highs[idx]))

for idx in indexes[1:-1]:
    if (lows[idx] < lows[idx+1] and lows[idx] < lows[idx-1]) or \
    (lows[idx] < lows[idx+1] and lows[idx] == lows[idx-1] and lows[idx-1] < lows[idx-2]):   
        td_lows.append((idx, dates[idx], lows[idx]))




#%%

''' TRUE RANGE and AVERAGE TRUE RANGE - TR/ATR 

    TR = max[(High - Low),abs(high-previous close),abs(low-previous close)]
    ATR = ((ATR at t-1)*(atr_period-1) + TR at t)/ atr_period or use EMA '''
    
def add_tr_to_df (df,atr_period):
    df['ATR1'] = df.apply(lambda row : abs(row['High'] - row['Low']),axis=1)
    df['ATR2'] = df.apply(lambda row : abs(row['High'] - row['Close_t-1']),axis=1)
    df['ATR3'] = df.apply(lambda row : abs(row['Low'] - row['Close_t-1']),axis=1)
    df['TR'] = df[['ATR1','ATR2','ATR3']].max(axis=1)
    df['ATR'] = df['TR'].ewm(span = atr_period).mean()
    df.drop(['ATR1','ATR2','ATR3','TR'], axis=1,inplace=True)

add_tr_to_df(df,atr_period)    

#%%

''' FIND TD LINES '''
td_lines = []

for tidx_1 in range(len(td_highs)-1):
    for tidx_2 in range(tidx_1, len(td_highs)):
        if td_highs[tidx_2][0] - td_highs[tidx_1][0] < rolling_window_lines:
            if td_highs[tidx_1][2] > td_highs[tidx_2][2]:
                m = (td_highs[tidx_2][2] - td_highs[tidx_1][2]) / (td_highs[tidx_2][0] - td_highs[tidx_1][0])
                b = td_highs[tidx_1][2]
                line_breached = False
                for idx in range(td_highs[tidx_1][0]+1,td_highs[tidx_2][0]):  # no close higher then line btw the two TDP
                        line_price_y_breach = (m * (idx-td_highs[tidx_1][0]) + b)
                        if closes[idx] > line_price_y_breach: 
                            line_breached = True
                            break
                if not line_breached:
                    td_lines.append(((td_highs[tidx_1][1],td_highs[tidx_1][0], td_highs[tidx_1][2]), (td_highs[tidx_2][1],td_highs[tidx_2][0], td_highs[tidx_2][2]), m, b))
                    #print(((td_highs[tidx_1][0], td_highs[tidx_1][2]), (td_highs[tidx_2][0], td_highs[tidx_2][2]), m, b))

# find low lines from td lows
for tidx_1 in range(len(td_lows)-1):
    for tidx_2 in range(tidx_1, len(td_lows)):
        if td_lows[tidx_2][0] - td_lows[tidx_1][0] < rolling_window_lines:
            if td_lows[tidx_1][2] < td_lows[tidx_2][2]:
                m = (td_lows[tidx_2][2] - td_lows[tidx_1][2]) / (td_lows[tidx_2][0] - td_lows[tidx_1][0])
                b = td_lows[tidx_1][2]
                line_breached = False
                for idx in range(td_lows[tidx_1][0]+1,td_lows[tidx_2][0]):
                        line_price_y_breach = (m * (idx-td_lows[tidx_1][0]) + b)
                        if closes[idx] < line_price_y_breach: 
                            line_breached = True
                            break
                if not line_breached:
                    td_lines.append(((td_lows[tidx_1][1],td_lows[tidx_1][0], td_lows[tidx_1][2]), (td_lows[tidx_2][1],td_lows[tidx_2][0], td_lows[tidx_2][2]), m, b))
                    #print(((td_lows[tidx_1][0], td_lows[tidx_1][2]), (td_lows[tidx_2][0], td_lows[tidx_2][2]), m, b))

for tdl in td_lines:
    if Daily:
        data_first_TDP = tdl[0][0].strftime("%d-%m-%Y") # convert date in dd/mm/yyyy hh:mm for first TDP for the line
        data_second_TDP = tdl[1][0].strftime("%d-%m-%Y")
    else:
        data_first_TDP = tdl[0][0].strftime("%d-%m-%Y %H:%M") # convert date in dd/mm/yyyy hh:mm for first TDP for the line
        data_second_TDP = tdl[1][0].strftime("%d-%m-%Y %H:%M")
    price_first_TDP = tdl[0][2]
    price_second_TDP = tdl[1][2]
    slope_line = '{:.2f}'.format(tdl[2])
#    print(data_first_TDP + " @ " + str(price_first_TDP) + " / " + data_second_TDP + " @ " + str(price_second_TDP) + " Slope: " + slope_line)
    
#%%

''' REVERSE BREAK 
    buy: close[b] > close[b-1] < close[b-2]) (for sell: close[b] < close[b-1] > close[b-2]) '''

revs_buy = []
revs_sell = []

for idx in indexes[2:]:  # skip first and last datapoints
    if closes[idx] > closes[idx-1] and closes[idx-1] < closes[idx-2]:
        revs_buy.append((idx, dates[idx], closes[idx]))
    if closes[idx] < closes[idx-1] and closes[idx-1] > closes[idx-2]:
        revs_sell.append((idx, dates[idx], closes[idx]))
        
#''' PANDAS RB '''
#def Revs_Buy(df):
#    df['Revs_Buy'] = df.apply(lambda row : row['Close'] if row['Close'] > row['Close_t-1'] and row['Close_t-1'] < row['Close_t-2'] else np.NaN,axis=1)    
#    
#def Revs_Sell(df):
#    df['Revs_Sell'] = df.apply(lambda row : row['Close'] if row['Close'] < row['Close_t-1'] and row['Close_t-1'] > row['Close_t-2'] else np.NaN,axis=1)
#    
#Revs_Buy
#Revs_Sell
        
#%%
''' REV BREAK AND TD LINES '''

buy_signals = []
sell_signals = []

buy_signals_date_px_dict = defaultdict(list)
sell_signals_date_px_dict = defaultdict(list)

''' BUY '''
for rev_buy in revs_buy:
    for td_line in td_lines:  
        if rev_buy[0] > td_line[1][1]:  # consider only rev breaks after the second td_point of the line
            if td_line[2] < 0:  #check against negative slope TD lines (m coefficient td_line[2])
               line_breached = False
               for idx in range(td_line[1][1],rev_buy[0]):  # check if no close is higher then line btw rev break and 2nd tdpoint
                    line_price_y_breach = td_line[2] * (idx-td_line[0][1]) + td_line[3]
                    if closes[idx]>=line_price_y_breach:
                        line_breached = True
                        break
               if not line_breached:
                        #find value of TD line at rev_buy x-axis position 
                    line_price_y_rev_break = td_line[2] * (rev_buy[0]-td_line[0][1]) + td_line[3]
                    if rev_buy[2] > line_price_y_rev_break:
                        buy_signals.append((rev_buy,td_line))
                        buy_signals_date_px_dict['Date'].append(rev_buy[1])
                        buy_signals_date_px_dict['BUY'].append(rev_buy[2])
                        
df_Buy_Signals = pd.DataFrame(buy_signals_date_px_dict)
df_Buy_Signals.drop_duplicates(subset=['Date'],keep='first',inplace=True)

''' SELL '''                        
for rev_sell in revs_sell:
    for td_line in td_lines:  
        if rev_sell[0] > td_line[1][1]:  # consider only rev breaks after the second td_point of the line
            if td_line[2] > 0:  #check against negative slope TD lines (m coefficient td_line[2])
               line_breached = False
               for idx in range(td_line[1][1],rev_sell[0]):  # check if no close is higher then line btw rev break and 2nd tdpoint
                    line_price_y_breach = td_line[2] * (idx-td_line[0][1]) + td_line[3]
                    if closes[idx]<=line_price_y_breach:
                        line_breached = True
                        break
               if not line_breached:
                        #find value of TD line at rev_buy x-axis position 
                    line_price_y_rev_break = td_line[2] * (rev_sell[0]-td_line[0][1]) + td_line[3]
                    if rev_sell[2] < line_price_y_rev_break:
                        sell_signals.append((rev_sell,td_line))
                        sell_signals_date_px_dict['Date'].append(rev_sell[1])
                        sell_signals_date_px_dict['SELL'].append(rev_sell[2])
 
df_Sell_Signals = pd.DataFrame(sell_signals_date_px_dict)
df_Sell_Signals.drop_duplicates(subset=['Date'],keep='first',inplace=True)

df = df.merge(df_Buy_Signals,how='left',on='Date')
df = df.merge(df_Sell_Signals,how='left',on='Date')

df['lines_b'] = df.apply(lambda row : np.NaN if np.isnan(row['BUY']) else row['BUY'], axis=1)
df['lines_s'] = df.apply(lambda row : np.NaN if np.isnan(row['SELL']) else row['SELL'], axis=1)

#%%

df['TD_High_RC'] = df.apply(lambda row : row['High'] if row['High'] > row['High_t-1'] and row['High'] > row['High_t+1'] else \
                   row['High'] if (row['High'] > row['High_t+1'] and row['High'] == row['High_t-1'] and row['High_t-1'] > row['High_t-2']) else \
                   row['High'] if (row['High'] == row['High_t+1'] and row['High'] > row['High_t-1']) \
                   else np.NaN, axis=1)

df['TD_Low_RC'] = df.apply(lambda row: row['Low'] if row['Low'] <= row['Low_t-1'] and row['Low'] <= row['Low_t+1'] else np.NaN, axis=1)

df['TD_Point'] = df.apply(lambda row : 1 if not np.isnan(row['TD_High_RC']) else 1 if not np.isnan(row['TD_Low_RC']) else 0, axis=1 )

def ref_close_test(df):
    for row in range(5,len(df)):
        if df.loc[row,'TD_Point'] == 1:
            df.loc[row,'RCU_4back'] = max(df.loc[row-1,'Close'],df.loc[row-2,'Close'],df.loc[row-3,'Close'],df.loc[row-4,'Close'])
            for idx in range(1,5):
                if df.loc[row+idx,'Close'] <= df.loc[row,'RCU_4back']:
                    df.loc[row+idx,'RCU_4fwd'] = df.loc[row,'RCU_4back']
                if df.loc[row+idx,'Close'] > df.loc[row,'RCU_4back']:
                    break
#ref_close_test(df)
    
# 1 if there is a td high or td low



#%%
''' DEMARKER '''
''' Numerator: if High >= High t-1 then DeMax = High - High t-1 else 0
               if Low <= Low t-1 then DeMin = - Low + Low t-1 else 0
    DeMarker1 = RollingSum(DeMax)/RollingSum(DeMax) + RollingSum(DeMin)
    DeMarker period 14 '''
  
def DeMarker_1 (df, demarker1_period, OB_lvl, OS_lvl):
    df['DeMax'] = df.apply(lambda row : (row['High'] - row['High_t-1']) if (row['High'] - row['High_t-1']) >= 0 else 0, axis=1)
    df['DeMin'] = df.apply(lambda row : (- row['Low'] + row['Low_t-1']) if (- row['Low'] + row['Low_t-1']) >= 0 else 0, axis=1)
    df['DeMarker1'] = round(df['DeMax'].rolling(demarker1_period).sum() / (df['DeMax'].rolling(demarker1_period).sum() + df['DeMin'].rolling(demarker1_period).sum() ),2)
    df.drop(['DeMax','DeMin'], axis=1,inplace=True)
    
    df['OB'] = df.apply(lambda row : True if row['DeMarker1'] > OB_lvl else False, axis=1)
    df['OS'] = df.apply(lambda row : True if row['DeMarker1'] < OS_lvl else False, axis=1)

DeMarker_1 (df, demarker1_period, OB_lvl, OS_lvl)   

    
def countConsecutiveEntriesOB (df,demarker1_period):
    res = [0]*len(df)
    count = 0
    for i,e in enumerate(df['OB']):
        if e == True:
            count+=1
            if count >= demarker1_period:
                res[i] = True
        else:
            count = 0
    return res

df['EOB'] = countConsecutiveEntriesOB (df, demarker1_period)

def countConsecutiveEntriesOS (df,demarker1_period):
    res = [0]*len(df)
    count = 0
    for i,e in enumerate(df['OS']):
        if e == True:
            count +=1
            if count >= demarker1_period:
                res[i] = True
        else:
            count = 0
    return res

df['EOS'] = countConsecutiveEntriesOS (df, demarker1_period)
   
#%%
    
''' FILTER SWITCHES '''
if ATR_LIMIT:
    df['BUY'] = df.apply(lambda row : row['BUY'] if (row['Close'] - row['Low']) > row['ATR'] else np.NaN, axis=1)
    df['SELL'] = df.apply(lambda row : row['SELL'] if ( - row['Close'] + row['High']) > row['ATR'] else np.NaN, axis=1)

if REF_CLOSE:
    df['BUY'] = df.apply(lambda row : row['BUY'] if row['BUY'] > row['Ref_C_Ups'] else np.NaN, axis=1)
    df['SELL'] = df.apply(lambda row : row['SELL'] if row['SELL'] < row['Ref_C_Downs'] else np.NaN, axis=1)
   
if DEMARKER1_EXTREME:
    df['BUY'] = df.apply(lambda row : row['BUY'] if row['EOS'] == False else np.NaN, axis=1)
    df['SELL'] = df.apply(lambda row : row['SELL'] if row['EOB'] == False else np.NaN, axis=1)

if REV_HILO:
    df['BUY'] = df.apply(lambda row : row['BUY'] if row['High_t-1'] < row['High'] else np.NaN, axis=1)
    df['SELL'] = df.apply(lambda row : row['SELL'] if row['Low_t-1'] > row['Low'] else np.NaN, axis=1)

if TP_OUTERBAR:
    df['BUY'] = df.apply(lambda row: row['BUY'] if (row['BUY_TP'] + row['BUY']) > row['High'] else np.NaN, axis=1)
    df['SELL'] = df.apply(lambda row: row['SELL'] if (-row['SELL_TP'] + row['SELL']) < row['Low'] else np.NaN, axis=1)

if TF_CONFO:
    df_tf['RC_up_TF'] = df_tf.apply(lambda row : True if row['Close'] > row['Ref_C_Ups'] else False, axis=1)
    df_tf['RC_down_TF'] = df_tf.apply(lambda row : True if row['Close'] < row['Ref_C_Downs'] else False, axis=1)
    df_tf['RB_TF_Buy'] = df_tf.apply(lambda row : True if row['Close'] > row['Close_t-1'] and row['Close_t-1'] < row['Close_t-2'] else False, axis=1)
    df_tf['RB_TF_Sell'] = df_tf.apply(lambda row : True if row['Close'] < row['Close_t-1'] and row['Close_t-1'] > row['Close_t-2'] else False, axis=1)
  
    if Skip_Bar_at_9PM == False:
        df_tf.set_index('Date',inplace=True)
        df_tf = df_tf.between_time(datetime.time(6,59),datetime.time(20,59))        
        df_tf.reset_index(inplace=True)    
    
    df_tf['TF_Confo_Buy'] = df_tf.apply(lambda row : True if row['RC_up_TF'] == True and row['RB_TF_Buy'] == True else False, axis=1)
    df_tf['TF_Confo_Sell'] = df_tf.apply(lambda row : True if row['RC_down_TF'] == True and row['RB_TF_Sell'] == True else False, axis=1)

    
    if 'IK' or 'OAT' in MAIN_DF_FILE:
        if '60' in MAIN_DF_FILE:
            for row in range(0,len(df_tf),2):
                df_tf.loc[row,'TF_BUY'] = df_tf.loc[row:row+1,'TF_Confo_Buy'].any()
                df_tf.loc[row,'TF_SELL'] = df_tf.loc[row:row+1,'TF_Confo_Sell'].any()
        if '120' in MAIN_DF_FILE:
            for row in range(0,len(df_tf)):
                if '07:00' in str(df_tf['Time Stamp'][row]):
                    df_tf.loc[row,'TF_BUY'] = df_tf.loc[row:row+1,'TF_Confo_Buy'].any()
                    df_tf.loc[row,'TF_SELL'] = df_tf.loc[row:row+1,'TF_Confo_Sell'].any()
                elif '09:00' in str(df_tf['Time Stamp'][row]):
                    df_tf.loc[row,'TF_BUY'] = df_tf.loc[row:row+1,'TF_Confo_Buy'].any()
                    df_tf.loc[row,'TF_SELL'] = df_tf.loc[row:row+1,'TF_Confo_Sell'].any()
                elif '11:00' in str(df_tf['Time Stamp'][row]):
                    df_tf.loc[row,'TF_BUY'] = df_tf.loc[row:row+1,'TF_Confo_Buy'].any()
                    df_tf.loc[row,'TF_SELL'] = df_tf.loc[row:row+1,'TF_Confo_Sell'].any()
                elif '13:00' in str(df_tf['Time Stamp'][row]):
                    df_tf.loc[row,'TF_BUY'] = df_tf.loc[row:row+1,'TF_Confo_Buy'].any()
                    df_tf.loc[row,'TF_SELL'] = df_tf.loc[row:row+1,'TF_Confo_Sell'].any()
                elif '15:00' in str(df_tf['Time Stamp'][row]):
                    df_tf.loc[row,'TF_BUY'] = df_tf.loc[row:row+1,'TF_Confo_Buy'].any()
                    df_tf.loc[row,'TF_SELL'] = df_tf.loc[row:row+1,'TF_Confo_Sell'].any()
                elif '17:00' in str(df_tf['Time Stamp'][row]):
                    df_tf.loc[row,'TF_BUY'] = df_tf.loc[row,'TF_Confo_Buy']
                    df_tf.loc[row,'TF_SELL'] = df_tf.loc[row,'TF_Confo_Sell']
                else:
                    np.NaN
        if '240' in MAIN_DF_FILE:
            for row in range(0,len(df_tf)):
                if '07:00' in str(df_tf['Time Stamp'][row]):
                    df_tf.loc[row,'TF_BUY'] = df_tf.loc[row:row+1,'TF_Confo_Buy'].any()
                    df_tf.loc[row,'TF_SELL'] = df_tf.loc[row:row+1,'TF_Confo_Sell'].any()
                elif '11:00' in str(df_tf['Time Stamp'][row]):
                    df_tf.loc[row,'TF_BUY'] = df_tf.loc[row:row+1,'TF_Confo_Buy'].any()
                    df_tf.loc[row,'TF_SELL'] = df_tf.loc[row:row+1,'TF_Confo_Sell'].any()        
                elif '15:00' in str(df_tf['Time Stamp'][row]):
                    df_tf.loc[row,'TF_BUY'] = df_tf.loc[row:row+1,'TF_Confo_Buy'].any()
                    df_tf.loc[row,'TF_SELL'] = df_tf.loc[row:row+1,'TF_Confo_Sell'].any()
                else:
                    np.NaN
    
        df = df.merge(df_tf[['Date','TF_BUY','TF_SELL']], on='Date', how='left')
        
        df['BUY'] = df.apply(lambda row : row['BUY'] if row['TF_BUY'] == True else np.NaN, axis=1)
        df['SELL'] = df.apply(lambda row : row['SELL'] if row['TF_SELL'] == True else np.NaN, axis=1)

#%%

''' SIZING '''
dollar_risk = RISK * CAPITAL
df['BUY_t-1'] = df['BUY'].shift(+1)
df['SELL_t-1'] = df['SELL'].shift(+1)

df['L_POS'] = df.apply(lambda row : int((dollar_risk/1000) / (row['Close'] - row['Low'])) if not np.isnan(row['BUY']) and row['Close'] != row['Low'] else np.NaN, axis=1)
df['L_TP'] = df.apply(lambda row : (row['BUY_TP'] + row['BUY']) if not np.isnan(row['BUY']) else np.NaN, axis=1)
df['L_STOP'] = df.apply(lambda row : row['BUY_STOP'] if not np.isnan(row['BUY']) else np.NaN, axis=1)

df['S_POS'] = df.apply(lambda row : int((dollar_risk/1000) / (- row['Close'] + row['High'])) if not np.isnan(row['SELL']) and row['Close'] != row['High'] else np.NaN, axis=1)
df['S_TP'] = df.apply(lambda row : (- row['SELL_TP'] + row['SELL']) if not np.isnan(row['SELL']) else np.NaN, axis=1)
df['S_STOP'] = df.apply(lambda row : row['SELL_STOP'] if not np.isnan(row['SELL']) else np.NaN, axis=1)

df['L_POS_t-1'] = df['L_POS'].shift(+1)
df['L_STOP_t-1'] = df['L_STOP'].shift(+1)
df['L_TP_t-1'] = df['L_TP'].shift(+1)
df['S_POS_t-1'] = df['S_POS'].shift(+1)
df['S_STOP_t-1'] = df['S_STOP'].shift(+1)
df['S_TP_t-1'] = df['S_TP'].shift(+1)
    
#%%

''' HILO CONFO '''

if HILO_CONFO_STRICT:
    df['B_HILO'] = df.apply(lambda row : True if not np.isnan(row['BUY_t-1']) and row['High'] > row['High_t-1'] else False, axis=1)
    df['S_HILO'] = df.apply(lambda row : True if not np.isnan(row['SELL_t-1']) and row['Low'] < row['Low_t-1'] else False, axis=1)

if HILO_CONFO_LOOSE:
    df['B_HILO'] = df.apply(lambda row : True if not np.isnan(row['BUY_t-1']) and row['High'] >= row['High_t-1'] else False, axis=1)
    df['S_HILO'] = df.apply(lambda row : True if not np.isnan(row['SELL_t-1']) and row['Low'] <= row['Low_t-1'] else False, axis=1)    
    
#%%
''' TRADING '''

df['TRADE'] = df.apply(lambda row: 'BUY' if not np.isnan(row['BUY']) else ('SELL' if not np.isnan(row['SELL']) else np.NAN), axis=1)
df['TRADE_t-1'] = df['TRADE'].shift(+1)

def trading_long(df):
    df['Live_P&L_Longs'] = 0.0
    for row in range(len(df)-1):
        if df.loc[row,'L_POS_t-1'] > 0:
            df.loc[row,'L_POS'] = df.loc[row,'L_POS_t-1']
            df.loc[row,'L_STOP'] = df.loc[row,'L_STOP_t-1'] 
            df.loc[row,'L_TP'] =df.loc[row,'L_TP_t-1']
            if df.loc[row,'Low'] <= df.loc[row,'L_STOP']:
                df.loc[row,'L_POS'] = np.NaN
                df.loc[row,'EXIT'] = 'STOP'
                df.loc[row,'Live_P&L_Longs'] = (df.loc[row,'L_STOP'] - df.loc[row,'Close_t-1']) * df.loc[row,'L_POS_t-1'] * 1000
            elif df.loc[row,'Low'] > df.loc[row,'L_STOP']:
                if df.loc[row,'High'] >= df.loc[row,'L_TP']:
                    df.loc[row,'L_POS'] = np.NaN
                    df.loc[row,'EXIT'] = 'TP'
                    df.loc[row,'Live_P&L_Longs'] = (df.loc[row,'L_TP'] - df.loc[row,'Close_t-1']) * df.loc[row,'L_POS_t-1'] * 1000
                elif df.loc[row,'Close'] < df.loc[row,'L_TP']:
                    if HILO:
                        if df.loc[row,'B_HILO'] == True:
                            df.loc[row+1,'L_POS_t-1'] = df.loc[row,'L_POS_t-1']
                            df.loc[row+1,'L_STOP_t-1'] = df.loc[row,'L_STOP_t-1'] 
                            df.loc[row+1,'L_TP_t-1'] = df.loc[row,'L_TP_t-1']
                            df.loc[row+1,'B_HILO'] = df.loc[row,'B_HILO']
                            df.loc[row,'Live_P&L_Longs'] = (df.loc[row,'Close'] - df.loc[row,'Close_t-1']) * df.loc[row,'L_POS_t-1'] * 1000
                        elif df.loc[row,'B_HILO'] == False:
                            df.loc[row,'L_POS'] = np.NaN
                            df.loc[row,'EXIT'] = 'NO HILO'
                            df.loc[row,'Live_P&L_Longs'] = (df.loc[row,'Close'] - df.loc[row,'Close_t-1']) * df.loc[row,'L_POS_t-1'] * 1000
                    else:    
                        df.loc[row+1,'L_POS_t-1'] = df.loc[row,'L_POS_t-1']
                        df.loc[row+1,'L_STOP_t-1'] = df.loc[row,'L_STOP_t-1'] 
                        df.loc[row+1,'L_TP_t-1'] = df.loc[row,'L_TP_t-1']
                        df.loc[row,'Live_P&L_Longs'] = (df.loc[row,'Close'] - df.loc[row,'Close_t-1']) * df.loc[row,'L_POS_t-1'] * 1000
    df['Tot_P&L_Longs'] = df['Live_P&L_Longs'].cumsum()
    
def trading_short(df):
    df['Live_P&S_Shorts'] = 0.0
    for row in range(len(df)-1):
        if df.loc[row,'S_POS_t-1'] > 0:
            df.loc[row,'S_POS'] = df.loc[row,'S_POS_t-1']
            df.loc[row,'S_STOP'] = df.loc[row,'S_STOP_t-1'] 
            df.loc[row,'S_TP'] = df.loc[row,'S_TP_t-1']
            if df.loc[row,'High'] >= df.loc[row,'S_STOP']:
                df.loc[row,'S_POS'] = np.NaN
                df.loc[row,'EXIT'] = 'STOP'
                df.loc[row,'Live_P&S_Shorts'] = (- df.loc[row,'S_STOP'] + df.loc[row,'Close_t-1']) * df.loc[row,'S_POS_t-1'] * 1000
            elif df.loc[row,'High'] < df.loc[row,'S_STOP']:
                if df.loc[row,'Low'] <= df.loc[row,'S_TP']:
                    df.loc[row,'S_POS'] = np.NaN
                    df.loc[row,'EXIT'] = 'TP'
                    df.loc[row,'Live_P&S_Shorts'] = ( - df.loc[row,'S_TP'] + df.loc[row,'Close_t-1']) * df.loc[row,'S_POS_t-1'] * 1000
                elif df.loc[row,'Close'] > df.loc[row,'S_TP']:
                    if HILO:
                        if df.loc[row,'S_HILO'] == True:
                                df.loc[row+1,'S_POS_t-1'] = df.loc[row,'S_POS_t-1']
                                df.loc[row+1,'S_STOP_t-1'] = df.loc[row,'S_STOP_t-1'] 
                                df.loc[row+1,'S_TP_t-1'] = df.loc[row,'S_TP_t-1']
                                df.loc[row+1,'S_HILO'] = df.loc[row,'S_HILO']
                                df.loc[row,'Live_P&S_Shorts'] = ( - df.loc[row,'Close'] + df.loc[row,'Close_t-1']) * df.loc[row,'S_POS_t-1'] * 1000
                        elif df.loc[row,'S_HILO'] == False:
                                df.loc[row,'S_POS'] = np.NaN
                                df.loc[row,'EXIT'] = 'NO HILO'
                                df.loc[row,'Live_P&S_Shorts'] = ( - df.loc[row,'Close'] + df.loc[row,'Close_t-1']) * df.loc[row,'S_POS_t-1'] * 1000
                    else:
                        df.loc[row+1,'S_POS_t-1'] = df.loc[row,'S_POS_t-1']
                        df.loc[row+1,'S_STOP_t-1'] = df.loc[row,'S_STOP_t-1'] 
                        df.loc[row+1,'S_TP_t-1'] = df.loc[row,'S_TP_t-1']
                        df.loc[row,'Live_P&S_Shorts'] = ( - df.loc[row,'Close'] + df.loc[row,'Close_t-1']) * df.loc[row,'S_POS_t-1'] * 1000
    df['Tot_P&L_Shorts'] = df['Live_P&S_Shorts'].cumsum()
       
trading_long(df)
trading_short(df)

df['TRADEx1'] = df.apply(lambda row : 'BUY' if row['TRADE'] == 'BUY' and np.isnan(row['L_POS_t-1']) else ('SELL' if row['TRADE'] == 'SELL' and np.isnan(row['S_POS_t-1']) else np.NaN), axis=1)

consecutive_trades =  df.apply(lambda row : 1 if row['TRADE'] in ['BUY','SELL'] and row['EXIT'] in ['TP','STOP','NO HILO'] else 0, axis=1)

df['Live_P&L'] = df.apply(lambda row: row['Live_P&L_Longs'] + row['Live_P&S_Shorts'], axis=1)
df['Tot_P&L'] = df.apply(lambda row : row['Tot_P&L_Longs'] + row['Tot_P&L_Shorts'], axis=1)
ROE = round(df['Tot_P&L'].iloc[-1]/CAPITAL*100,2)

#%%
''' MAX DD '''
max2date = pd.Series.expanding(df['Tot_P&L'],min_periods=1).max()
dd2date = df['Tot_P&L'] - max2date
Max_DD = dd2date.min()
end_date_dd = dd2date.argmin()
start_date_dd = df['Tot_P&L'].loc[:end_date_dd].argmax()

''' SHARPE RATIO '''
returns = df.apply(lambda row : row['Live_P&L'] / CAPITAL, axis =1)
def annualized_sharpe(df, TOTAL_BARS):
    return np.sqrt(TOTAL_BARS) * returns.mean()/returns.std()

SHARPE_RATIO = round(annualized_sharpe(df,TOTAL_BARS),2)

#%%

''' Binary Filters '''

df['f_atr_b'] = df.apply(lambda row : 1 if (row['Close'] - row['Low']) > row['ATR'] else 0, axis=1)
df['f_atr_s'] = df.apply(lambda row : 1 if (-row['Close'] + row['High']) > row['ATR'] else 0, axis=1)
df['f_refc_b'] = df.apply(lambda row : 1 if row['lines_b'] > row['Ref_C_Ups']  else 0, axis=1)
df['f_refc_s'] = df.apply(lambda row : 1 if row['lines_s'] < row['Ref_C_Downs']  else 0, axis=1)
df['f_revhilo_b'] =  df.apply(lambda row: 1 if row['High_t-1'] < row['High'] else 0, axis=1)
df['f_revhilo_s'] =  df.apply(lambda row: 1 if row['Low_t-1'] > row['Low'] else 0, axis=1)
df['f_eos_b'] =  df.apply(lambda row: 1 if row['EOS'] == False else 0, axis=1)
df['f_eos_s'] =  df.apply(lambda row: 1 if row['EOB'] == False else 0, axis=1)
df['f_tfc_b'] = df.apply(lambda row: 1 if row['TF_BUY'] == True else 0, axis=1)
df['f_tfc_s'] = df.apply(lambda row: 1 if row['TF_SELL'] == True else 0, axis=1)
df['f_tpout_b'] = df.apply(lambda row: 1 if (row['BUY_TP'] + row['BUY']) > row['High'] else 0, axis=1)
df['f_tpout_s'] = df.apply(lambda row: 1 if (-row['SELL_TP'] + row['SELL']) < row['Low'] else 0, axis=1)

#%%
''' TEXT FORMATTING FOR BUY AND SELLS WITH LINES NO FILTERS'''
           
if PRINT_LINES:
    for bs in buy_signals:
        date_bs = bs[0][1].strftime("%d-%m-%Y %H:%M")
        price_bs = bs[0][2]
        line = bs[1]
        data_first_TDP = line[0][0].strftime("%d-%m-%Y %H:%M") # convert date in dd/mm/yyyy hh:mm for first TDP for the line
        data_second_TDP = line[1][0].strftime("%d-%m-%Y %H:%M")
        price_first_TDP = line[0][2]
        price_second_TDP = line[1][2]
        slope_line = '{:.2f}'.format(tdl[2])
        print(date_bs + " BUY @ " + str(price_bs) + ' - line: ' + data_first_TDP + " @ " + (str(price_first_TDP) + " / " + data_second_TDP + " @ " + str(price_second_TDP)))
        
    for ss in sell_signals:
        date_ss = ss[0][1].strftime("%d-%m-%Y %H:%M")
        price_ss = ss[0][2]
        line = ss[1]
        data_first_TDP = line[0][0].strftime("%d-%m-%Y %H:%M") # convert date in dd/mm/yyyy hh:mm for first TDP for the line
        data_second_TDP = line[1][0].strftime("%d-%m-%Y %H:%M")
        price_first_TDP = line[0][2]
        price_second_TDP = line[1][2]
        slope_line = '{:.2f}'.format(tdl[2])
        print(date_ss + " SELL @ " + str(price_ss) + ' - line: ' + data_first_TDP + " @ " + (str(price_first_TDP) + " / " + data_second_TDP + " @ " + str(price_second_TDP)))

def print_active_filters(ATR_LIMIT,REF_CLOSE,DEMARKER1_EXTREME,TF_CONFO,HILO):
    print('FILTERS:')
    if ATR_LIMIT == True:
        print('ATR Limit.......: ON')
    if ATR_LIMIT == False:
        print('ATR Limit.......: OFF')
    if REF_CLOSE == True:
        print('Ref Close.......: ON')
    if REF_CLOSE == False:
        print('Ref Close.......: OFF')       
    if DEMARKER1_EXTREME == True:
        print('DeMrk Ext.......: ON')
    if DEMARKER1_EXTREME == False:
        print('DeMrk Ext.......: OFF')    
    if TF_CONFO == True:
        print('TF Confo........: ON')
    if TF_CONFO == False:
        print('TF Confo........: OFF')    
    if HILO == True:
        print('HILO............: ON')
    if HILO == False:
        print('HILO............: OFF')
    if REV_HILO == True:
        print('REV HILO........: ON')
    if REV_HILO == False:
        print('REV HILO........: OFF') 
    if TP_OUTERBAR == True:
        print('TP OUTSIDE BAR..: ON')
    if TP_OUTERBAR == False:
        print('TP OUTSIDE BAR..: OFF') 

        
TOTAL_PL = round(df['Tot_P&L'].iloc[-1],2)       

BUYS = (df['TRADEx1'] == 'BUY').sum()
SELLS = (df['TRADEx1'] == 'SELL').sum()
TP = (df['EXIT'] == 'TP').sum()
STOP = (df['EXIT'] == 'STOP').sum()
NO_HILO = (df['EXIT'] == 'NO_HILO').sum()
TRADES = TP + STOP + NO_HILO
PL_over_MaxDD = round(df['Tot_P&L'].iloc[-1]/abs(Max_DD),2)
BARS_L_POS = df['L_POS'].count()
BARS_S_POS = df['S_POS'].count()
BARS_in_POS = BARS_L_POS + BARS_S_POS
IN_POS_OVER_FLAT = int((BARS_in_POS/TOTAL_BARS)*100)


if PRINT_RESULTS:
    
    print('')
    print('*** ' + MAIN_DF_FILE[:8] + ' ***')
    print('START...........: ' + str(df.loc[0, 'Date']))
    print('END.............: ' + str(df.loc[len(df)-1,'Date']))
    print('Total Bars......: ' + str(TOTAL_BARS))
    print('')
    print_active_filters(ATR_LIMIT,REF_CLOSE,DEMARKER1_EXTREME,TF_CONFO,HILO)
    print('')
    print('Total Trades....: ' + str(TRADES))
    print('BUY.............: ' + str(BUYS))
    print('SELL............: ' + str(SELLS))
    print('TP..............: ' + str(TP))
    print('STOP............: ' + str(STOP))
    if HILO:
        print('NO HILO.........: ' + str(NO_HILO))
    else:
        print('NO HILO.........: HILO is off' )
    print('% df in Pos.....: ' + str(IN_POS_OVER_FLAT))
    print('TOTAL P&L.......: ' + str(TOTAL_PL) + ' EUR')
    print('ROE.............: ' + str(ROE) + ' %')
    print('Ann Sharpe......: ' + str(SHARPE_RATIO))
    print('Max DD..........: ' + str(Max_DD) + ' Eur within ' + str(end_date_dd - start_date_dd +1) + ' bars')
    print('P&L over MaxDD..: ' + str(PL_over_MaxDD))
    print('Lowest P&L......: ' + str(df['Tot_P&L'].min()) + ' Eur')
    print('Highest P&L.....: ' + str(df['Tot_P&L'].max()) + ' Eur')

    if consecutive_trades.sum() != 0:
        print('*** CHECK: Consecutive Trades:........: ' + str(consecutive_trades.sum()))
    if (df.TRADE.count() - df.TRADEx1.count()) == 0:
        print('No 2nd singals when in pos')
    else:
        print('*** CHECK: 2nd singals when in pos...: ' + str(df.TRADE.count() - df.TRADEx1.count() - consecutive_trades.sum()))
        
    # df['Tot_P&L'].plot()
    # plt.show()
    plt.plot(df['Tot_P&L'])

#%%

'''CLEANING LADY'''

cols = ['Date','High','Low','Close', \
       'TD_High_RC','TD_Low_RC', 'Ref_C_Ups','RCU_4back','RCU_4fwd']
#        'lines_b','lines_s', \
#        'f_atr_b', 'f_atr_s', 'ATR','f_refc_b', 'f_refc_s', 'f_revhilo_b', 'f_revhilo_s', 'f_eos_b', 'f_eos_s', 'f_tfc_b', 'f_tfc_s', 'f_tpout_b', 'f_tpout_s', \
#        'TRADE','EXIT','Live_P&L','Tot_P&L']
df = df[cols]


# 'f_atr_b', 'f_atr_s', 'f_refc_b', 'f_refc_s', 'f_revhilo_b', 'f_revhilo_s', 'f_eos_b', 'f_eos_s', 'f_tfc_b', 'f_tfc_s', 'f_tpout_b', 'f_tpout_s'


#df.drop(['Open','High','Low','Close', \
#         'Ref_C_Ups','Ref_C_Downs', \
#         'Close_t-1', 'Close_t-2','High_t+1','High_t-1','Low_t+1','Low_t-1', \
#         'ATR','DeMarker1', \
#         'OB','OS','EOS','EOB', \
#         'BUY_t-1','SELL_t-1','BUY_STOP','SELL_STOP','BUY_TP','SELL_TP', \
#         'L_POS','L_STOP','L_TP','S_POS',
#         'L_POS_t-1', 'S_POS_t-1','L_STOP_t-1','L_TP_t-1','S_STOP_t-1','S_TP_t-1', \
#         'B_HILO','S_HILO', \
#         'Tot_P&L_Longs','Tot_P&L_Shorts'], axis=1, inplace=True)
#    
#df_tf.drop(['Open','High','Low','Close', \
#         'Ref_C_Ups','Ref_C_Downs', \
#         'Close_t-1', 'Close_t-2','High_t+1','High_t-1','Low_t+1','Low_t-1'], axis=1, inplace=True)
#
#if HILO:
#    df.drop(['B_HILO','S_HILO'], axis=1, inplace = True)
#    
#if TF_CONFO:
#    df.drop(['TF_BUY','TF_SELL'], axis=1, inplace = True)
#    df_tf.drop(['Open','High','Low','Close','Ref_C_Ups','Ref_C_Downs', \
#                'Close_t-1','Close_t-2','High_t+1','High_t-1','Low_t+1','Low_t-1'], axis=1, inplace = True)












        
