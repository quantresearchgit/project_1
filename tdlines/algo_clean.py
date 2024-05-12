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


# %%

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

atr_period = 50
rolling_window_lines = 400
HILO_CONFO_STRICT = True
HILO_CONFO_LOOSE = False  # pick either strict or loose

demarker1_period = 13
OB_lvl = 0.60
OS_lvl = 0.40

multiplierTP = 1
multiplierSTOP = 1

PATH = r'C:\Users\manga\Desktop\Trading\nico model\tdlines\\'
MAIN_DF_FILE = 'IKZ7_60.csv'
TF_CONFO_FILE = 'IKZ7_30.csv'
MAIN_DF_FILE = 'IKZ7_30.csv'



PRINT_LINES = False
PRINT_RESULTS = True

def get_data(MAIN_DF_FILE, Skip_Bar_at_9PM):
    df = pd.read_csv(PATH + MAIN_DF_FILE)
    if Daily:
        df.Date = df.Date.apply(lambda x: datetime.datetime.strptime(x, '%d/%m/%y'))
    else:
        df.Date = df.Date.apply(lambda x: datetime.datetime.strptime(x, '%d/%m/%y %H:%M'))
    df.rename(columns={'TD Reference Close Up': 'Ref_C_Ups', 'TD Reference Close Down': 'Ref_C_Downs'}, inplace=True)
    df.index = df.Date
    if Daily == False:
        if Skip_Bar_at_9PM == True:
            df = df.between_time(datetime.time(6, 59), datetime.time(20, 59))  # modify depending on data
        else:
            df = df.between_time(datetime.time(6, 59), datetime.time(21, 00))  # modify depending on data
    df = df.iloc[::-1]
    del df['Date']
    df.reset_index(inplace=True)
    df['Close_t-1'] = df['Close'].shift(+1)  # previous day close
    df['Close_t-2'] = df['Close'].shift(+2)
    df['High_t-1'] = df['High'].shift(+1)
    df['High_t+1'] = df['High'].shift(-1)
    df['Low_t-1'] = df['Low'].shift(+1)
    df['Low_t+1'] = df['Low'].shift(-1)
    df['High_t-2'] = df['High'].shift(+2)
    df['High_t+2'] = df['High'].shift(-2)
    df['Low_t-2'] = df['Low'].shift(+2)
    df['Low_t+2'] = df['Low'].shift(-2)
    return df


def get_td_points(indexes, highs, lows, dates):
    td_highs = []
    td_lows = []

    for idx in indexes[1:-1]:  # skip first and last datapoints
        if (highs[idx] > highs[idx + 1] and highs[idx] > highs[idx - 1]) or \
                (highs[idx] > highs[idx + 1] and highs[idx] == highs[idx - 1] and highs[idx - 1] > highs[
                    idx - 2]):  # if two consecutive tdpoints, take the 2nd
            td_highs.append((idx, dates[idx], highs[idx]))

    for idx in indexes[1:-1]:
        if (lows[idx] < lows[idx + 1] and lows[idx] < lows[idx - 1]) or \
                (lows[idx] < lows[idx + 1] and lows[idx] == lows[idx - 1] and lows[idx - 1] < lows[idx - 2]):
            td_lows.append((idx, dates[idx], lows[idx]))
    return td_highs, td_lows
def add_tr_to_df(df, atr_period):
    df['ATR1'] = df.apply(lambda row: abs(row['High'] - row['Low']), axis=1)
    df['ATR2'] = df.apply(lambda row: abs(row['High'] - row['Close_t-1']), axis=1)
    df['ATR3'] = df.apply(lambda row: abs(row['Low'] - row['Close_t-1']), axis=1)
    df['TR'] = df[['ATR1', 'ATR2', 'ATR3']].max(axis=1)
    df['ATR'] = df['TR'].ewm(span=atr_period).mean()
    df.drop(['ATR1', 'ATR2', 'ATR3', 'TR'], axis=1, inplace=True)


def get_td_lines(td_highs, td_lows, closes):
    td_lines = []
    for tidx_1 in range(len(td_highs) - 1):
        for tidx_2 in range(tidx_1, len(td_highs)):
            if td_highs[tidx_2][0] - td_highs[tidx_1][0] < rolling_window_lines:
                if td_highs[tidx_1][2] > td_highs[tidx_2][2]:
                    m = (td_highs[tidx_2][2] - td_highs[tidx_1][2]) / (td_highs[tidx_2][0] - td_highs[tidx_1][0])
                    b = td_highs[tidx_1][2]
                    line_breached = False
                    for idx in range(td_highs[tidx_1][0] + 1,
                                     td_highs[tidx_2][0]):  # no close higher then line btw the two TDP
                        line_price_y_breach = (m * (idx - td_highs[tidx_1][0]) + b)
                        if closes[idx] > line_price_y_breach:
                            line_breached = True
                            break
                    if not line_breached:
                        td_lines.append(((td_highs[tidx_1][1], td_highs[tidx_1][0], td_highs[tidx_1][2]),
                                         (td_highs[tidx_2][1], td_highs[tidx_2][0], td_highs[tidx_2][2]), m, b))

    # find low lines from td lows
    for tidx_1 in range(len(td_lows) - 1):
        for tidx_2 in range(tidx_1, len(td_lows)):
            if td_lows[tidx_2][0] - td_lows[tidx_1][0] < rolling_window_lines:
                if td_lows[tidx_1][2] < td_lows[tidx_2][2]:
                    m = (td_lows[tidx_2][2] - td_lows[tidx_1][2]) / (td_lows[tidx_2][0] - td_lows[tidx_1][0])
                    b = td_lows[tidx_1][2]
                    line_breached = False
                    for idx in range(td_lows[tidx_1][0] + 1, td_lows[tidx_2][0]):
                        line_price_y_breach = (m * (idx - td_lows[tidx_1][0]) + b)
                        if closes[idx] < line_price_y_breach:
                            line_breached = True
                            break
                    if not line_breached:
                        td_lines.append(((td_lows[tidx_1][1], td_lows[tidx_1][0], td_lows[tidx_1][2]),
                                         (td_lows[tidx_2][1], td_lows[tidx_2][0], td_lows[tidx_2][2]), m, b))
    return td_lines

def revs(indexes,dates,closes ):
    revs_buy = []
    revs_sell = []
    for idx in indexes[2:]:  # skip first and last datapoints
        if closes[idx] > closes[idx - 1] and closes[idx - 1] < closes[idx - 2]:
            revs_buy.append((idx, dates[idx], closes[idx]))
        if closes[idx] < closes[idx - 1] and closes[idx - 1] > closes[idx - 2]:
            revs_sell.append((idx, dates[idx], closes[idx]))
    return revs_buy, revs_sell



def td_lines_signal(indexes,dates,closes, td_lines ):
    buy_signals = []
    sell_signals = []
    buy_signals_date_px_dict = defaultdict(list)
    sell_signals_date_px_dict = defaultdict(list)
    closes_list = []

    for idx in indexes[2:]:  # skip first and last datapoints
            closes_list.append((idx, dates[idx], closes[idx]))

    ''' BUY '''
    for rev_buy in closes_list:
        for td_line in td_lines:
            if rev_buy[0] > td_line[1][1]:  # consider only rev breaks after the second td_point of the line
                if td_line[2] < 0:  # check against negative slope TD lines (m coefficient td_line[2])
                    line_breached = False
                    for idx in range(td_line[1][1],
                                     rev_buy[0]):  # check if no close is higher then line btw rev break and 2nd tdpoint
                        line_price_y_breach = td_line[2] * (idx - td_line[0][1]) + td_line[3]
                        if closes[idx] >= line_price_y_breach:
                            line_breached = True
                            break
                    if not line_breached:
                        # find value of TD line at rev_buy x-axis position
                        line_price_y_rev_break = td_line[2] * (rev_buy[0] - td_line[0][1]) + td_line[3]
                        if rev_buy[2] > line_price_y_rev_break:
                            buy_signals.append((rev_buy, td_line))
                            buy_signals_date_px_dict['Date'].append(rev_buy[1])
                            buy_signals_date_px_dict['BUY'].append(rev_buy[2])

    df_Buy_Signals = pd.DataFrame(buy_signals_date_px_dict)
    df_Buy_Signals.drop_duplicates(subset=['Date'], keep='first', inplace=True)

    ''' SELL '''
    for rev_sell in closes_list:
        for td_line in td_lines:
            if rev_sell[0] > td_line[1][1]:  # consider only rev breaks after the second td_point of the line
                if td_line[2] > 0:  # check against negative slope TD lines (m coefficient td_line[2])
                    line_breached = False
                    for idx in range(td_line[1][1],
                                     rev_sell[
                                         0]):  # check if no close is higher then line btw rev break and 2nd tdpoint
                        line_price_y_breach = td_line[2] * (idx - td_line[0][1]) + td_line[3]
                        if closes[idx] <= line_price_y_breach:
                            line_breached = True
                            break
                    if not line_breached:
                        # find value of TD line at rev_buy x-axis position
                        line_price_y_rev_break = td_line[2] * (rev_sell[0] - td_line[0][1]) + td_line[3]
                        if rev_sell[2] < line_price_y_rev_break:
                            sell_signals.append((rev_sell, td_line))
                            sell_signals_date_px_dict['Date'].append(rev_sell[1])
                            sell_signals_date_px_dict['SELL'].append(rev_sell[2])
    df_Sell_Signals = pd.DataFrame(sell_signals_date_px_dict)
    df_Sell_Signals.drop_duplicates(subset=['Date'], keep='first', inplace=True)
    return df_Sell_Signals, df_Buy_Signals


def get_signal_tdlines(df):
    dates = df.Date.values
    opens = df.Open.values
    lows = df.Low.values
    highs = df.High.values
    indexes = df.index.values
    closes = df.Close.values
    td_highs, td_lows = get_td_points(indexes, highs, lows, dates)
    td_lines = get_td_lines(td_highs, td_lows, closes)
    df_Sell_Signals, df_Buy_Signals = td_lines_signal(indexes,dates,closes, td_lines)
    df_Buy_Signals.columns = ['Date','tdlines_buy']
    df_Buy_Signals.tdlines_buy = np.where(df_Buy_Signals.tdlines_buy!=np.nan, 1, np.nan)
    df_Sell_Signals.columns = ['Date','tdlines_sell']
    df_Sell_Signals.tdlines_sell = np.where(df_Sell_Signals.tdlines_sell!=np.nan, 1, np.nan)
    df = df.merge(df_Buy_Signals, how='left', on='Date')
    df = df.merge(df_Sell_Signals, how='left', on='Date')
    return df


def countConsecutiveEntriesOB(df, demarker1_period):
    res = [0] * len(df)
    count = 0
    for i, e in enumerate(df['OB']):
        if e == True:
            count += 1
            if count >= demarker1_period:
                res[i] = True
        else:
            count = 0
    return res
def countConsecutiveEntriesOS(df, demarker1_period):
    res = [0] * len(df)
    count = 0
    for i, e in enumerate(df['OS']):
        if e == True:
            count += 1
            if count >= demarker1_period:
                res[i] = True
        else:
            count = 0
    return res

def get_signal_demarker(df, demarker1_period, OB_lvl, OS_lvl):
    df['DeMax'] = df.apply(lambda row: (row['High'] - row['High_t-1']) if (row['High'] - row['High_t-1']) >= 0 else 0,
                           axis=1)
    df['DeMin'] = df.apply(lambda row: (- row['Low'] + row['Low_t-1']) if (- row['Low'] + row['Low_t-1']) >= 0 else 0,
                           axis=1)
    df['DeMarker1'] = round(df['DeMax'].rolling(demarker1_period).sum() / (
                df['DeMax'].rolling(demarker1_period).sum() + df['DeMin'].rolling(demarker1_period).sum()), 2)
    df.drop(['DeMax', 'DeMin'], axis=1, inplace=True)
    df['OB'] = df.apply(lambda row: True if row['DeMarker1'] > OB_lvl else False, axis=1)
    df['OS'] = df.apply(lambda row: True if row['DeMarker1'] < OS_lvl else False, axis=1)
    df['EOB'] = countConsecutiveEntriesOB(df, demarker1_period)
    df['EOS'] = countConsecutiveEntriesOS(df, demarker1_period)
    df['demarker_buy'] = np.where(df.EOS == False, 1, 0)
    df['demarker_sell'] = np.where(df.EOB == False, 1, 0)
    df.drop(['OB', 'OS', 'EOB', 'EOS', 'DeMarker1'], axis=1, inplace=True)
    return df


def get_signal_atr(df, atr_period):
    df['ATR1'] = df.apply(lambda row: abs(row['High'] - row['Low']), axis=1)
    df['ATR2'] = df.apply(lambda row: abs(row['High'] - row['Close_t-1']), axis=1)
    df['ATR3'] = df.apply(lambda row: abs(row['Low'] - row['Close_t-1']), axis=1)
    df['TR'] = df[['ATR1', 'ATR2', 'ATR3']].max(axis=1)
    df['ATR'] = df['TR'].ewm(span=atr_period).mean()
    df['atr_buy'] = np.where((df['Close'] - df['Low']) > df['ATR'],1, 0)
    df['atr_sell'] = np.where((- df['Close'] + df['High']) > df['ATR'],1, 0)
    df.drop(['ATR1', 'ATR2', 'ATR3', 'TR', 'ATR'], axis=1, inplace=True)
    return df


def revs(indexes,dates,closes ):
    revs_buy = []
    revs_sell = []
    for idx in indexes[2:]:  # skip first and last datapoints
        if closes[idx] > closes[idx - 1] and closes[idx - 1] < closes[idx - 2]:
            revs_buy.append((idx, dates[idx], closes[idx]))
        if closes[idx] < closes[idx - 1] and closes[idx - 1] > closes[idx - 2]:
            revs_sell.append((idx, dates[idx], closes[idx]))
    return revs_buy, revs_sell

def get_signal_rev(df):
    df['rev_buy'] = np.where( (df['Close'] > df.Close.shift()) & (df['Close'].shift() < df.Close.shift(2)),1, 0)
    df['rev_sell'] = np.where( (df['Close'] < df.Close.shift()) & (df['Close'].shift() > df.Close.shift(2)),1, 0)
    return df

def get_signal_tpouter(df):
    df['tpouter_buy'] = np.where(  (df['BUY_TP'] + df.Close)> df.High ,1, 0)
    df['tpouter_sell'] =  np.where( (df.Close - df['SELL_TP']) < df.Low ,1, 0)
    return df

def get_signal_revhilo(df):
    df['revhilo_buy'] = np.where(  df['High']> df.High.shift() ,1, 0)
    df['revhilo_sell'] =  np.where( df.Low  < df.Low.shift() ,1, 0)
    return df
##################################################################################################################
##################################################################################################################
##################################################################################################################

def trading_long(df):
        df['Live_P&L_Longs'] = 0.0
        for row in range(len(df) - 1):
            if df.loc[row, 'L_POS_t-1'] > 0:
                df.loc[row, 'L_POS'] = df.loc[row, 'L_POS_t-1']
                df.loc[row, 'L_STOP'] = df.loc[row, 'L_STOP_t-1']
                df.loc[row, 'L_TP'] = df.loc[row, 'L_TP_t-1']
                if df.loc[row, 'Low'] <= df.loc[row, 'L_STOP']:
                    df.loc[row, 'L_POS'] = np.NaN
                    df.loc[row, 'EXIT'] = 'STOP'
                    df.loc[row, 'Live_P&L_Longs'] = (df.loc[row, 'L_STOP'] - df.loc[row, 'Close_t-1']) * df.loc[
                        row, 'L_POS_t-1'] * 1000
                elif df.loc[row, 'Low'] > df.loc[row, 'L_STOP']:
                    if df.loc[row, 'High'] >= df.loc[row, 'L_TP']:
                        df.loc[row, 'L_POS'] = np.NaN
                        df.loc[row, 'EXIT'] = 'TP'
                        df.loc[row, 'Live_P&L_Longs'] = (df.loc[row, 'L_TP'] - df.loc[row, 'Close_t-1']) * df.loc[
                            row, 'L_POS_t-1'] * 1000
                    elif df.loc[row, 'Close'] < df.loc[row, 'L_TP']:
                        if HILO:
                            if df.loc[row, 'B_HILO'] == True:
                                df.loc[row + 1, 'L_POS_t-1'] = df.loc[row, 'L_POS_t-1']
                                df.loc[row + 1, 'L_STOP_t-1'] = df.loc[row, 'L_STOP_t-1']
                                df.loc[row + 1, 'L_TP_t-1'] = df.loc[row, 'L_TP_t-1']
                                df.loc[row + 1, 'B_HILO'] = df.loc[row, 'B_HILO']
                                df.loc[row, 'Live_P&L_Longs'] = (df.loc[row, 'Close'] - df.loc[row, 'Close_t-1']) * df.loc[
                                    row, 'L_POS_t-1'] * 1000
                            elif df.loc[row, 'B_HILO'] == False:
                                df.loc[row, 'L_POS'] = np.NaN
                                df.loc[row, 'EXIT'] = 'NO HILO'
                                df.loc[row, 'Live_P&L_Longs'] = (df.loc[row, 'Close'] - df.loc[row, 'Close_t-1']) * df.loc[
                                    row, 'L_POS_t-1'] * 1000
                        else:
                            df.loc[row + 1, 'L_POS_t-1'] = df.loc[row, 'L_POS_t-1']
                            df.loc[row + 1, 'L_STOP_t-1'] = df.loc[row, 'L_STOP_t-1']
                            df.loc[row + 1, 'L_TP_t-1'] = df.loc[row, 'L_TP_t-1']
                            df.loc[row, 'Live_P&L_Longs'] = (df.loc[row, 'Close'] - df.loc[row, 'Close_t-1']) * df.loc[
                                row, 'L_POS_t-1'] * 1000
        df['Tot_P&L_Longs'] = df['Live_P&L_Longs'].cumsum()

def trading_short(df):
        df['Live_P&S_Shorts'] = 0.0
        for row in range(len(df) - 1):
            if df.loc[row, 'S_POS_t-1'] > 0:
                df.loc[row, 'S_POS'] = df.loc[row, 'S_POS_t-1']
                df.loc[row, 'S_STOP'] = df.loc[row, 'S_STOP_t-1']
                df.loc[row, 'S_TP'] = df.loc[row, 'S_TP_t-1']
                if df.loc[row, 'High'] >= df.loc[row, 'S_STOP']:
                    df.loc[row, 'S_POS'] = np.NaN
                    df.loc[row, 'EXIT'] = 'STOP'
                    df.loc[row, 'Live_P&S_Shorts'] = (- df.loc[row, 'S_STOP'] + df.loc[row, 'Close_t-1']) * df.loc[
                        row, 'S_POS_t-1'] * 1000
                elif df.loc[row, 'High'] < df.loc[row, 'S_STOP']:
                    if df.loc[row, 'Low'] <= df.loc[row, 'S_TP']:
                        df.loc[row, 'S_POS'] = np.NaN
                        df.loc[row, 'EXIT'] = 'TP'
                        df.loc[row, 'Live_P&S_Shorts'] = (- df.loc[row, 'S_TP'] + df.loc[row, 'Close_t-1']) * df.loc[
                            row, 'S_POS_t-1'] * 1000
                    elif df.loc[row, 'Close'] > df.loc[row, 'S_TP']:
                        if HILO:
                            if df.loc[row, 'S_HILO'] == True:
                                df.loc[row + 1, 'S_POS_t-1'] = df.loc[row, 'S_POS_t-1']
                                df.loc[row + 1, 'S_STOP_t-1'] = df.loc[row, 'S_STOP_t-1']
                                df.loc[row + 1, 'S_TP_t-1'] = df.loc[row, 'S_TP_t-1']
                                df.loc[row + 1, 'S_HILO'] = df.loc[row, 'S_HILO']
                                df.loc[row, 'Live_P&S_Shorts'] = (- df.loc[row, 'Close'] + df.loc[row, 'Close_t-1']) * \
                                                                 df.loc[row, 'S_POS_t-1'] * 1000
                            elif df.loc[row, 'S_HILO'] == False:
                                df.loc[row, 'S_POS'] = np.NaN
                                df.loc[row, 'EXIT'] = 'NO HILO'
                                df.loc[row, 'Live_P&S_Shorts'] = (- df.loc[row, 'Close'] + df.loc[row, 'Close_t-1']) * \
                                                                 df.loc[row, 'S_POS_t-1'] * 1000
                        else:
                            df.loc[row + 1, 'S_POS_t-1'] = df.loc[row, 'S_POS_t-1']
                            df.loc[row + 1, 'S_STOP_t-1'] = df.loc[row, 'S_STOP_t-1']
                            df.loc[row + 1, 'S_TP_t-1'] = df.loc[row, 'S_TP_t-1']
                            df.loc[row, 'Live_P&S_Shorts'] = (- df.loc[row, 'Close'] + df.loc[row, 'Close_t-1']) * df.loc[
                                row, 'S_POS_t-1'] * 1000
        df['Tot_P&L_Shorts'] = df['Live_P&S_Shorts'].cumsum()

def run_backtest(df):
    trading_long(df)
    trading_short(df)
    df['TRADEx1'] = df.apply(lambda row: 'BUY' if row['TRADE'] == 'BUY' and np.isnan(row['L_POS_t-1']) else ( 'SELL' if row['TRADE'] == 'SELL' and np.isnan(row['S_POS_t-1']) else np.NaN), axis=1)
    df['Live_P&L'] = df.apply(lambda row: row['Live_P&L_Longs'] + row['Live_P&S_Shorts'], axis=1)
    df['Tot_P&L'] = df.apply(lambda row: row['Tot_P&L_Longs'] + row['Tot_P&L_Shorts'], axis=1)
    return df

if __name__ == '__main__':
    df = get_data(MAIN_DF_FILE, Skip_Bar_at_9PM)

    df['BUY_TP'] = df['Close'] - df['Low']
    df['SELL_TP'] = - df['Close'] + df['High']
    df['BUY_STOP'] = df['Low']
    df['SELL_STOP'] = df['High']

    df = get_signal_tdlines(df)
    df = get_signal_demarker(df, demarker1_period, OB_lvl, OS_lvl)
    df = get_signal_atr(df, atr_period)
    df = get_signal_rev(df)
    df = get_signal_tpouter(df)
    df = get_signal_revhilo(df)

    df['BUY']  = np.where( (df.tdlines_buy==1) & (df.demarker_buy==1) & (df.atr_buy==1) & (df.rev_buy==1) & (df.tpouter_buy==1), df.Close, np.nan  )
    df['SELL']  = np.where( (df.tdlines_sell==1) & (df.demarker_sell==1) & (df.atr_sell==1) & (df.rev_sell==1) & (df.tpouter_sell==1), df.Close, np.nan  )

    # df['BUY']  = np.where( (df.tdlines_buy==1) , df.Close, np.nan  )
    # df['SELL']  = np.where( (df.tdlines_sell==1)  , df.Close, np.nan  )


    ''' SIZING '''
    dollar_risk = RISK * CAPITAL
    df['BUY_t-1'] = df['BUY'].shift(+1)
    df['SELL_t-1'] = df['SELL'].shift(+1)
    df['L_POS'] = df.apply(lambda row: int((dollar_risk / 1000) / (row['Close'] - row['Low'])) if not np.isnan(row['BUY']) and row['Close'] !=row['Low'] else np.NaN, axis=1)
    df['L_TP'] = df.apply(lambda row: (row['BUY_TP'] + row['BUY']) if not np.isnan(row['BUY']) else np.NaN, axis=1)
    df['L_STOP'] = df.apply(lambda row: row['BUY_STOP'] if not np.isnan(row['BUY']) else np.NaN, axis=1)
    df['S_POS'] = df.apply(lambda row: int((dollar_risk / 1000) / (- row['Close'] + row['High'])) if not np.isnan(row['SELL']) and row[
            'Close'] != row['High'] else np.NaN, axis=1)
    df['S_TP'] = df.apply(lambda row: (- row['SELL_TP'] + row['SELL']) if not np.isnan(row['SELL']) else np.NaN, axis=1)
    df['S_STOP'] = df.apply(lambda row: row['SELL_STOP'] if not np.isnan(row['SELL']) else np.NaN, axis=1)
    df['L_POS_t-1'] = df['L_POS'].shift(+1)
    df['L_STOP_t-1'] = df['L_STOP'].shift(+1)
    df['L_TP_t-1'] = df['L_TP'].shift(+1)
    df['S_POS_t-1'] = df['S_POS'].shift(+1)
    df['S_STOP_t-1'] = df['S_STOP'].shift(+1)
    df['S_TP_t-1'] = df['S_TP'].shift(+1)


    ''' TRADING '''
    df['B_HILO'] = df.apply(
        lambda row: True if not np.isnan(row['BUY_t-1']) and row['High'] > row['High_t-1'] else False, axis=1)
    df['S_HILO'] = df.apply(
        lambda row: True if not np.isnan(row['SELL_t-1']) and row['Low'] < row['Low_t-1'] else False, axis=1)
    df['TRADE'] = df.apply(lambda row: 'BUY' if not np.isnan(row['BUY']) else ('SELL' if not np.isnan(row['SELL']) else np.NAN), axis=1)
    df['TRADE_t-1'] = df['TRADE'].shift(+1)

    df = run_backtest(df)
    plt.plot(df['Tot_P&L'])
