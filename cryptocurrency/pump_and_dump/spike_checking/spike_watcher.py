#!/usr/bin/python3.8
# -*- coding: utf-8 -*-

import sys
import numpy as np
import pickle as rick
import time
import datetime
import pandas as pd
import requests


def dt_transform(binstamp):
    return datetime.datetime.fromtimestamp(int(binstamp)/1000)


def binstamp_to_timestamp(df):
    df.open_time = df.open_time.apply(dt_transform)
    df.close_time = df.close_time.apply(dt_transform)
    return


def getKlines(symbol, interval, limit=200):
    url = "https://api.binance.com/api/v3/klines?symbol=" + symbol + "&interval=" + interval

    r_json = requests.get(url + "&limit=" + str(limit))
    if int(r_json.headers['X-MBX-USED-WEIGHT-1M']) > 1000:
        print("Query weight close to limit (1200), shutting down the bot...")
        sys.exit()

    if r_json.status_code == 429 or r_json.status_code == 418:
        print("Weight limit of the API excedeed with code" + str(r_json.status_code))
        sys.exit()

    if r_json.status_code != 200:
        print(url)
        print("Exiting because of unhandled status code from http header:")
        print("Status Code : " + str(r_json.status_code))
        print(str(r_json.json()))
        sys.exit()

    return pd.DataFrame(r_json.json(), columns=["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "trade_number", "taker_buy_asset", "taker_quote_asset", "ignore"])


def compute_features(kline_dictionary, btc_price):
    total_pump_file = open("./data/live_features.pkl", "rb")
    live_features = rick.load(total_pump_file)
    total_pump_file.close()

    rm_key = []
    xs = [1, 3, 12, 24, 36, 48, 60, 72]
    ys = [3, 12, 24, 36, 48, 60, 72]

    for symbol in kline_dictionary.keys():
        df = kline_dictionary[symbol]
        df['listing_date'] = live_features[symbol][1]
        df['existence_time'] = (df.open_time - live_features[symbol][1]).dt.days

        df['open'] = df.open.astype(np.float)
        df['high'] = df.high.astype(np.float)
        df['low'] = df.low.astype(np.float)
        df['close'] = df.close.astype(np.float)
        df['volume'] = df.volume.astype(np.float)
        df['quote_asset_volume'] = df.quote_asset_volume.astype(np.float)
        df['trade_number'] = df.volume.astype(np.int)
        btc_price = btc_price.astype(np.float)

        df['ret1'] = np.log(df.open) - np.log(df.open.shift(1))
        df.loc[df.close <= df.open, 'amplitude_intra'] = (df.low - df.high) * 100 / df.open
        df.loc[df.close > df.open, 'amplitude_intra'] = (df.high - df.low) * 100 / df.open
        df["pump_count"] = live_features[symbol][0]
        df = pd.merge(df, btc_price, how='inner', left_index=True, right_index=True)
        df["market_cap"] = df.open * df.volume * df.btc_price

        for x in xs:
            strx = str(x)
            ret = "ret" + strx
            volf = "volf" + strx
            volbtc = "volbtc" + strx
            df[ret] = df['ret1'].rolling(x).sum()
            df[volf] = df['volume'].shift(1).rolling(x).sum() # FIXME : volume is look ahead, shift ?
            df[volbtc] = (df['volume'].shift(1) * df['open']) / df['btc_price'] # FIXME : volume is look ahead, shift ?

        for y in ys:
            stry = str(y)
            vola = "vola" + stry
            volavol = "volavol" + stry
            rtvol = "rtvol" + stry
            df[vola] = df['ret1'].rolling(y).std()
            df[volavol] = df['volf' + str(y)].rolling(y).std()
            df[rtvol] = df['volbtc' + str(y)].rolling(y).std()

        df['last_open_price'] = df.open.shift(1)
        df['quote_asset_volume'] = df.quote_asset_volume.shift(1)
        df['amplitude_intra'] = df.amplitude_intra.shift(1)

        kline_dictionary[symbol] = df.dropna()

    return rm_key


if __name__ == '__main__':
    symbols = [ 'LTCBTC', 'BNBBTC', 'NEOBTC', 'QTUMBTC', 'EOSBTC', 'SNTBTC', 'BNTBTC', 'GASBTC', 'OAXBTC',
                 'DNTBTC', 'WTCBTC', 'LRCBTC', 'OMGBTC', 'ZRXBTC', 'SNGLSBTC', 'KNCBTC', 'FUNBTC', 'SNMBTC', 'LINKBTC',
                 'XVGBTC', 'MDABTC', 'MTLBTC', 'ETCBTC', 'MTHBTC', 'ZECBTC', 'ASTBTC', 'DASHBTC', 'BTGBTC', 'EVXBTC',
                 'REQBTC', 'VIBBTC', 'TRXBTC', 'POWRBTC', 'ARKBTC', 'XRPBTC', 'ENJBTC', 'STORJBTC', 'KMDBTC', 'NULSBTC',
                 'RCNBTC', 'RDNBTC', 'XMRBTC', 'DLTBTC', 'AMBBTC', 'BATBTC', 'GVTBTC', 'CDTBTC', 'QSPBTC', 'BTSBTC',
                 'LSKBTC', 'MANABTC', 'BCDBTC', 'ADXBTC', 'ADABTC', 'PPTBTC', 'XLMBTC', 'CNDBTC', 'WABIBTC', 'WAVESBTC',
                 'GTOBTC', 'ICXBTC', 'OSTBTC', 'ELFBTC', 'AIONBTC', 'NEBLBTC', 'BRDBTC', 'NAVBTC', 'APPCBTC', 'RLCBTC',
                 'PIVXBTC', 'IOSTBTC', 'STEEMBTC', 'NANOBTC', 'VIABTC', 'BLZBTC', 'POABTC', 'ZILBTC', 'ONTBTC',
                 'XEMBTC',
                 'WANBTC', 'WPRBTC', 'QLCBTC', 'SYSBTC', 'GRSBTC', 'LOOMBTC', 'REPBTC', 'ZENBTC', 'SKYBTC', 'CVCBTC',
                 'THETABTC', 'IOTXBTC', 'QKCBTC', 'AGIBTC', 'NXSBTC', 'DATABTC', 'SCBTC', 'NASBTC', 'ARDRBTC', 'VETBTC',
                 'DOCKBTC', 'POLYBTC', 'GOBTC', 'RVNBTC', 'DCRBTC', 'MITHBTC', 'RENBTC', 'ONGBTC', 'FETBTC', 'CELRBTC',
                 'MATICBTC', 'ATOMBTC', 'PHBBTC', 'TFUELBTC', 'ONEBTC', 'FTMBTC', 'ALGOBTC', 'DOGEBTC', 'DUSKBTC',
                 'ANKRBTC', 'COSBTC', 'TOMOBTC', 'PERLBTC', 'CHZBTC', 'BANDBTC', 'BEAMBTC', 'XTZBTC', 'HBARBTC',
                 'NKNBTC', 'STXBTC', 'KAVABTC', 'ARPABTC', 'CTXCBTC', 'BCHBTC', 'TROYBTC', 'VITEBTC', 'FTTBTC',
                 'OGNBTC',
                 'DREPBTC', 'TCTBTC', 'WRXBTC', 'LTOBTC', 'COTIBTC', 'STPTBTC', 'SOLBTC', 'CTSIBTC', 'HIVEBTC',
                 'CHRBTC',
                 'MDTBTC', 'STMXBTC', 'PNTBTC', 'DGBBTC', 'COMPBTC', 'SXPBTC', 'SNXBTC', 'IRISBTC', 'MKRBTC', 'RUNEBTC',
                 'FIOBTC', 'AVABTC', 'BALBTC', 'YFIBTC', 'JSTBTC', 'SRMBTC', 'ANTBTC', 'CRVBTC', 'SANDBTC', 'OCEANBTC',
                 'NMRBTC', 'DOTBTC', 'LUNABTC', 'IDEXBTC', 'RSRBTC', 'PAXGBTC', 'WNXMBTC', 'TRBBTC', 'BZRXBTC',
                 'SUSHIBTC', 'YFIIBTC', 'KSMBTC', 'EGLDBTC', 'DIABTC', 'UMABTC', 'BELBTC', 'WINGBTC',
                 'UNIBTC', 'NBSBTC', 'OXTBTC', 'SUNBTC', 'AVAXBTC', 'HNTBTC', 'FLMBTC', 'SCRTBTC', 'CAKEBTC', 'ORNBTC',
                 'UTKBTC', 'XVSBTC', 'ALPHABTC', 'VIDTBTC', 'AAVEBTC', 'NEARBTC', 'FILBTC', 'INJBTC', 'AERGOBTC',
                 'EASYBTC', 'AUDIOBTC', 'CTKBTC', 'AKROBTC', 'AXSBTC', 'HARDBTC', 'RENBTCBTC', 'STRAXBTC', 'FORBTC',
                 'UNFIBTC', 'FRONTBTC', 'ROSEBTC', 'SKLBTC', 'SUSDBTC', 'GLMBTC', 'GRTBTC', 'JUVBTC', 'PSGBTC',
                 'REEFBTC', 'OGBTC', 'ATMBTC', 'ASRBTC', 'CELOBTC', 'RIFBTC', 'BTCSTBTC', 'TRUBTC', 'CKBBTC', 'TWTBTC',
                 'FIROBTC', 'LITBTC', 'SFPBTC', 'FXSBTC', 'DODOBTC', 'ACMBTC', 'AUCTIONBTC', 'PHABTC', 'TVKBTC',
                 'BADGERBTC', 'FISBTC', 'OMBTC', 'PONDBTC', 'DEGOBTC', 'LINABTC', 'PERPBTC']

    file_model = open("./data/model_spike_temp.pkl", "rb")
    model = rick.load(file_model)
    counter = 0
    timeframe = "1h"
    timeframe_seconds = 3600  # 1 hour
    auto_stop = 24  # 1 day
    look_back = 72  # 3 days

    while counter < auto_stop:
        counter += 1

        btc_klines = getKlines("BTCUSDT", timeframe)
        btc_price = btc_klines.close
        btc_price.name = "btc_price"

        kline_dictionary = {}
        for symbol in symbols:
            kline_dictionary[symbol] = getKlines(symbol, timeframe)
            binstamp_to_timestamp(kline_dictionary[symbol])

        df = compute_features(kline_dictionary, btc_price)


        seer_buffer = {}
        for symbol in kline_dictionary:
            last_row = kline_dictionary[symbol].tail(1)
            cols_to_drop = ["open_time", "open", "high", "low", "close", "volume", "close_time", "taker_buy_asset",
                            "taker_quote_asset", "ignore", "listing_date"]
            feature_input = last_row.drop(cols_to_drop, axis=1)
            if feature_input.isnull().values.any() or feature_input.empty:
                continue
            oracle = model.predict_proba(feature_input)

            seer_buffer[symbol] = {}
            seer_buffer[symbol]["pos"] = oracle[0][1]
            seer_buffer[symbol]["neg"] = oracle[0][0]

        unsorted_predictions = pd.DataFrame(seer_buffer).transpose().sort_values("pos")
        negs = unsorted_predictions.loc[unsorted_predictions.neg >= 0.5]
        poss = unsorted_predictions.loc[unsorted_predictions.neg < 0.5]

        print("### NEGATIVE " + "#" * 30)
        for line in negs.iterrows():
            print(line[0] + ": " + str(line[1]["pos"]) + "|" + str(line[1]["neg"]))

        print("### POSITIVE " + "#" * 30)
        for line in poss.iterrows():
            print(line[0] + ": " + str(line[1]["pos"]) + "|" + str(line[1]["neg"]))

        time_delta = timeframe_seconds - (int(datetime.datetime.now().strftime('%s')) % timeframe_seconds)
        print("Timedelta = " + str(time_delta) + " seconds")
        time.sleep(time_delta + 1)
