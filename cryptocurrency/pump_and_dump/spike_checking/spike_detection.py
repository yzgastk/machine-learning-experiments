#!/usr/bin/python3.8
#-*- coding: utf-8 -*-


import numpy as np
import datetime
import pandas as pd
import sqlite3 as s3
import matplotlib.pyplot as plt
import pickle as rick

def humanDate(date):
    secondsDate = int(date) / 1000
    hDate = datetime.datetime.fromtimestamp(secondsDate)
    return hDate

def createPandaFrame(conn, tablename, cols, convert_date=True):
    loadQuery = "SELECT * FROM " + tablename + " ORDER BY open_time;"
    cur = conn.execute(loadQuery)
    pdFrame = pd.DataFrame(cur.fetchall())
    pdFrame.columns = cols

    if convert_date:
        pdFrame['open_time'] = pdFrame['open_time'].apply(humanDate)
        pdFrame['close_time_decomp'] = pd.to_datetime(pdFrame['close_time'].apply(lambda x: int(x) / 1000))
        pdFrame['close_time'] = pdFrame['close_time'].apply(humanDate)
    else:
        pdFrame['open_time'] = pd.to_datetime(pdFrame['open_time'])
        pdFrame['close_time'] = pd.to_datetime(pdFrame['close_time'])

    return pdFrame


def get_cmc():
    from requests import Request, Session
    from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
    import json

    url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
    parameters = {
        'start':'1',
        'limit':'5000',
        'convert':'USD'
    }
    headers = {
        'Accepts': 'application/json',
        'X-CMC_PRO_API_KEY': 'your-api-key-here',
    }

    session = Session()
    session.headers.update(headers)

    try:
        response = session.get(url, params=parameters)
        data = json.loads(response.text)
        print(data)
    except (ConnectionError, Timeout, TooManyRedirects) as e:
        print(e)

    return data


def dt_transform(binstamp):
    return datetime.datetime.fromtimestamp(int(binstamp)/1000)


def binstamp_to_timestamp(df):
    df.open_time = df.open_time.apply(dt_transform)
    df.close_time = df.close_time.apply(dt_transform)
    return


def plot_spike(df):
    plt.figure(figsize=(20,15))
    plt.plot(df.index, df['close'], color='lightgrey')
    df['label'].replace([0, -1], np.nan)
    df['pos'] = np.where(df['label'] == 1, df['close'], np.nan)
    print(df['label'].value_counts())
    plt.plot(df.index, df['pos'], 'o', color="green")
    plt.show()


if __name__ == '__main__':
    symbols = ['LTCBTC', 'BNBBTC', 'NEOBTC', 'QTUMBTC', 'EOSBTC', 'SNTBTC', 'BNTBTC', 'GASBTC', 'OAXBTC', 'DNTBTC', 'WTCBTC', 'LRCBTC', 'OMGBTC', 'ZRXBTC', 'SNGLSBTC', 'KNCBTC', 'FUNBTC', 'SNMBTC', 'LINKBTC', 'XVGBTC', 'MDABTC', 'MTLBTC', 'ETCBTC', 'MTHBTC', 'ZECBTC', 'ASTBTC', 'DASHBTC', 'BTGBTC', 'EVXBTC', 'REQBTC', 'VIBBTC', 'TRXBTC', 'POWRBTC', 'ARKBTC', 'XRPBTC', 'ENJBTC', 'STORJBTC', 'KMDBTC', 'NULSBTC', 'RCNBTC', 'RDNBTC', 'XMRBTC', 'DLTBTC', 'AMBBTC', 'BATBTC', 'GVTBTC', 'CDTBTC', 'QSPBTC', 'BTSBTC', 'LSKBTC', 'MANABTC', 'BCDBTC', 'ADXBTC', 'ADABTC', 'PPTBTC', 'XLMBTC', 'CNDBTC', 'WABIBTC', 'WAVESBTC', 'GTOBTC', 'ICXBTC', 'OSTBTC', 'ELFBTC', 'AIONBTC', 'NEBLBTC', 'BRDBTC', 'NAVBTC', 'APPCBTC', 'RLCBTC', 'PIVXBTC', 'IOSTBTC', 'STEEMBTC', 'NANOBTC', 'VIABTC', 'BLZBTC', 'POABTC', 'ZILBTC', 'ONTBTC', 'XEMBTC', 'WANBTC', 'WPRBTC', 'QLCBTC', 'SYSBTC', 'GRSBTC', 'LOOMBTC', 'REPBTC', 'ZENBTC', 'SKYBTC', 'CVCBTC', 'THETABTC', 'IOTXBTC', 'QKCBTC', 'AGIBTC', 'NXSBTC', 'DATABTC', 'SCBTC', 'NASBTC', 'ARDRBTC', 'VETBTC', 'DOCKBTC', 'POLYBTC', 'GOBTC', 'RVNBTC', 'DCRBTC', 'MITHBTC', 'RENBTC', 'ONGBTC', 'FETBTC', 'CELRBTC', 'MATICBTC', 'ATOMBTC', 'PHBBTC', 'TFUELBTC', 'ONEBTC', 'FTMBTC', 'ALGOBTC', 'DOGEBTC', 'DUSKBTC', 'ANKRBTC', 'COSBTC', 'TOMOBTC', 'PERLBTC', 'CHZBTC', 'BANDBTC', 'BEAMBTC', 'XTZBTC', 'HBARBTC', 'NKNBTC', 'STXBTC', 'KAVABTC', 'ARPABTC', 'CTXCBTC', 'BCHBTC', 'TROYBTC', 'VITEBTC', 'FTTBTC', 'OGNBTC', 'DREPBTC', 'TCTBTC', 'WRXBTC', 'LTOBTC', 'COTIBTC', 'STPTBTC', 'SOLBTC', 'CTSIBTC', 'HIVEBTC', 'CHRBTC', 'MDTBTC', 'STMXBTC', 'PNTBTC', 'DGBBTC', 'COMPBTC', 'SXPBTC', 'SNXBTC', 'IRISBTC', 'MKRBTC', 'RUNEBTC', 'FIOBTC', 'AVABTC', 'BALBTC', 'YFIBTC', 'JSTBTC', 'SRMBTC', 'ANTBTC', 'CRVBTC', 'SANDBTC', 'OCEANBTC', 'NMRBTC', 'DOTBTC', 'LUNABTC', 'IDEXBTC', 'RSRBTC', 'PAXGBTC', 'WNXMBTC', 'TRBBTC', 'BZRXBTC', 'SUSHIBTC', 'YFIIBTC', 'KSMBTC', 'EGLDBTC', 'DIABTC', 'UMABTC', 'BELBTC', 'WINGBTC', 'UNIBTC', 'NBSBTC', 'OXTBTC', 'SUNBTC', 'AVAXBTC', 'HNTBTC', 'FLMBTC', 'SCRTBTC', 'CAKEBTC', 'ORNBTC', 'UTKBTC', 'XVSBTC', 'ALPHABTC', 'VIDTBTC', 'AAVEBTC', 'NEARBTC', 'FILBTC', 'INJBTC', 'AERGOBTC', 'EASYBTC', 'AUDIOBTC', 'CTKBTC', 'AKROBTC', 'AXSBTC', 'HARDBTC', 'RENBTCBTC', 'STRAXBTC', 'FORBTC', 'UNFIBTC', 'FRONTBTC', 'ROSEBTC', 'SKLBTC', 'SUSDBTC', 'GLMBTC', 'GRTBTC', 'JUVBTC', 'PSGBTC', 'REEFBTC', 'OGBTC', 'ATMBTC', 'ASRBTC', 'CELOBTC', 'RIFBTC', 'BTCSTBTC', 'TRUBTC', 'CKBBTC', 'TWTBTC', 'FIROBTC', 'LITBTC', 'SFPBTC', 'FXSBTC', 'DODOBTC', 'ACMBTC', 'AUCTIONBTC', 'PHABTC', 'TVKBTC', 'BADGERBTC', 'FISBTC', 'OMBTC', 'PONDBTC', 'DEGOBTC', 'LINABTC', 'PERPBTC']
    conn_output = s3.connect("./data/spike_data.sqlite")
    conn = s3.connect("./data/pd_crypto.sqlite")
    cols = ['open_time', 'btc_price', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'trade_number', 'tb_base_av', 'tb_quote_av', 'label']
    timeframe = "1h"
    btc_price = createPandaFrame(conn, "BTCUSDT" + "_KLINES" + timeframe, cols)
    btc_price = btc_price.set_index('open_time')
    btc_price = btc_price['btc_price']
    btc_price = btc_price[~btc_price.index.duplicated(keep="first")]

    xs = [1, 3, 12, 24, 36, 48, 60, 72]
    ys = [3, 12, 24, 36, 48, 60, 72]

    total_spike = 0
    total_row = 0
    pump_count_dict = {}

    for symbol in symbols:
        coin_klines = pd.read_sql('SELECT * FROM  '+symbol+'_KLINES'+timeframe+' ORDER BY open_time;', conn)

        binstamp_to_timestamp(coin_klines)
        coin_klines = coin_klines.set_index("open_time")

        listing_date = coin_klines.index[0]
        coin_klines['listing_date'] = listing_date
        coin_klines['existence_time'] = (coin_klines.index - listing_date).days
        coin_klines = coin_klines[~coin_klines.index.duplicated(keep="first")]

        coin_klines['ret1'] = np.log(coin_klines.open) - np.log(coin_klines.open.shift(1))
        coin_klines.loc[coin_klines.close <= coin_klines.open, 'amplitude_intra'] = (coin_klines.low - coin_klines.high) * 100 / coin_klines.open
        coin_klines.loc[coin_klines.close > coin_klines.open, 'amplitude_intra'] = (coin_klines.high - coin_klines.low) * 100 / coin_klines.open
        coin_klines.label = 0
        coin_klines.loc[coin_klines['amplitude_intra'] > 20, "label"] = 1

        spikes = coin_klines[coin_klines.label == 1]
        coin_klines['pump_count'] = 0
        last_date = "2010-01-01"
        pump_count = 0
        for idx, spike in spikes.iterrows():
            coin_klines.loc[np.logical_and(coin_klines.index >= last_date, coin_klines.index < idx), 'pump_count'] = pump_count
            last_date = idx
            pump_count += 1
        coin_klines.loc[coin_klines.index >= last_date, "pump_count"] = pump_count
        pump_count_dict[symbol] = []
        pump_count_dict[symbol].append(pump_count)
        pump_count_dict[symbol].append(listing_date)
        coin_klines = pd.merge(coin_klines, btc_price, how='inner', left_index=True, right_index=True)
        coin_klines["market_cap"] = coin_klines.open * coin_klines.volume * coin_klines.btc_price

        for x in xs:
            strx = str(x)
            ret = "ret" + strx
            volf = "volf" + strx
            volbtc = "volbtc" + strx
            coin_klines[ret] = coin_klines['ret1'].rolling(x).sum()
            coin_klines[volf] = coin_klines['volume'].shift(1).rolling(x).sum()
            coin_klines[volbtc] = (coin_klines['volume'].shift(1) * coin_klines['open']) / coin_klines['btc_price']

        for y in ys:
            stry = str(y)
            vola = "vola" + stry
            volavol = "volavol" + stry
            rtvol = "rtvol" + stry
            coin_klines[vola] = coin_klines['ret1'].rolling(y).std()
            coin_klines[volavol] = coin_klines['volf' + str(y)].rolling(y).std()
            coin_klines[rtvol] = coin_klines['volbtc' + str(y)].rolling(y).std()

        coin_klines['last_open_price'] = coin_klines.open.shift(1)
        coin_klines['quote_asset_volume'] = coin_klines.quote_asset_volume.shift(1)
        coin_klines['amplitude_intra'] = coin_klines.amplitude_intra.shift(1)
        coin_klines['symbol'] = symbol[:-3]

        coin_klines = coin_klines.dropna()

        if coin_klines.shape[0] < 1:
            print("Not enough data for: " + symbol)
            continue

        ratio = coin_klines[coin_klines.label == 1].shape[0] / coin_klines.shape[0]
        if ratio == 0:
            pass
        elif ratio > 0:
            total_spike += coin_klines[coin_klines.label == 1].shape[0]
            total_row += coin_klines.shape[0]
            coin_klines.to_sql("aggregated_data", conn_output, if_exists='append', index=False)
        print(symbol+": "+str(ratio))

    total_pump_file = open("./data/live_features.pkl", "wb")
    rick.dump(pump_count_dict, total_pump_file)
    total_pump_file.close()

    print("Total # spikes: "+str(total_spike / total_row * 100)+"%")
    pass
