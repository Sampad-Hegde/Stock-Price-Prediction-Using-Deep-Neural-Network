from requests import get, session
# from pprint import pprint
# from yahoofinscraper import YahooFinance
session = session()

head = {
    'user-agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/87.0.4280.88 Safari/537.36 "
}


def getSymbolData(keyword):
    keyword = keyword.upper()
    keyword = keyword.replace(' ', '%20')
    keyword = keyword.replace('-', '%20')
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={keyword}&quotesCount=6"
    webdata = get(url=url, headers=head)
    return webdata.json()['quotes'][0]


# symbol = getSymbolData('nifty')
# print(symbol)
#
# YF = YahooFinance(symbol['symbol'], result_range='10y', interval='1d')
# print(YF.to_csv('Nifty.csv'))
