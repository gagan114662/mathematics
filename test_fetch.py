import yfinance as yf
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_fetch():
    try:
        logger.info("Attempting to fetch AAPL data...")
        stock = yf.Ticker("AAPL")
        df = stock.history(period="1mo")
        logger.info(f"Successfully fetched data: {len(df)} rows")
        print(df.head())
    except Exception as e:
        logger.error(f"Error fetching data: {str(e)}")

if __name__ == "__main__":
    test_fetch()
