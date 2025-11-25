"""
Task 3: Correlation Analysis between News Sentiment and Stock Movements
Object-Oriented Implementation with Comprehensive Error Handling
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
from scipy.stats import pearsonr, spearmanr
import nltk
import warnings
import os
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    print("📥 Downloading NLTK VADER lexicon...")
    nltk.download('vader_lexicon')

class DataLoader:
    """Handles loading and validation of news and stock data"""
    
    def __init__(self):
        self.news_data = None
        self.stock_data = {}
    
    def load_news_data(self, file_path):
        """Load and validate news data"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"News file not found: {file_path}")
            
            self.news_data = pd.read_csv(file_path)
            print(f"✅ News data loaded: {len(self.news_data)} articles")
            
            # Basic validation
            required_columns = ['headline', 'date', 'stock', 'publisher']
            missing_columns = [col for col in required_columns if col not in self.news_data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Convert dates
            self.news_data['date'] = pd.to_datetime(self.news_data['date'], errors='coerce')
            invalid_dates = self.news_data['date'].isna().sum()
            if invalid_dates > 0:
                print(f"⚠️ {invalid_dates} articles have invalid dates and will be excluded")
                self.news_data = self.news_data.dropna(subset=['date'])
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading news data: {e}")
            return False
    
    def load_stock_data(self, symbol, file_path):
        """Load and validate stock data for a symbol"""
        try:
            if not os.path.exists(file_path):
                print(f"⚠️ Stock file not found: {file_path}")
                return False
            
            stock_df = pd.read_csv(file_path)
            print(f"✅ Stock data loaded for {symbol}: {len(stock_df)} records")
            
            # Find date column
            date_col = None
            for col in ['Date', 'date', 'timestamp']:
                if col in stock_df.columns:
                    date_col = col
                    break
            
            if not date_col:
                raise ValueError("No date column found in stock data")
            
            # Convert dates and set index
            stock_df[date_col] = pd.to_datetime(stock_df[date_col], errors='coerce')
            stock_df = stock_df.dropna(subset=[date_col])
            stock_df.set_index(date_col, inplace=True)
            
            # Validate required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in stock_df.columns]
            if missing_columns:
                print(f"⚠️ Missing columns in {symbol}: {missing_columns}")
                return False
            
            self.stock_data[symbol] = stock_df
            return True
            
        except Exception as e:
            print(f"❌ Error loading stock data for {symbol}: {e}")
            return False
    
    def get_available_stocks(self):
        """Get list of stocks with available data"""
        return list(self.stock_data.keys())
    
    def analyze_news_coverage(self, target_stocks=None):
        """Analyze news coverage for target stocks"""
        if self.news_data is None:
            print("❌ No news data loaded")
            return {}
        
        if target_stocks is None:
            target_stocks = self.news_data['stock'].unique()
        
        coverage = {}
        for stock in target_stocks:
            stock_news = self.news_data[self.news_data['stock'] == stock]
            coverage[stock] = {
                'article_count': len(stock_news),
                'date_range': (stock_news['date'].min(), stock_news['date'].max()) if len(stock_news) > 0 else (None, None),
                'publishers': stock_news['publisher'].nunique()
            }
        
        return coverage

class SentimentAnalyzer:
    """Handles sentiment analysis of financial news"""
    
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
    
    def analyze_headline(self, headline):
        """Analyze sentiment of a single headline"""
        if pd.isna(headline) or not isinstance(headline, str) or headline.strip() == '':
            return 0.0, 'neutral', {'textblob': 0, 'vader': 0}
        
        try:
            # TextBlob sentiment
            blob = TextBlob(headline)
            tb_sentiment = blob.sentiment.polarity
            
            # VADER sentiment
            vader_scores = self.sia.polarity_scores(headline)
            vader_sentiment = vader_scores['compound']
            
            # Combined score (weighted average)
            combined_sentiment = (tb_sentiment * 0.4 + vader_sentiment * 0.6)
            
            # Classification
            if combined_sentiment > 0.1:
                sentiment_label = 'positive'
            elif combined_sentiment < -0.1:
                sentiment_label = 'negative'
            else:
                sentiment_label = 'neutral'
            
            return combined_sentiment, sentiment_label, {'textblob': tb_sentiment, 'vader': vader_sentiment}
            
        except Exception as e:
            print(f"⚠️ Sentiment analysis failed for headline: {headline[:50]}... Error: {e}")
            return 0.0, 'neutral', {'textblob': 0, 'vader': 0}
    
    def analyze_articles(self, articles_df):
        """Analyze sentiment for multiple articles"""
        print("🧠 Performing sentiment analysis...")
        
        results = []
        for idx, article in articles_df.iterrows():
            sentiment_score, sentiment_label, component_scores = self.analyze_headline(article['headline'])
            
            results.append({
                'date': article['date'],
                'headline': article['headline'],
                'stock': article['stock'],
                'publisher': article['publisher'],
                'sentiment_score': sentiment_score,
                'sentiment_label': sentiment_label,
                'textblob_score': component_scores['textblob'],
                'vader_score': component_scores['vader']
            })
        
        return pd.DataFrame(results)

class StockAnalyzer:
    """Handles stock price analysis and return calculations"""
    
    def calculate_daily_returns(self, prices_series):
        """Calculate daily percentage returns"""
        returns = prices_series.pct_change() * 100
        # Set first value to 0 (no previous day)
        if len(returns) > 0:
            returns.iloc[0] = 0
        return returns
    
    def calculate_volatility(self, returns_series, window=20):
        """Calculate rolling volatility"""
        return returns_series.rolling(window=window).std()
    
    def analyze_stock_performance(self, stock_df):
        """Comprehensive stock performance analysis"""
        close_prices = stock_df['Close']
        daily_returns = self.calculate_daily_returns(close_prices)
        
        analysis = {
            'close_prices': close_prices,
            'daily_returns': daily_returns,
            'cumulative_returns': (1 + daily_returns/100).cumprod() - 1,
            'volatility_20d': self.calculate_volatility(daily_returns),
            'total_return': (close_prices.iloc[-1] - close_prices.iloc[0]) / close_prices.iloc[0] * 100 if len(close_prices) > 0 else 0
        }
        
        return analysis

class CorrelationEngine:
    """Main engine for correlation analysis between news sentiment and stock returns"""
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.stock_analyzer = StockAnalyzer()
        self.analysis_results = {}
    
    def align_data_by_date(self, news_df, stock_df, stock_symbol):
        """Align news articles with stock trading dates"""
        print(f"🔄 Aligning data for {stock_symbol}...")
        
        # Extract dates from both datasets
        news_dates = news_df['date'].dt.date.unique()
        stock_dates = stock_df.index.date
        
        # Find overlapping dates
        overlapping_dates = set(news_dates) & set(stock_dates)
        
        aligned_data = []
        for date in sorted(overlapping_dates):
            # Get stock data for this date
            date_stock_data = stock_df[stock_df.index.date == date]
            if len(date_stock_data) == 0:
                continue
            
            # Get news for this date
            date_news = news_df[news_df['date'].dt.date == date]
            
            aligned_data.append({
                'date': date,
                'stock_data': date_stock_data.iloc[0],
                'news_articles': date_news,
                'news_count': len(date_news)
            })
        
        print(f"📅 Found {len(aligned_data)} days with both news and stock data")
        return aligned_data
    
    def perform_correlation_analysis(self, news_path, stock_symbols):
        """Complete correlation analysis pipeline"""
        print("🚀 Starting comprehensive correlation analysis...")
        
        # Load data
        if not self.data_loader.load_news_data(news_path):
            return {}
        
        # Load stock data for all symbols
        successful_loads = []
        for symbol in stock_symbols:
            stock_path = f"../data/{symbol}.csv"
            if self.data_loader.load_stock_data(symbol, stock_path):
                successful_loads.append(symbol)
        
        if not successful_loads:
            print("❌ No stock data successfully loaded")
            return {}
        
        print(f"✅ Successfully loaded data for {len(successful_loads)} stocks")
        
        # Analyze news coverage
        coverage = self.data_loader.analyze_news_coverage(successful_loads)
        
        # Perform analysis for each stock
        for symbol in successful_loads:
            try:
                print(f"\n{'='*60}")
                print(f"🔍 Analyzing: {symbol}")
                print(f"{'='*60}")
                
                # Get stock-specific news
                stock_news = self.data_loader.news_data[
                    self.data_loader.news_data['stock'] == symbol
                ].copy()
                
                if len(stock_news) == 0:
                    print(f"⚠️ No news articles found for {symbol}")
                    continue
                
                # Align dates
                aligned_data = self.align_data_by_date(
                    stock_news, 
                    self.data_loader.stock_data[symbol], 
                    symbol
                )
                
                if len(aligned_data) < 2:
                    print(f"⚠️ Insufficient overlapping data for {symbol} (need at least 2 days)")
                    continue
                
                # Perform sentiment analysis
                sentiment_results = self.sentiment_analyzer.analyze_articles(stock_news)
                
                # Calculate daily aggregated sentiment
                daily_sentiment = sentiment_results.groupby(
                    sentiment_results['date'].dt.date
                )['sentiment_score'].agg(['mean', 'count']).reset_index()
                daily_sentiment.columns = ['date', 'avg_sentiment', 'article_count']
                
                # Analyze stock performance
                stock_analysis = self.stock_analyzer.analyze_stock_performance(
                    self.data_loader.stock_data[symbol]
                )
                
                # Prepare correlation data
                correlation_data = []
                for day_data in aligned_data:
                    date = day_data['date']
                    
                    # Find sentiment for this date
                    date_sentiment = daily_sentiment[daily_sentiment['date'] == date]
                    avg_sentiment = date_sentiment['avg_sentiment'].iloc[0] if len(date_sentiment) > 0 else 0
                    
                    # Find stock return for this date
                    date_stock_returns = stock_analysis['daily_returns'][
                        stock_analysis['daily_returns'].index.date == date
                    ]
                    daily_return = date_stock_returns.iloc[0] if len(date_stock_returns) > 0 else 0
                    
                    correlation_data.append({
                        'date': date,
                        'avg_sentiment': avg_sentiment,
                        'daily_return': daily_return,
                        'news_count': day_data['news_count']
                    })
                
                corr_df = pd.DataFrame(correlation_data)
                
                # Calculate correlations
                if len(corr_df) >= 2:
                    pearson_corr, pearson_p = pearsonr(corr_df['avg_sentiment'], corr_df['daily_return'])
                    spearman_corr, spearman_p = spearmanr(corr_df['avg_sentiment'], corr_df['daily_return'])
                else:
                    pearson_corr = pearson_p = spearman_corr = spearman_p = 0
                
                # Store results
                self.analysis_results[symbol] = {
                    'aligned_data': aligned_data,
                    'sentiment_results': sentiment_results,
                    'correlation_data': corr_df,
                    'correlation_results': (pearson_corr, pearson_p, spearman_corr, spearman_p),
                    'stock_analysis': stock_analysis,
                    'summary': {
                        'total_trading_days': len(self.data_loader.stock_data[symbol]),
                        'days_with_news': len(corr_df),
                        'total_articles': len(sentiment_results),
                        'news_coverage_pct': (len(corr_df) / len(self.data_loader.stock_data[symbol])) * 100,
                        'avg_sentiment': sentiment_results['sentiment_score'].mean(),
                        'avg_daily_return': corr_df['daily_return'].mean() if len(corr_df) > 0 else 0,
                        'sentiment_distribution': sentiment_results['sentiment_label'].value_counts().to_dict()
                    }
                }
                
                # Generate report
                self._generate_stock_report(symbol)
                
            except Exception as e:
                print(f"❌ Analysis failed for {symbol}: {e}")
                continue
        
        return self.analysis_results
    
    def _generate_stock_report(self, symbol):
        """Generate analysis report for a stock"""
        if symbol not in self.analysis_results:
            return
        
        results = self.analysis_results[symbol]
        summary = results['summary']
        pearson_corr, pearson_p, spearman_corr, spearman_p = results['correlation_results']
        
        print(f"\n📊 CORRELATION ANALYSIS REPORT: {symbol}")
        print("="*50)
        print(f"📈 DATA OVERVIEW:")
        print(f"  • Trading Days: {summary['total_trading_days']}")
        print(f"  • Days with News: {summary['days_with_news']}")
        print(f"  • News Coverage: {summary['news_coverage_pct']:.1f}%")
        print(f"  • Articles Analyzed: {summary['total_articles']}")
        
        print(f"\n🎯 CORRELATION RESULTS:")
        print(f"  • Pearson: {pearson_corr:.4f} (p-value: {pearson_p:.4f})")
        print(f"  • Spearman: {spearman_corr:.4f} (p-value: {spearman_p:.4f})")
        print(f"  • Statistical Significance: {'YES' if pearson_p < 0.05 else 'NO'}")
        
        print(f"\n📋 SENTIMENT SUMMARY:")
        print(f"  • Average Sentiment: {summary['avg_sentiment']:.4f}")
        print(f"  • Sentiment Distribution: {summary['sentiment_distribution']}")
        print(f"  • Average Daily Return: {summary['avg_daily_return']:.3f}%")
        
        print(f"\n💡 BUSINESS IMPLICATIONS:")
        if abs(pearson_corr) > 0.5 and pearson_p < 0.05:
            direction = "positive" if pearson_corr > 0 else "negative"
            print(f"  • Strong {direction} correlation detected - sentiment has predictive power")
            print(f"  • Consider incorporating sentiment analysis in trading strategies")
        elif abs(pearson_corr) > 0.3 and pearson_p < 0.05:
            direction = "positive" if pearson_corr > 0 else "negative"
            print(f"  • Moderate {direction} correlation detected")
            print(f"  • Sentiment may provide supplementary signals")
        else:
            print(f"  • Weak or insignificant correlation observed")
            print(f"  • News sentiment alone may not be reliable for predictions")
        
        print(f"\n✅ ANALYSIS COMPLETED FOR {symbol}")
    
    def generate_comparative_analysis(self):
        """Generate comparative analysis across all stocks"""
        if not self.analysis_results:
            print("❌ No analysis results available")
            return
        
        print(f"\n🏆 COMPARATIVE ANALYSIS ACROSS ALL STOCKS")
        print("="*50)
        
        comparison_data = []
        for symbol, results in self.analysis_results.items():
            pearson_corr, pearson_p, spearman_corr, spearman_p = results['correlation_results']
            summary = results['summary']
            
            comparison_data.append({
                'Symbol': symbol,
                'Pearson_Correlation': pearson_corr,
                'Pearson_p_value': pearson_p,
                'Spearman_Correlation': spearman_corr,
                'Days_with_News': summary['days_with_news'],
                'News_Coverage_Pct': summary['news_coverage_pct'],
                'Total_Articles': summary['total_articles'],
                'Avg_Sentiment': summary['avg_sentiment'],
                'Statistical_Significance': pearson_p < 0.05
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        print("\n📊 CORRELATION COMPARISON:")
        print(comparison_df.round(4).to_string(index=False))
        
        # Find strongest valid correlations
        valid_correlations = comparison_df[comparison_df['Days_with_News'] >= 3]
        
        if len(valid_correlations) > 0:
            strongest_pearson = valid_correlations.loc[valid_correlations['Pearson_Correlation'].abs().idxmax()]
            strongest_spearman = valid_correlations.loc[valid_correlations['Spearman_Correlation'].abs().idxmax()]
            
            print(f"\n🎯 STRONGEST CORRELATIONS:")
            print(f"  Pearson: {strongest_pearson['Symbol']} (r = {strongest_pearson['Pearson_Correlation']:.4f})")
            print(f"  Spearman: {strongest_spearman['Symbol']} (r = {strongest_spearman['Spearman_Correlation']:.4f})")
        
        return comparison_df
    
    def create_visualizations(self, symbol):
        """Create visualizations for a specific stock"""
        if symbol not in self.analysis_results:
            print(f"❌ No analysis results for {symbol}")
            return
        
        results = self.analysis_results[symbol]
        corr_df = results['correlation_data']
        sentiment_df = results['sentiment_results']
        
        if len(corr_df) == 0:
            print(f"⚠️ No correlation data for {symbol}")
            return
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'News Sentiment vs Stock Returns: {symbol}', fontsize=16, fontweight='bold')
        
        # Plot 1: Scatter plot
        axes[0,0].scatter(corr_df['avg_sentiment'], corr_df['daily_return'], alpha=0.6, color='blue')
        if len(corr_df) >= 2:
            z = np.polyfit(corr_df['avg_sentiment'], corr_df['daily_return'], 1)
            p = np.poly1d(z)
            axes[0,0].plot(corr_df['avg_sentiment'], p(corr_df['avg_sentiment']), "r--", alpha=0.8)
        axes[0,0].set_xlabel('Average Sentiment Score')
        axes[0,0].set_ylabel('Daily Return (%)')
        axes[0,0].set_title('Sentiment vs Returns Correlation')
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[0,0].axvline(x=0, color='black', linestyle='-', alpha=0.5)
        
        # Plot 2: Sentiment distribution
        sentiment_counts = sentiment_df['sentiment_label'].value_counts()
        colors = ['green', 'gray', 'red']
        axes[0,1].pie(sentiment_counts.values, labels=sentiment_counts.index, 
                     autopct='%1.1f%%', colors=colors, startangle=90)
        axes[0,1].set_title('Sentiment Distribution')
        
        # Plot 3: Time series
        if len(corr_df) > 1:
            dates = pd.to_datetime(corr_df['date'])
            axes[1,0].plot(dates, corr_df['avg_sentiment'], label='Avg Sentiment', 
                          color='orange', linewidth=2)
            axes[1,0].set_ylabel('Sentiment Score', color='orange')
            axes[1,0].tick_params(axis='y', labelcolor='orange')
            
            ax2 = axes[1,0].twinx()
            ax2.bar(dates, corr_df['daily_return'], alpha=0.5, color='blue', label='Daily Returns')
            ax2.set_ylabel('Daily Return (%)', color='blue')
            ax2.tick_params(axis='y', labelcolor='blue')
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
            lines1, labels1 = axes[1,0].get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            axes[1,0].legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            axes[1,0].set_title('Sentiment & Returns Over Time')
        
        # Plot 4: News volume vs returns
        axes[1,1].scatter(corr_df['news_count'], corr_df['daily_return'], alpha=0.6, color='purple')
        axes[1,1].set_xlabel('Number of News Articles')
        axes[1,1].set_ylabel('Daily Return (%)')
        axes[1,1].set_title('News Volume vs Returns')
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(f'../images/task3_{symbol}_correlation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Visualization saved: task3_{symbol}_correlation.png")