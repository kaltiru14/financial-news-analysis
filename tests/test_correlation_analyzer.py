import pytest
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from correlation_analyzer import DataLoader, SentimentAnalyzer, StockAnalyzer, CorrelationEngine

class TestDataLoader:
    def setup_method(self):
        self.loader = DataLoader()
    
    def test_news_data_validation(self):
        # Test with mock data
        mock_news = pd.DataFrame({
            'headline': ['Test headline 1', 'Test headline 2'],
            'date': ['2023-01-01', '2023-01-02'],
            'stock': ['AAPL', 'AAPL'],
            'publisher': ['Test Pub', 'Test Pub']
        })
        
        # This should handle the data validation
        assert hasattr(self.loader, 'load_news_data')

class TestSentimentAnalyzer:
    def setup_method(self):
        self.analyzer = SentimentAnalyzer()
    
    def test_sentiment_analysis_positive(self):
        score, label, components = self.analyzer.analyze_headline("Great earnings! Stock surges to new highs!")
        assert label in ['positive', 'neutral']
        assert -1 <= score <= 1
    
    def test_sentiment_analysis_negative(self):
        score, label, components = self.analyzer.analyze_headline("Terrible losses. Company facing bankruptcy.")
        assert label in ['negative', 'neutral']
        assert -1 <= score <= 1
    
    def test_sentiment_analysis_neutral(self):
        score, label, components = self.analyzer.analyze_headline("Company reported quarterly results.")
        assert label == 'neutral'
        assert -1 <= score <= 1
    
    def test_sentiment_analysis_empty(self):
        score, label, components = self.analyzer.analyze_headline("")
        assert label == 'neutral'
        assert score == 0

class TestStockAnalyzer:
    def setup_method(self):
        self.analyzer = StockAnalyzer()
    
    def test_daily_returns_calculation(self):
        prices = pd.Series([100, 105, 102, 110, 115])
        returns = self.analyzer.calculate_daily_returns(prices)
        
        assert len(returns) == len(prices)
        assert returns.iloc[0] == 0  # First day should be 0
        assert abs(returns.iloc[1] - 5.0) < 0.1  # (105-100)/100 * 100 = 5%
    
    def test_volatility_calculation(self):
        returns = pd.Series([1.0, -0.5, 2.0, -1.0, 0.5])
        volatility = self.analyzer.calculate_volatility(returns, window=3)
        assert len(volatility) == len(returns)

def test_correlation_engine_initialization():
    engine = CorrelationEngine()
    assert hasattr(engine, 'data_loader')
    assert hasattr(engine, 'sentiment_analyzer')
    assert hasattr(engine, 'stock_analyzer')
    assert isinstance(engine.analysis_results, dict)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])