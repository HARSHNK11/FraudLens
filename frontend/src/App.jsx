import { useState, useEffect } from 'react';
import './index.css';
import './layout.css';
import TransactionForm from './components/TransactionForm';
import PredictionResult from './components/PredictionResult';
import ShapChart from './components/ShapChart';
import Dashboard from './components/Dashboard';
import { predictFraud, getModelInfo } from './api/fraudApi';

function Navbar() {
  return (
    <nav className="navbar">
      <div className="navbar-inner">
        <a href="#" className="navbar-brand">
          <div className="navbar-logo">🔍</div>
          <span className="navbar-name">Fraud<span>Lens</span></span>
        </a>
        <div className="navbar-status">
          <span className="pulse-dot green" />
          <span>ML-Powered Fraud Detection</span>
        </div>
      </div>
    </nav>
  );
}

function Hero({ onAnalyze }) {
  return (
    <section className="hero">
      <div className="hero-eyebrow">
        <span>🛡️</span> Real-Time Credit Card Fraud Detection
      </div>
      <h1 className="hero-title">
        Detect fraud with{' '}
        <span className="gradient-text">machine precision</span>
      </h1>
      <p className="hero-subtitle">
        FraudLens uses ensemble machine learning trained on 284,807 real transactions
        with Logistic Regression, Random Forest, and XGBoost models.
      </p>
      <div className="hero-cta">
        <button className="btn btn-primary" onClick={onAnalyze} id="hero-analyze-btn">
          🔍 Analyze a Transaction
        </button>
      </div>
    </section>
  );
}

export default function App() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('analyzer'); // 'analyzer' | 'dashboard'
  const [requiredFeatures, setRequiredFeatures] = useState(null);

  useEffect(() => {
    async function fetchInfo() {
      try {
        const info = await getModelInfo();
        if (info && info.features) {
          setRequiredFeatures(info.features);
        }
      } catch (err) {
        console.error('Failed to fetch model info', err);
      }
    }
    fetchInfo();
  }, []);

  useEffect(() => {
    document.title = activeTab === 'analyzer' ? 'Analyzer | FraudLens' : 'Dashboard | FraudLens';
  }, [activeTab]);

  const handleAnalyze = () => {
    setActiveTab('analyzer');
    setTimeout(() => {
      document.getElementById('analyzer-section')?.scrollIntoView({ behavior: 'smooth' });
    }, 80);
  };

  const handleSubmit = async (transaction) => {
    setLoading(true);
    setError(null);
    try {
      const data = await predictFraud(transaction);
      setResult(data);
      setTimeout(() => {
        document.getElementById('result-section')?.scrollIntoView({ behavior: 'smooth' });
      }, 100);
    } catch (err) {
      const msg = err.response?.data?.detail || err.message || 'Prediction failed. Is the backend running?';
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <Navbar />
      <Hero onAnalyze={handleAnalyze} />

      <div className="container">
        {/* Tab Navigation */}
        <div className="tabs" style={{ marginBottom: 32, maxWidth: 400 }}>
          <button
            id="tab-analyzer"
            className={`tab-btn ${activeTab === 'analyzer' ? 'active' : ''}`}
            onClick={() => setActiveTab('analyzer')}
          >
            🔍 Analyzer
          </button>
          <button
            id="tab-dashboard"
            className={`tab-btn ${activeTab === 'dashboard' ? 'active' : ''}`}
            onClick={() => setActiveTab('dashboard')}
          >
            📊 Dashboard
          </button>
        </div>

        {/* Analyzer Tab */}
        {activeTab === 'analyzer' && (
          <div id="analyzer-section">
            <div className="main-layout">
              {/* Left: Form */}
              <div className="glass-card panel">
                <TransactionForm onSubmit={handleSubmit} loading={loading} requiredFeatures={requiredFeatures} />
              </div>

              {/* Right: Results */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
                {error && (
                  <div className="glass-card panel" style={{ borderColor: 'rgba(239,68,68,0.3)' }}>
                    <div style={{ color: '#f87171', fontWeight: 600, marginBottom: 6 }}>⚠ Error</div>
                    <div style={{ fontSize: '0.88rem', color: 'var(--text-secondary)' }}>{error}</div>
                  </div>
                )}

                {!result && !error && !loading && (
                  <div className="glass-card panel" style={{ textAlign: 'center', padding: '60px 24px' }}>
                    <div style={{ fontSize: '3rem', marginBottom: 16 }}>🔍</div>
                    <div style={{ fontWeight: 700, fontSize: '1.1rem', marginBottom: 8 }}>
                      No analysis yet
                    </div>
                    <div style={{ color: 'var(--text-muted)', fontSize: '0.88rem' }}>
                      Fill in the transaction details and click Analyze, or use a sample transaction.
                    </div>
                  </div>
                )}

                {loading && (
                  <div className="glass-card panel" style={{ textAlign: 'center', padding: '60px 24px' }}>
                    <div className="spinner" style={{ margin: '0 auto 20px', width: '40px', height: '40px', borderTopColor: '#3b82f6' }}></div>
                    <div style={{ fontWeight: 700, fontSize: '1.1rem', marginBottom: 8 }}>
                      Running Ensemble Models...
                    </div>
                    <div style={{ color: 'var(--text-muted)', fontSize: '0.88rem' }}>
                      Computing SHAP values and evaluating risk...
                    </div>
                  </div>
                )}

                {result && !loading && (
                  <>
                    <div id="result-section">
                      <PredictionResult result={result} />
                    </div>
                    {result.top_features && result.top_features.length > 0 && (
                      <ShapChart topFeatures={result.top_features} />
                    )}
                  </>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Dashboard Tab */}
        {activeTab === 'dashboard' && (
          <div>
            <Dashboard />
          </div>
        )}
      </div>

      {/* Footer */}
      <footer style={{
        textAlign: 'center', padding: '32px 24px',
        fontSize: '0.8rem', color: 'var(--text-muted)',
        borderTop: '1px solid var(--border)', marginTop: 40
      }}>
        FraudLens — ML-Powered Fraud Detection &nbsp;·&nbsp;
        Trained on <strong style={{ color: 'var(--blue-400)' }}>284,807</strong> real transactions
      </footer>
    </div>
  );
}
