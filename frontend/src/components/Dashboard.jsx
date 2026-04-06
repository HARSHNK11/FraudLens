import { useState, useEffect } from 'react';
import { getModelInfo, getModelLog, getAllModelsInfo } from '../api/fraudApi';
import './dashboard.css';

const PLOT_NAMES = [
  { key: 'class_distribution',  label: 'Class Distribution',   icon: '📊' },
  { key: 'metrics_comparison',  label: 'SMOTE Comparison',      icon: '📈' },
  { key: 'pr_curve',            label: 'Precision-Recall Curve', icon: '🎯' },
  { key: 'roc_curve',           label: 'ROC Curve',             icon: '〽' },
  { key: 'feature_importance',  label: 'Feature Importance',    icon: '🏆' },
  { key: 'confusion_matrix',    label: 'Confusion Matrix',      icon: '🔢' },
  { key: 'shap_summary',        label: 'SHAP Summary',          icon: '🔍' },
];

const METRIC_CONFIG = [
  { key: 'pr_auc',    label: 'PR-AUC',    desc: 'Primary metric for imbalanced data', color: '#3b82f6' },
  { key: 'roc_auc',  label: 'ROC-AUC',   desc: 'Area under ROC curve',              color: '#8b5cf6' },
  { key: 'f1',       label: 'F1 Score',   desc: 'Harmonic mean of precision/recall', color: '#10b981' },
  { key: 'precision',label: 'Precision',  desc: 'True positives / predicted positives', color: '#f59e0b' },
  { key: 'recall',   label: 'Recall',     desc: 'True positives / actual positives', color: '#ef4444' },
];

export default function Dashboard() {
  const [modelInfo, setModelInfo] = useState(null);
  const [allModels, setAllModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState(null);
  const [modelLog, setModelLog] = useState('');
  const [activeViz, setActiveViz] = useState(0);
  const [imgError, setImgError] = useState({});

  useEffect(() => {
    getModelInfo()
      .then(info => {
        setModelInfo(info);
        setSelectedModel(info);
      })
      .catch(() => setModelInfo(null));

    getAllModelsInfo()
      .then(setAllModels)
      .catch(() => setAllModels([]));

    getModelLog()
      .then(data => setModelLog(data.log))
      .catch(() => setModelLog('Failed to load training log.'));
  }, []);

  const handleImgError = (key) => setImgError(prev => ({ ...prev, [key]: true }));

  return (
    <div className="dashboard">
      {/* Model Info Banner */}
      {modelInfo && selectedModel && (
        <div className="model-banner glass-card">
          <div className="model-banner-info">
            <div className="model-banner-icon">🤖</div>
            <div>
              <div className="model-dropdown-container">
                <select 
                  className="model-select"
                  value={selectedModel.model_name}
                  onChange={(e) => {
                    const sel = e.target.value;
                    if (sel === modelInfo.model_name) {
                      setSelectedModel(modelInfo);
                    } else {
                      const found = allModels.find(m => m.model_name === sel);
                      if (found) setSelectedModel(found);
                    }
                  }}
                >
                  <option value={modelInfo.model_name}>{modelInfo.model_name} (Best)</option>
                  {allModels
                    .filter(m => m.model_name !== modelInfo.model_name)
                    .map(m => (
                      <option key={m.model_name} value={m.model_name}>
                        {m.model_name}
                      </option>
                  ))}
                </select>
                <div className="model-dropdown-icon">▼</div>
              </div>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: 2 }}>
                Threshold: <span className="mono" style={{ color: 'var(--blue-400)' }}>
                  {selectedModel.threshold ? `${(selectedModel.threshold * 100).toFixed(1)}%` : 'Default (50.0%)'}
                </span>
              </div>
            </div>
          </div>
          <span className="badge badge-legit">
            <span className="pulse-dot green" style={{ width: 7, height: 7 }} />
            {selectedModel.model_name === modelInfo.model_name ? 'Model Online' : 'Metrics Loaded'}
          </span>
        </div>
      )}

      {/* Metrics Grid */}
      {selectedModel?.metrics && (
        <div>
          <div className="section-title">Model Performance</div>
          <div className="metrics-grid">
            {METRIC_CONFIG.map(({ key, label, desc, color }) => {
              const val = selectedModel.metrics[key];
              if (val === undefined) return null;
              return (
                <div key={key} className="metric-card" style={{ '--m-color': color }}>
                  <div className="metric-label">{label}</div>
                  <div className="metric-value" style={{ color }}>{(val * 100).toFixed(2)}%</div>
                  <div className="metric-sub">{desc}</div>
                  <div className="metric-bar" style={{ marginTop: 12 }}>
                    <div className="progress-bar">
                      <div className="progress-fill"
                        style={{ width: `${val * 100}%`, background: color }} />
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {!modelInfo && (
        <div className="model-offline glass-card">
          <span style={{ fontSize: '2rem' }}>⚠️</span>
          <div>
            <div style={{ fontWeight: 700, color: 'var(--amber-400)' }}>Backend Offline</div>
            <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginTop: 4 }}>
              Start the API with: <code className="mono" style={{ color: 'var(--blue-400)' }}>
                uvicorn backend.main:app --reload
              </code>
            </div>
          </div>
        </div>
      )}

      {/* Visualization Gallery */}
      <div style={{ marginTop: 32 }}>
        <div className="section-title">Visualizations</div>
        <div className="viz-tabs">
          {PLOT_NAMES.map((p, i) => (
            <button
              key={p.key}
              className={`viz-tab-btn ${activeViz === i ? 'active' : ''}`}
              onClick={() => setActiveViz(i)}
            >
              <span>{p.icon}</span> {p.label}
            </button>
          ))}
        </div>

        <div className="viz-display glass-card">
          {!imgError[PLOT_NAMES[activeViz].key] ? (
            <img
              key={PLOT_NAMES[activeViz].key}
              src={`http://localhost:8000/plots/${PLOT_NAMES[activeViz].key}.png`}
              alt={PLOT_NAMES[activeViz].label}
              className="viz-img-full"
              onError={() => handleImgError(PLOT_NAMES[activeViz].key)}
            />
          ) : (
            <div className="viz-empty">
              <div style={{ fontSize: '2.5rem', marginBottom: 12 }}>📊</div>
              <div style={{ fontWeight: 600, color: 'var(--text-secondary)' }}>
                {PLOT_NAMES[activeViz].label}
              </div>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: 8 }}>
                Run <code className="mono" style={{ color: 'var(--blue-400)' }}>python ml/train.py</code> to generate this visualization.
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Training Log Component */}
      <div style={{ marginTop: 32 }}>
        <div className="section-title">Training Output</div>
        <div className="glass-card" style={{ padding: 16 }}>
          <pre style={{ 
            background: 'rgba(15, 17, 23, 0.6)', 
            padding: '16px', 
            borderRadius: '8px', 
            border: '1px solid var(--border)',
            color: '#a0aec0',
            fontFamily: 'Consolas, Monaco, "Courier New", monospace',
            fontSize: '0.85rem',
            overflowX: 'auto',
            whiteSpace: 'pre-wrap',
            maxHeight: '600px',
            overflowY: 'auto'
          }}>
            {modelLog || 'Fetching log...'}
          </pre>
        </div>
      </div>
    </div>
  );
}
