import { useState, useEffect } from 'react';

// For local testing, default to port 8000 (FastAPI).
// In production, you can replace this with your Render API URL.
const DEFAULT_API_BASE = 'http://localhost:8000';

const LOADING_STEPS = [
  'Detecting human face...',
  'Extracting facial landmark region...',
  'Normalizing input resolution to 256x256...',
  'Running classification through InceptionResnetV1...',
  'Backpropagating target layer gradients...',
  'Generating GradCAM localization overlay...',
];

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadingStepIndex, setLoadingStepIndex] = useState(0);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [apiUrl, setApiUrl] = useState(DEFAULT_API_BASE);

  // Set API URL from environment variable or query params if needed
  useEffect(() => {
    const queryParams = new URLSearchParams(window.location.search);
    const apiParam = queryParams.get('api_url');
    if (apiParam) {
      setApiUrl(apiParam);
    } else {
      // In production (Vercel), we could dynamically resolve Render base URL
      // But we will expose a config input or let it load from localStorage
      const savedUrl = localStorage.getItem('deepfake_api_url');
      if (savedUrl) setApiUrl(savedUrl);
    }
  }, []);

  const handleApiUrlChange = (e) => {
    const val = e.target.value;
    setApiUrl(val);
    localStorage.setItem('deepfake_api_url', val);
  };

  // Cycle loading steps
  useEffect(() => {
    let interval;
    if (loading) {
      interval = setInterval(() => {
        setLoadingStepIndex((prev) => (prev + 1) % LOADING_STEPS.length);
      }, 1500);
    } else {
      setLoadingStepIndex(0);
    }
    return () => clearInterval(interval);
  }, [loading]);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile.type.startsWith('image/')) {
        setFile(droppedFile);
        setPreview(URL.createObjectURL(droppedFile));
        setResult(null);
        setError(null);
      } else {
        setError('Please drop a valid image file (PNG, JPG, JPEG, WEBP).');
      }
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setResult(null);
      setError(null);
    }
  };

  const triggerUpload = () => {
    document.getElementById('file-upload').click();
  };

  const resetAll = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  const runDetection = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${apiUrl}/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server error (${response.status})`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      console.error(err);
      setError(err.message || 'Failed to connect to the backend server. Make sure the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <header className="app-header">
        <div className="header-container">
          <div className="logo">
            <span>🛡️</span> DEEPFAKE DETECTOR
          </div>
          <div className="status-badge">
            <div className="status-dot"></div>
            <span>System Active</span>
          </div>
        </div>
      </header>

      <main className="app-main">
        <section className="hero-section">
          <h1>Neural Face Integrity Verifier</h1>
          <p>
            Upload any face profile photo. Our double-check Inception-ResNet model analyses structural coherence and spotlights suspicious regions using GradCAM localization.
          </p>
          
          <div style={{ marginTop: '1.5rem', display: 'flex', justifyContent: 'center', gap: '0.5rem', alignItems: 'center' }}>
            <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Backend URL:</label>
            <input 
              type="text" 
              value={apiUrl} 
              onChange={handleApiUrlChange}
              style={{
                background: 'var(--bg-secondary)',
                border: '1px solid var(--border-color)',
                color: 'var(--accent-cyan)',
                padding: '0.3rem 0.6rem',
                borderRadius: '6px',
                width: '260px',
                fontSize: '0.8rem',
                fontFamily: 'var(--font-body)'
              }}
            />
          </div>
        </section>

        <section className="detector-container">
          {/* Upload Panel / Input */}
          <div className="upload-panel">
            {!preview ? (
              <div 
                className={`dropzone ${dragActive ? 'active' : ''}`}
                onDragEnter={handleDrag}
                onDragOver={handleDrag}
                onDragLeave={handleDrag}
                onDrop={handleDrop}
                onClick={triggerUpload}
              >
                <div className="upload-icon">📤</div>
                <h3>Drag & Drop Face Profile</h3>
                <p>or click to browse local storage</p>
                <p style={{ marginTop: '0.5rem', fontSize: '0.75rem' }}>Supports PNG, JPG, JPEG, WEBP</p>
                <input 
                  type="file" 
                  id="file-upload" 
                  className="file-input" 
                  accept="image/*" 
                  onChange={handleFileChange}
                />
              </div>
            ) : (
              <div className="preview-container">
                <img src={preview} alt="Input Face" className="img-preview" />
                <div className="action-bar">
                  <button className="btn btn-secondary" onClick={resetAll} disabled={loading}>
                    Clear
                  </button>
                  <button className="btn btn-primary" onClick={runDetection} disabled={loading}>
                    Analyze Image
                  </button>
                </div>
              </div>
            )}

            {error && (
              <div className="error-banner">
                ⚠️ {error}
              </div>
            )}
          </div>

          {/* Result / Output Panel */}
          {loading ? (
            <div className="loading-panel">
              <div className="cyber-spinner"></div>
              <h3>Analyzing Face Vectors</h3>
              <div className="loading-step">{LOADING_STEPS[loadingStepIndex]}</div>
            </div>
          ) : result ? (
            <div className="results-panel">
              <div className="result-header">
                <span className={`result-badge ${result.prediction}`}>
                  {result.prediction}
                </span>
                <h3 style={{ margin: '0.5rem 0 0', fontSize: '1.2rem', color: 'var(--text-secondary)' }}>
                  Classification Verdict
                </h3>
              </div>

              {result.explainability_image && (
                <div className="visualization-wrapper">
                  <div className="visualization-box">
                    <img 
                      src={result.explainability_image} 
                      alt="Explainability Heatmap" 
                      className="vis-img"
                    />
                    <div className="vis-label">
                      GradCAM Heatmap (red highlighted zones represent focus regions)
                    </div>
                  </div>
                </div>
              )}

              <div className="confidence-section">
                {/* Real probability */}
                <div className="confidence-bar-container">
                  <div className="bar-labels">
                    <span>Authentic Profile</span>
                    <span>{(result.confidences.real * 100).toFixed(2)}%</span>
                  </div>
                  <div className="bar-outer">
                    <div 
                      className="bar-inner real" 
                      style={{ width: `${result.confidences.real * 100}%` }}
                    ></div>
                  </div>
                </div>

                {/* Fake probability */}
                <div className="confidence-bar-container">
                  <div className="bar-labels">
                    <span>Deepfake Artificial Probability</span>
                    <span>{(result.confidences.fake * 100).toFixed(2)}%</span>
                  </div>
                  <div className="bar-outer">
                    <div 
                      className="bar-inner fake" 
                      style={{ width: `${result.confidences.fake * 100}%` }}
                    ></div>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="loading-panel" style={{ background: 'rgba(31, 40, 51, 0.1)' }}>
              <span style={{ fontSize: '3rem', marginBottom: '1.5rem', opacity: 0.6 }}>🤖</span>
              <h3 style={{ color: 'var(--text-muted)' }}>Ready for Diagnostics</h3>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', maxWidth: '300px' }}>
                Load an image on the left side and press "Analyze Image" to run models.
              </p>
            </div>
          )}
        </section>
      </main>

      <footer className="app-footer">
        🛡️ Deepfake Diagnostics Portal • Powered by PyTorch & FastAPI
      </footer>
    </>
  );
}

export default App;
