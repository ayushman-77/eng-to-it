import React, { useState, useEffect, useRef } from 'react';
import { submitTranslation, getJobStatus } from '../api';

export default function Translator({ token }) {
  const [sourceText, setSourceText] = useState('');
  const [targetText, setTargetText] = useState('');
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  
  const pollingRef = useRef(null);

  const startPolling = (jobId) => {
    if (pollingRef.current) clearInterval(pollingRef.current);
    
    pollingRef.current = setInterval(async () => {
      try {
        const job = await getJobStatus(token, jobId);
        setStatus(job.status);
        
        if (job.status === 'completed') {
          setTargetText(job.translated_text);
          setLoading(false);
          clearInterval(pollingRef.current);
        } else if (job.status === 'failed') {
          setError(job.error_message || 'Translation failed');
          setLoading(false);
          clearInterval(pollingRef.current);
        }
      } catch (err) {
        console.error(err);
      }
    }, 2000); // Poll every 2 seconds
  };

  useEffect(() => {
    return () => {
      if (pollingRef.current) clearInterval(pollingRef.current);
    };
  }, []);

  const handleTranslate = async () => {
    if (!sourceText.trim()) return;
    
    setLoading(true);
    setError('');
    setTargetText('');
    setStatus('enqueued');

    try {
      const data = await submitTranslation(token, sourceText);
      startPolling(data.job_id);
    } catch (err) {
      setError(err.message || 'Failed to submit translation');
      setLoading(false);
      setStatus(null);
    }
  };

  return (
    <div className="glass-panel">
      <h2>English to Italian</h2>
      
      <div className="grid-2">
        <div className="form-group">
          <label>Source Text (English)</label>
          <textarea 
            placeholder="Enter text to translate..."
            value={sourceText}
            onChange={e => setSourceText(e.target.value)}
          />
        </div>

        <div className="form-group">
          <label>Translation (Italian)</label>
          <div className="result-box">
            {targetText ? (
              <p style={{color: 'var(--text-main)', margin: 0}}>{targetText}</p>
            ) : status ? (
              <div style={{display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1rem'}}>
                <span className={`status-badge status-${status === 'enqueued' ? 'pending' : status}`}>
                  {status}
                </span>
                {loading && <div className="loader"></div>}
              </div>
            ) : (
              <p style={{margin: 0}}>Translation will appear here...</p>
            )}
          </div>
        </div>
      </div>

      {error && <div className="error-text" style={{marginBottom: '1rem'}}>{error}</div>}

      <div style={{display: 'flex', justifyContent: 'center', marginTop: '1rem'}}>
        <button onClick={handleTranslate} disabled={loading || !sourceText.trim()}>
          {loading ? 'Processing...' : 'Translate'}
        </button>
      </div>
    </div>
  );
}
