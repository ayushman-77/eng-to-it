import React, { useState, useEffect } from 'react';
import { getHistory } from '../api';

export default function History({ token }) {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const data = await getHistory(token);
        setHistory(data.translations);
      } catch (err) {
        setError(err.message || 'Failed to load history');
      } finally {
        setLoading(false);
      }
    };
    fetchHistory();
  }, [token]);

  if (loading) return <div className="glass-panel" style={{textAlign: 'center'}}><div className="loader" style={{margin: '0 auto'}}></div></div>;
  if (error) return <div className="glass-panel error-text">{error}</div>;

  return (
    <div className="glass-panel">
      <h2>Translation History</h2>
      
      {history.length === 0 ? (
        <p style={{textAlign: 'center', marginTop: '2rem'}}>No translations yet.</p>
      ) : (
        <div className="history-list">
          {history.map((job) => (
            <div key={job.job_id} className="history-item">
              <div style={{overflow: 'hidden'}}>
                <div className="text-preview" style={{fontWeight: 600, marginBottom: '0.25rem'}}>
                  EN: {job.source_text}
                </div>
                <div className="text-preview" style={{color: 'var(--text-muted)'}}>
                  IT: {job.translated_text || '...'}
                </div>
                <div className="date">
                  {new Date(job.created_at).toLocaleString()}
                </div>
              </div>
              <div>
                <span className={`status-badge status-${job.status}`}>
                  {job.status}
                </span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
