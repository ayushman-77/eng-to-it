import React, { useState } from 'react';
import { login, register } from '../api';

export default function Auth({ onLogin }) {
  const [isLogin, setIsLogin] = useState(true);
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      if (isLogin) {
        const data = await login(username, password);
        onLogin(data.access_token, username);
      } else {
        await register(username, email, password);
        // auto login after register
        const data = await login(username, password);
        onLogin(data.access_token, username);
      }
    } catch (err) {
      setError(err.message || 'Authentication failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <div className="auth-tabs">
        <button 
          className={`auth-tab ${isLogin ? 'active' : ''}`} 
          onClick={() => setIsLogin(true)}
        >
          Login
        </button>
        <button 
          className={`auth-tab ${!isLogin ? 'active' : ''}`} 
          onClick={() => setIsLogin(false)}
        >
          Register
        </button>
      </div>

      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label>Username</label>
          <input 
            type="text" 
            value={username} 
            onChange={e => setUsername(e.target.value)} 
            required 
          />
        </div>
        
        {!isLogin && (
          <div className="form-group">
            <label>Email</label>
            <input 
              type="email" 
              value={email} 
              onChange={e => setEmail(e.target.value)} 
              required 
            />
          </div>
        )}

        <div className="form-group">
          <label>Password</label>
          <input 
            type="password" 
            value={password} 
            onChange={e => setPassword(e.target.value)} 
            required 
            minLength={8}
          />
        </div>

        {error && <div className="error-text">{error}</div>}

        <button type="submit" disabled={loading} style={{width: '100%', marginTop: '1rem'}}>
          {loading ? <div className="loader"></div> : (isLogin ? 'Sign In' : 'Create Account')}
        </button>
      </form>
    </div>
  );
}
