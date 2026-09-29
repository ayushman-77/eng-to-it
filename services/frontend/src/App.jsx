import React, { useState, useEffect } from 'react'
import Auth from './components/Auth'
import Translator from './components/Translator'
import History from './components/History'
import './index.css'

function App() {
  const [token, setToken] = useState(localStorage.getItem('token'))
  const [username, setUsername] = useState(localStorage.getItem('username'))
  const [activeTab, setActiveTab] = useState('translate') // translate or history

  const handleLogin = (newToken, user) => {
    localStorage.setItem('token', newToken)
    localStorage.setItem('username', user)
    setToken(newToken)
    setUsername(user)
  }

  const handleLogout = () => {
    localStorage.removeItem('token')
    localStorage.removeItem('username')
    setToken(null)
    setUsername(null)
  }

  if (!token) {
    return (
      <div className="auth-container solid-panel">
        <h1>English to Italian Translator</h1>
        <p style={{textAlign: 'center', marginBottom: '2rem'}}>High-contrast, clean UI for Neural Machine Translation.</p>
        <Auth onLogin={handleLogin} />
      </div>
    )
  }

  return (
    <>
      <header className="app-header solid-panel" style={{padding: '1.5rem 2rem', marginBottom: '2rem'}}>
        <h2 style={{margin: 0, color: 'var(--primary)'}}>
          English to Italian Translator
        </h2>
        <div className="user-controls">
          <span className="username-badge">{username}</span>
          <button className="btn-secondary" style={{padding: '0.4rem 1rem'}} onClick={handleLogout}>Logout</button>
        </div>
      </header>

      <div className="auth-tabs" style={{maxWidth: '400px', margin: '0 auto 2rem'}}>
        <button 
          className={`auth-tab ${activeTab === 'translate' ? 'active' : ''}`}
          onClick={() => setActiveTab('translate')}
        >
          Translator
        </button>
        <button 
          className={`auth-tab ${activeTab === 'history' ? 'active' : ''}`}
          onClick={() => setActiveTab('history')}
        >
          History
        </button>
      </div>

      <main>
        {activeTab === 'translate' ? (
          <Translator token={token} />
        ) : (
          <History token={token} />
        )}
      </main>
    </>
  )
}

export default App
