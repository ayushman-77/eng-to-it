const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';

export const register = async (username, email, password) => {
    const res = await fetch(`${API_URL}/auth/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, email, password })
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
};

export const login = async (username, password) => {
    const res = await fetch(`${API_URL}/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password })
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
};

export const submitTranslation = async (token, source_text) => {
    const res = await fetch(`${API_URL}/translate/`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ source_text })
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
};

export const getJobStatus = async (token, job_id) => {
    const res = await fetch(`${API_URL}/translate/${job_id}`, {
        headers: { 'Authorization': `Bearer ${token}` }
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
};

export const getHistory = async (token) => {
    const res = await fetch(`${API_URL}/translate/history/me`, {
        headers: { 'Authorization': `Bearer ${token}` }
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
};
