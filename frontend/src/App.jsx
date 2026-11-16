import React, { useState } from 'react';
import LiveDetection from './components/LiveDetection';
import TextToSpeech from './components/TextToSpeech';
import LearnSign from './components/LearnSign';
import About from './components/About';

export default function App() {
  const [activeTab, setActiveTab] = useState('home');

  return (
    <div className="min-vh-100 bg-light">
      <nav className="navbar navbar-expand-lg bg-white border-bottom shadow-sm sticky-top">
        <div className="container-fluid px-4">
          <span className="navbar-brand fw-bold text-primary d-flex align-items-center gap-2">
            <i className="bi bi-hand-index-thumb fs-4 text-purple"></i>
            SignFlow AI
          </span>
          <div className="d-flex gap-2">
            {['home', 'live', 'tts', 'learn', 'about'].map(tab => (
              <button
                key={tab}
                className={`btn btn-sm ${activeTab === tab ? 'btn-primary' : 'btn-outline-secondary'}`}
                onClick={() => setActiveTab(tab)}
              >
                {tab.toUpperCase()}
              </button>
            ))}
          </div>
        </div>
      </nav>

      <main className="container py-4">
        {activeTab === 'home' && (
          <div className="p-5 bg-white rounded-4 shadow-sm text-center">
            <h1 className="fw-bold mb-3">Sign Language Recognition System</h1>
            <p className="lead text-muted mb-4">Deep Learning and Computer Vision for Accessible Communication</p>
            <button className="btn btn-primary btn-lg" onClick={() => setActiveTab('live')}>
              Start Live Camera
            </button>
          </div>
        )}
        {activeTab === 'live' && <LiveDetection />}
        {activeTab === 'tts' && <TextToSpeech />}
        {activeTab === 'learn' && <LearnSign />}
        {activeTab === 'about' && <About />}
      </main>
    </div>
  );
}

