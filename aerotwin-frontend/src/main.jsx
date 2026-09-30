/**
 * src/main.jsx
 * Application entry point for NIDAN.
 * Imports global styles and renders the React app.
 */
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import App from './App';

/* Global styles (order matters: theme → glass → background) */
import './styles/theme.css';
import './styles/glass.css';
import './styles/background.css';

import { PROJECT_NAME } from './config';

/* Set document title */
document.title = `${PROJECT_NAME} — UAV Engine Digital Twin`;

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </StrictMode>
);
