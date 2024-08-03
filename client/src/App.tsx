// App.tsx
import React from 'react';
import Dashboard from './pages/Dashboard';
import { GoogleOAuthProvider } from '@react-oauth/google'; // Adjust the import path as necessary
import './App.css'; // Import custom CSS for branding

const App: React.FC = () => {
  return (
    <GoogleOAuthProvider clientId="YOUR_GOOGLE_CLIENT_ID">
      <Dashboard />
    </GoogleOAuthProvider>
  );
};

export default App;