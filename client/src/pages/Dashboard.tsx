import React, { useState, useEffect } from 'react';
import AnswerGenerator from '../components/AnswerGenerator';
import GhostLoader from '../components/GhostLoader';
import './Dashboard.css'; // Ensure this path is correct

const Dashboard: React.FC = () => {
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    // Simulate a loading process
    const timer = setTimeout(() => {
      setLoading(false);
    }, 1000); // Adjust the timeout as needed

    return () => clearTimeout(timer);
  }, []);

  return (
    <div className="container">
      <h1 className="title is-1">
        REPO
        <img src="./src/assets/images/logo.png" alt="Repo Ripper Logo" className="logo-inline" />
        RIPPER
      </h1>
      <h2 className="subtitle is-4">Unlocks the power of your code repositories</h2>
      <hr />
      <div className="section">
        {loading && <GhostLoader />}
        {!loading && (
          <>
            <div className="columns is-centered">
              <div className="column is-half">
                <AnswerGenerator />
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default Dashboard;