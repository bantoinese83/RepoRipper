import React, { useState } from 'react';
import { generateAnswer } from '../api/api';
import { toast, ToastContainer } from 'react-toastify';
import { CopyOutlined, InfoCircleOutlined } from '@ant-design/icons';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { solarizedlight } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { Tooltip } from 'antd';
import 'react-toastify/dist/ReactToastify.css';
import './Loader.css'; // Import the custom CSS for loader animation

const AnswerGenerator: React.FC = () => {
  const [repoFullName, setRepoFullName] = useState('');
  const [branch, setBranch] = useState('main');
  const [customBranch, setCustomBranch] = useState('');
  const [query, setQuery] = useState('');
  const [filePath, setFilePath] = useState('');
  const [answer, setAnswer] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleGenerateAnswer = async () => {
    try {
      setLoading(true);
      const selectedBranch = branch === 'custom' ? customBranch : branch;
      const data = await generateAnswer(repoFullName, query, selectedBranch, filePath);
      setAnswer(data.answer);
      toast.success('Answer generated successfully!', {
        position: 'top-right',
        autoClose: 3000,
        theme: 'colored',
      });
    } catch (error) {
      toast.error(`Error generating answer: ${error.response?.data?.detail || error.message}`, {
        position: 'top-right',
        autoClose: 5000,
        theme: 'colored',
      });
    } finally {
      setLoading(false);
    }
  };

  const handleCopyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    toast.success('Copied to clipboard!', {
      position: 'top-right',
      autoClose: 3000,
      theme: 'colored',
    });
  };

  const parseAnswer = (answer: string) => {
    const codeBlockRegex = /```(\w+)?\n([\s\S]*?)```/g;
    const parsedElements = [];
    let lastIndex = 0;
    let match;

    while ((match = codeBlockRegex.exec(answer)) !== null) {
      if (match.index > lastIndex) {
        const textBlock = answer.slice(lastIndex, match.index);
        parsedElements.push(
          <div key={`text-${lastIndex}`} className="copy-button-container">
            <SyntaxHighlighter language="plaintext" style={solarizedlight}>
              {textBlock}
            </SyntaxHighlighter>
            <Tooltip title={<SyntaxHighlighter language="plaintext" style={solarizedlight}>{textBlock}</SyntaxHighlighter>}>
              <button
                className="button is-link is-small copy-button"
                onClick={() => handleCopyToClipboard(textBlock)}
              >
                <CopyOutlined />
              </button>
            </Tooltip>
          </div>
        );
      }

      const language = match[1] || 'text';
      const code = match[2];
      parsedElements.push(
        <div key={match.index} className="copy-button-container">
          <SyntaxHighlighter language={language} style={solarizedlight}>
            {code}
          </SyntaxHighlighter>
          <Tooltip title={<SyntaxHighlighter language={language} style={solarizedlight}>{code}</SyntaxHighlighter>}>
            <button
              className="button is-link is-small copy-button"
              onClick={() => handleCopyToClipboard(code)}
            >
              <CopyOutlined />
            </button>
          </Tooltip>
        </div>
      );

      lastIndex = match.index + match[0].length;
    }

    if (lastIndex < answer.length) {
      const textBlock = answer.slice(lastIndex);
      parsedElements.push(
        <div key={`text-${lastIndex}`} className="copy-button-container">
          <SyntaxHighlighter language="plaintext" style={solarizedlight}>
            {textBlock}
          </SyntaxHighlighter>
          <Tooltip title={<SyntaxHighlighter language="plaintext" style={solarizedlight}>{textBlock}</SyntaxHighlighter>}>
            <button
              className="button is-link is-small copy-button"
              onClick={() => handleCopyToClipboard(textBlock)}
            >
              <CopyOutlined />
            </button>
          </Tooltip>
        </div>
      );
    }

    return parsedElements;
  };

  return (
    <div className="box p-4 wider-box">
      <ToastContainer />
      <div className="field">
        <label className="label">
          Repository Full Name
          <Tooltip title="Enter the full name of the repository (e.g., facebook/react)">
            <InfoCircleOutlined className="ml-2" />
          </Tooltip>
        </label>
        <div className="control">
          <input
            className="input"
            type="text"
            value={repoFullName}
            onChange={(e) => setRepoFullName(e.target.value)}
            placeholder="e.g., bantoinese83/smart-invoice-gen"
          />
        </div>
      </div>

      <div className="field">
        <label className="label">
          Branch
          <Tooltip title="Select or enter the branch name from which to generate the answer. You can choose 'Custom' to specify your branch name.">
            <InfoCircleOutlined className="ml-2" />
          </Tooltip>
        </label>
        <div className="control">
          <div className="select">
            <select value={branch} onChange={(e) => setBranch(e.target.value)}>
              <option value="main">main</option>
              <option value="master">master</option>
              <option value="develop">develop</option>
              <option value="feature">feature</option>
              <option value="release">release</option>
              <option value="custom">Custom</option>
            </select>
          </div>
        </div>
      </div>

      {branch === 'custom' && (
        <div className="field">
          <label className="label">
            Custom Branch
            <Tooltip title="Enter your custom branch name here.">
              <InfoCircleOutlined className="ml-2" />
            </Tooltip>
          </label>
          <div className="control">
            <input
              className="input"
              type="text"
              value={customBranch}
              onChange={(e) => setCustomBranch(e.target.value)}
              placeholder="Enter custom branch name"
            />
          </div>
        </div>
      )}

      <div className="field">
        <label className="label">
          Query
          <Tooltip title="Enter the query you want to generate an answer for.">
            <InfoCircleOutlined className="ml-2" />
          </Tooltip>
        </label>
        <div className="control">
          <input
            className="input"
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Enter your query"
          />
        </div>
      </div>

      <div className="field">
        <label className="label">
          File Path
          <Tooltip title="Enter the file path if applicable (optional).">
            <InfoCircleOutlined className="ml-2" />
          </Tooltip>
        </label>
        <div className="control">
          <input
            className="input"
            type="text"
            value={filePath}
            onChange={(e) => setFilePath(e.target.value)}
            placeholder="Enter file path (optional)"
          />
        </div>
      </div>

      <div className="field">
        <div className="control">
          <button className="button is-primary" onClick={handleGenerateAnswer} disabled={loading}>
            {loading ? (
              <div className="loader-container">
                <div className="loader-blocks"></div>
                <div className="loader-blocks"></div>
                <div className="loader-blocks"></div>
              </div>
            ) : (
              'Generate Answer'
            )}
          </button>
        </div>
      </div>

      {answer && (
        <div className="content">
          <div className="box">
            <h2 className="subtitle">Generated Answer</h2>
            {parseAnswer(answer)}
          </div>
        </div>
      )}
    </div>
  );
};

export default AnswerGenerator;
