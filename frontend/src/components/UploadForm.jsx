import { useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  FaUpload,
  FaShieldAlt,
  FaFileAlt,
  FaSearch
} from "react-icons/fa";

import API from "../services/api";
import "./UploadPage.css";

function UploadForm() {

  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [progressStatus, setProgressStatus] = useState(null);
  const navigate = useNavigate();

  async function handleUpload() {

    if (!file) {
      alert("Please select a file");
      return;
    }

    const wait = (milliseconds) =>
      new Promise((resolve) =>
        setTimeout(resolve, milliseconds)
      );

    try {

      setLoading(true);

      const formData = new FormData();

      formData.append(
        "file",
        file
      );

      setProgressStatus({
        stage: "Upload received",
        progress: 5,
        message: "Starting analysis"
      });

      const startResponse = await API.post(
        "/analyze/start",
        formData,
        {
          headers: {
            "Content-Type":
              "multipart/form-data"
          }
        }
      );

      const caseId = startResponse.data?.case_id;

      if (!caseId) {
        throw new Error("Analysis did not return a case ID");
      }

      while (true) {
        await wait(1000);

        let statusResponse = null;

        try {
          statusResponse = await API.get(
            `/analysis/status/${caseId}`
          );
        } catch {
          // Status may not exist during the first few milliseconds.
          continue;
        }

        setProgressStatus(statusResponse.data);

        if (statusResponse.data?.error) {
          throw new Error(statusResponse.data.error);
        }

        if (statusResponse.data?.stage === "Analysis complete") {
          break;
        }
      }

      setFile(null);
      navigate(
        `/results/case/${caseId}`
      );

    } catch (error) {

      console.error(error);

      alert(
        "Upload failed"
      );

    } finally {

      setLoading(false);

    }
  }

  return (

    <div className="upload-page">

      <div className="hero-card">

        <h1 className="hero-title">
          NOVAC
        </h1>

        <p className="hero-subtitle">
          AI-Powered Document Fraud Detection System
        </p>

        <label className="upload-zone">

          <FaUpload size={60} />

          <h3>
            Drop Document Here
          </h3>

          <p>
            or click to browse
          </p>

          <input
            type="file"
            hidden
            accept=".pdf,.jpg,.jpeg,.png"
            onChange={(e) =>
              setFile(
                e.target.files[0]
              )
            }
          />

        </label>

        {file && (

          <div className="selected-file">
            File: {file.name}
          </div>

        )}

        <button
          className="analyze-btn"
          onClick={handleUpload}
          disabled={loading}
        >

          {loading
            ? "Analyzing..."
            : "Analyze Document"}

        </button>

        {loading && progressStatus && (
          <div className="analysis-progress-card">
            <div className="analysis-progress-head">
              <span>{progressStatus.stage}</span>
              <strong>{progressStatus.progress}%</strong>
            </div>
            <div className="analysis-progress-track">
              <div
                className="analysis-progress-fill"
                style={{
                  width: `${progressStatus.progress}%`
                }}
              />
            </div>
            <p>{progressStatus.message}</p>
            {progressStatus.error && (
              <p className="analysis-progress-error">
                {progressStatus.error}
              </p>
            )}
          </div>
        )}

      </div>

      <div className="features-grid">

        <div className="feature-card">

          <FaSearch size={40} />

          <h3>
            OCR Analysis
          </h3>

          <p>
            Extract and validate
            document text
          </p>

        </div>

        <div className="feature-card">

          <FaShieldAlt size={40} />

          <h3>
            Fraud Detection
          </h3>

          <p>
            AI-powered tampering
            detection
          </p>

        </div>

        <div className="feature-card">

          <FaFileAlt size={40} />

          <h3>
            Metadata Scan
          </h3>

          <p>
            Analyze hidden document
            information
          </p>

        </div>

      </div>

    </div>
  );
}

export default UploadForm;
