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
  const navigate = useNavigate();

  async function handleUpload() {

    if (!file) {
      alert("Please select a file");
      return;
    }

    try {

      setLoading(true);

      const formData = new FormData();

      formData.append(
        "file",
        file
      );

      const response = await API.post(
        "/upload",
        formData,
        {
          headers: {
            "Content-Type":
              "multipart/form-data"
          }
        }
      );

      alert(
        "Analysis completed successfully!"
      );

      setFile(null);

      if (response.data?.case_id) {
        navigate(
          `/result/${response.data.case_id}`
        );
      }

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
