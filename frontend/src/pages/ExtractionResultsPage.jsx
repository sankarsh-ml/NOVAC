import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";

import API from "../services/api";
import Navbar from "../components/Navbar";

import "./ExtractionResultsPage.css";

const API_ORIGIN = API.defaults.baseURL ?? "http://127.0.0.1:8000";

const label = (value) =>
  String(value ?? "")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (character) => character.toUpperCase())
    .replace("Dob", "DOB")
    .replace("Pan", "PAN")
    .replace("Aadhaar", "Aadhaar");

const confidence = (value) => {
  if (value === null || value === undefined || value === "") {
    return "N/A";
  }

  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return String(value);
  }

  if (numeric >= 0 && numeric <= 1) {
    return `${Math.round(numeric * 100)}%`;
  }

  return `${Math.round(numeric)}%`;
};

const isAadhaarDocument = (documentType) =>
  ["aadhaar", "aadhaar card", "aadhar"].includes(
    String(documentType ?? "").trim().toLowerCase()
  );

const formatAadhaar = (value) => {
  const digits = String(value ?? "").replace(/\D/g, "");
  if (digits.length === 12) {
    return `${digits.slice(0, 4)}-${digits.slice(4, 8)}-${digits.slice(8)}`;
  }
  return String(value ?? "");
};

function FieldInspectionCard({
  fieldKey,
  fieldData,
  aadhaar,
  maskAadhaar,
  displayValue,
  cropUrl,
  cropError,
  setFailedCropUrl,
}) {
  if (!fieldData) {
    return (
      <div className="focused-field-card">
        <h3>Field Not Found</h3>
        <p>This field is not available in the extraction result.</p>
      </div>
    );
  }

  const hasBBox =
    Array.isArray(fieldData?.bbox)
    && fieldData.bbox.length === 4
    && fieldData.bbox.every((value) => Number.isFinite(Number(value)));

  return (
    <div className="focused-field-card">
      <div className="focused-field-header">
        <div>
          <span>Selected Field</span>
          <h3>{label(fieldKey)}</h3>
        </div>
      </div>

      <div className="field-detail-grid">
        <div className="field-detail-row">
          <span>Value</span>
          <strong>{displayValue(fieldKey, fieldData) || "Not available"}</strong>
        </div>

        <div className="field-detail-row">
          <span>Box Confidence</span>
          <strong>{confidence(fieldData?.box_confidence)}</strong>
        </div>

        <div className="field-detail-row">
          <span>OCR Confidence</span>
          <strong>{confidence(fieldData?.ocr_confidence)}</strong>
        </div>

        <div className="field-detail-row">
          <span>Bounding Box</span>
          <strong>{hasBBox ? `[${fieldData.bbox.join(", ")}]` : "N/A"}</strong>
        </div>
      </div>

      {fieldKey === "aadhaar_number" && aadhaar && (
        <p className="aadhaar-note">
          Aadhaar is {maskAadhaar ? "currently masked" : "currently unmasked"}.
          Use the Aadhaar control in the summary card to change this.
        </p>
      )}

      <div className="field-crop-section">
        <h3>Cropped Field Image</h3>
        {hasBBox ? (
          cropError ? (
            <p className="crop-error">Could not load field crop.</p>
          ) : (
            <img
              src={cropUrl}
              alt={`${label(fieldKey)} crop`}
              className="field-crop-image"
              onError={() => setFailedCropUrl(cropUrl)}
              onLoad={() => setFailedCropUrl("")}
            />
          )
        ) : (
          <p className="crop-unavailable">Crop unavailable for this field.</p>
        )}
      </div>
    </div>
  );
}

function ExtractionResultsPage() {
  const { caseId } = useParams();
  const navigate = useNavigate();

  const [analysis, setAnalysis] = useState(null);
  const [extraction, setExtraction] = useState(null);
  const [loading, setLoading] = useState(true);
  const [retrying, setRetrying] = useState(false);
  const [error, setError] = useState("");
  const [maskAadhaar, setMaskAadhaar] = useState(true);
  const [selectedField, setSelectedField] = useState("all");
  const [failedCropUrl, setFailedCropUrl] = useState("");

  const loadData = useCallback(async () => {
    setLoading(true);
    setError("");

    try {
      const [analysisResponse, extractionResponse] = await Promise.all([
        API.get(`/results/case/${caseId}`),
        API.get(`/api/extraction-result/${caseId}`),
      ]);
      setAnalysis(analysisResponse.data);
      setExtraction(extractionResponse.data);
    } catch (requestError) {
      console.error(requestError);
      setError(
        requestError?.response?.data?.detail
        ?? "Unable to load extraction results."
      );
    } finally {
      setLoading(false);
    }
  }, [caseId]);

  useEffect(() => {
    void Promise.resolve().then(loadData);
  }, [loadData]);

  const fields = useMemo(
    () => Object.entries(extraction?.fields ?? {}),
    [extraction]
  );

  const aadhaar = isAadhaarDocument(extraction?.document_type);
  const aadhaarField = extraction?.fields?.aadhaar_number;
  const canShowFullAadhaar = aadhaar && Boolean(aadhaarField?.raw_value);
  const selectedFieldExists =
    selectedField === "all"
    || Object.prototype.hasOwnProperty.call(extraction?.fields ?? {}, selectedField);
  const activeSelectedField = selectedFieldExists ? selectedField : "all";
  const selectedFieldData = extraction?.fields?.[activeSelectedField];
  const cropUrl =
    activeSelectedField !== "all"
      ? `${API_ORIGIN}/api/extraction-field-crop/${encodeURIComponent(caseId)}/${encodeURIComponent(activeSelectedField)}`
      : "";
  const cropError = Boolean(cropUrl) && failedCropUrl === cropUrl;

  const imagePath =
    analysis?.analysis_image_path
    ?? analysis?.file_path
    ?? "";

  const imageUrl = imagePath
    ? `${API_ORIGIN}/${String(imagePath).replace(/\\/g, "/")}`
    : "";

  const displayValue = (fieldKey, fieldData) => {
    if (
      fieldKey === "aadhaar_number"
      && aadhaar
      && !maskAadhaar
      && fieldData?.raw_value
    ) {
      return formatAadhaar(fieldData.raw_value);
    }

    return fieldData?.value ?? "";
  };

  const retryExtraction = async () => {
    setRetrying(true);
    setError("");

    try {
      const response = await API.post(`/api/extract-fields/${caseId}?force=true`);
      setExtraction(response.data);
    } catch (requestError) {
      console.error(requestError);
      setError(
        requestError?.response?.data?.detail
        ?? "Retry failed. Please try again."
      );
    } finally {
      setRetrying(false);
    }
  };

  const downloadExtractionReport = (masked = true) => {
    window.open(
      `${API_ORIGIN}/api/extraction-report/${caseId}?mask_aadhaar=${masked}`,
      "_blank"
    );
  };

  if (loading) {
    return (
      <>
        <Navbar />
        <div className="extraction-page">
          <div className="extraction-card">Loading extraction results...</div>
        </div>
      </>
    );
  }

  return (
    <>
      <Navbar />

      <div className="extraction-page">
        <div className="extraction-header">
          <div>
            <h1>Extraction Results</h1>
            <p>{analysis?.filename ?? "Document"} | {caseId}</p>
          </div>

          <button
            className="secondary-btn"
            onClick={() => navigate(`/results/case/${caseId}`)}
          >
            Back to Fraud Result
          </button>
        </div>

        {error && (
          <div className="extraction-alert error">
            {error}
          </div>
        )}

        {extraction?.status === "not_run" && (
          <div className="extraction-alert">
            Field extraction has not been run for this document yet.
          </div>
        )}

        {extraction?.status === "skipped" && (
          <div className="extraction-alert warning">
            {extraction.reason}
          </div>
        )}

        <div className="extraction-layout">
          <section className="extraction-card preview-card">
            <h2>Original Document</h2>
            {imageUrl ? (
              <img
                src={imageUrl}
                alt="Original uploaded document"
              />
            ) : (
              <p>No document preview available.</p>
            )}
          </section>

          <section className="extraction-card">
            <h2>Summary</h2>
            <div className="summary-row">
              <span>Document Type</span>
              <strong className="type-badge">
                {label(extraction?.document_type || "Unknown")}
              </strong>
            </div>
            <div className="summary-row">
              <span>Status</span>
              <strong>{label(extraction?.status || "Unknown")}</strong>
            </div>
            <div className="summary-row">
              <span>Type Confidence</span>
              <strong>{confidence(extraction?.document_type_confidence)}</strong>
            </div>

            {aadhaar && canShowFullAadhaar && (
              <div className="aadhaar-control">
                <p>
                  Full Aadhaar number is sensitive. Show only when required.
                </p>
                <button
                  className="secondary-btn"
                  onClick={() => setMaskAadhaar((current) => !current)}
                >
                  {maskAadhaar ? "Show Full Aadhaar Number" : "Hide Aadhaar Number"}
                </button>
              </div>
            )}
          </section>
        </div>

        <section className="extraction-card">
          <div className="section-header">
            <h2>Extracted Fields</h2>
            <div className="report-actions">
              {aadhaar ? (
                <>
                  <button
                    className="primary-btn"
                    onClick={() => downloadExtractionReport(true)}
                    disabled={extraction?.status !== "completed"}
                  >
                    Download Masked Report
                  </button>
                  {canShowFullAadhaar && (
                    <button
                      className="secondary-btn"
                      onClick={() => downloadExtractionReport(false)}
                      disabled={extraction?.status !== "completed"}
                    >
                      Download Full Aadhaar Report
                    </button>
                  )}
                </>
              ) : (
                <button
                  className="primary-btn"
                  onClick={() => downloadExtractionReport(true)}
                  disabled={extraction?.status !== "completed"}
                >
                  Download Extraction Report
                </button>
              )}
            </div>
          </div>

          {fields.length > 0 && (
            <div className="field-query-section">
              <label className="field-query-label" htmlFor="field-query-select">
                Select Field to Inspect
              </label>
              <select
                id="field-query-select"
                className="field-query-dropdown"
                value={activeSelectedField}
                onChange={(event) => setSelectedField(event.target.value)}
              >
                <option value="all">All Fields</option>
                {fields.map(([fieldKey]) => (
                  <option key={fieldKey} value={fieldKey}>
                    {label(fieldKey)}
                  </option>
                ))}
              </select>
            </div>
          )}

          {fields.length > 0 ? (
            activeSelectedField === "all" ? (
              <div className="field-table-wrap">
                <table className="field-table">
                  <thead>
                    <tr>
                      <th>Field</th>
                      <th>Value</th>
                      <th>Box Confidence</th>
                      <th>OCR Confidence</th>
                    </tr>
                  </thead>
                  <tbody>
                    {fields.map(([fieldKey, fieldData]) => (
                      <tr key={fieldKey}>
                        <td>{label(fieldKey)}</td>
                        <td>{displayValue(fieldKey, fieldData)}</td>
                        <td>{confidence(fieldData?.box_confidence)}</td>
                        <td>{confidence(fieldData?.ocr_confidence)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <FieldInspectionCard
                fieldKey={activeSelectedField}
                fieldData={selectedFieldData}
                aadhaar={aadhaar}
                maskAadhaar={maskAadhaar}
                displayValue={displayValue}
                cropUrl={cropUrl}
                cropError={cropError}
                setFailedCropUrl={setFailedCropUrl}
              />
            )
          ) : (
            <p>No extracted fields available.</p>
          )}
        </section>

        <div className="extraction-layout">
          <section className="extraction-card">
            <h2>Missing Fields</h2>
            {(extraction?.missing_fields ?? []).length > 0 ? (
              <div className="chip-list">
                {extraction.missing_fields.map((field) => (
                  <span key={field}>{label(field)}</span>
                ))}
              </div>
            ) : (
              <p>None reported.</p>
            )}
          </section>

          <section className="extraction-card">
            <h2>Warnings / Errors</h2>
            {(extraction?.warnings ?? []).length > 0 ? (
              <ul className="message-list">
                {extraction.warnings.map((warning, index) => (
                  <li key={index}>{warning}</li>
                ))}
              </ul>
            ) : extraction?.error ? (
              <p className="error-text">{extraction.error}</p>
            ) : (
              <p>None reported.</p>
            )}
          </section>
        </div>

        {extraction?.status === "failed" && (
          <div className="retry-row">
            <button
              className="primary-btn"
              onClick={retryExtraction}
              disabled={retrying}
            >
              {retrying ? "Retrying extraction..." : "Retry Extraction"}
            </button>
          </div>
        )}
      </div>
    </>
  );
}

export default ExtractionResultsPage;
