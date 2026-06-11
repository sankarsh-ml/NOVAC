import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";

import API from "../services/api";
import Navbar from "../components/Navbar";
import RiskBadge from "../components/RiskBadge";

import "./ResultsPage.css";

function ResultsPage() {
  const { caseId } = useParams();

  const [result, setResult] = useState(null);

  useEffect(() => {
    loadResult();
  }, []);

  async function loadResult() {
    try {
      const response = await API.get(
        `/results/case/${caseId}`
      );

      setResult(response.data);
    } catch (error) {
      console.error(error);
      alert("Failed to load analysis");
    }
  }

  const downloadReport = () => {
    window.open(
      `http://127.0.0.1:8000/report/${caseId}`,
      "_blank"
    );
  };

  const imageUrl = (path) =>
    path ? `http://localhost:8000/${path.replace(/\\/g, "/")}` : "";

  const metric = (value, fallback = 0) =>
    value ?? fallback;

  const normalizeScore = (value) => {
    const numeric = Number(value ?? 0);

    if (!Number.isFinite(numeric)) {
      return 0;
    }

    if (numeric >= 0 && numeric <= 1) {
      return Math.round(numeric * 100);
    }

    return Math.max(0, Math.min(100, Math.round(numeric)));
  };

  const DetectorCard = ({
    title,
    score,
    status,
    reasons = []
  }) => (
    <div className="detector-card">
      <div className="detector-head">
        <h3>{title}</h3>
        <span>{score}</span>
      </div>
      <p className="detector-status">
        {status}
      </p>
      {reasons.length > 0 && (
        <ul>
          {reasons.slice(0, 3).map((reason, index) => (
            <li key={index}>{reason}</li>
          ))}
        </ul>
      )}
    </div>
  );

  if (!result) {
    return (
      <>
        <Navbar />
        <h2
          style={{
            color: "white",
            textAlign: "center",
            marginTop: "100px",
          }}
        >
          Loading...
        </h2>
      </>
    );
  }

  const fraudScore =
    result?.fraud_analysis?.fraud_score ?? 0;

  const riskLevel =
    result?.fraud_analysis?.risk_level ?? "Unknown";

  const reasons =
    result?.fraud_analysis?.reasons ?? [];

  const components =
    result?.fraud_analysis?.components ?? {};

  const contributions =
    result?.fraud_analysis?.detector_contributions ?? {};

  const escalations =
    result?.fraud_analysis?.escalations ?? [];

  const fieldExtraction =
    result?.field_extraction_analysis ?? {};

  const condition =
    result?.document_condition_analysis ?? {};

  const photo =
    result?.photo_replacement_analysis ?? {};

  const visual =
    result?.visual_consistency_analysis ?? {};

  const quality =
    result?.document_quality_analysis ?? {};

  const forgery =
    result?.forgery_localization_analysis ?? {};

  const textConsistency =
    result?.text_consistency_analysis ?? {};

  const masking =
    result?.masking_analysis ?? {};

  const tampering =
    result?.tampering_analysis ?? {};

  const ela =
    result?.ela_analysis ?? {};

  const preprocessing =
    result?.mvss_preprocess_analysis
    ?? result?.preprocessing_analysis
    ?? {};

  return (
    <>
      <Navbar />

      <div className="results-page">
        <div className="case-banner">
          <div>
            <h1>Investigation Report</h1>
            <p>Case ID: {result.case_id}</p>
          </div>

          <button
            className="download-btn"
            onClick={downloadReport}
          >
            Download Report
          </button>
        </div>

        <div className="stats-grid">
          <div className="stat-card">
            <h3>Fraud Score</h3>
            <div className="big-number">
              {fraudScore}
            </div>
          </div>

          <div className="stat-card">
            <h3>Risk Level</h3>
            <RiskBadge level={riskLevel} />
          </div>

          <div className="stat-card">
            <h3>Evidence Count</h3>
            <div className="big-number">
              {reasons.length}
            </div>
          </div>
        </div>

        {quality.rejection_recommended && (
          <div className="reliability-banner">
            <h2>Analysis reliability low - please upload a clearer document.</h2>
            <p>
              Document appears too worn, blurred, or damaged for reliable automated verification.
            </p>
          </div>
        )}

        <div className="gauge-card">
          <h2>Fraud Probability</h2>

          <div
            className="gauge-circle"
            style={{
              background: `conic-gradient(
                #ef4444 ${fraudScore * 3.6}deg,
                #1e293b 0deg
              )`,
            }}
          >
            <div className="gauge-inner">
              {fraudScore}%
            </div>
          </div>
        </div>

        <div className="details-grid">
          {escalations.length > 0 && (
            <div className="escalation-card">
              <h2>Critical Escalations</h2>

              {escalations.map((escalation, index) => (
                <div
                  key={index}
                  className="escalation-item"
                >
                  {escalation}
                </div>
              ))}
            </div>
          )}

          <div className="reasons-card">
            <h2>Detection Reasons</h2>

            {reasons.length > 0 ? (
              reasons.map((reason, index) => (
                <div
                  key={index}
                  className="reason-item"
                >
                  {reason}
                </div>
              ))
            ) : (
              <p>No suspicious indicators found.</p>
            )}
          </div>

          <div className="metadata-card">
            <h2>Metadata</h2>

            <div className="meta-row">
              <span>Filename</span>
              <span>{result.filename}</span>
            </div>

            <div className="meta-row">
              <span>Case ID</span>
              <span>{result.case_id}</span>
            </div>

            <div className="meta-row">
              <span>Risk</span>
              <span>{riskLevel}</span>
            </div>

            <div className="meta-row">
              <span>Score</span>
              <span>{fraudScore}</span>
            </div>
          </div>
        </div>

        <div className="fields-card">
          <h2>Extracted Fields</h2>

          {Object.keys(fieldExtraction.fields ?? {}).length > 0 ? (
            <>
              <h3>Confirmed Fields</h3>
            <div className="fields-grid">
              {Object.entries(fieldExtraction.fields).map(([key, value]) => (
                <div
                  className="field-row"
                  key={key}
                >
                  <span>
                    {key.replaceAll("_", " ")}
                  </span>
                  <strong>{value}</strong>
                </div>
              ))}
            </div>
            </>
          ) : (
            <p>No structured fields extracted.</p>
          )}

          {(fieldExtraction.possible_values ?? []).length > 0 && (
            <>
              <h3>Possible Values</h3>
              <div className="fields-grid">
                {fieldExtraction.possible_values.slice(0, 12).map((item, index) => (
                  <div
                    className="field-row possible-value-row"
                    key={`${item.type}-${item.value}-${index}`}
                  >
                    <span>
                      {item.type.replaceAll("_", " ")}
                    </span>
                    <strong>{item.value}</strong>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>

        <div className="detector-section">
          <h2>Detector Signals</h2>

          <div className="detector-grid">
            <DetectorCard
              title="Document Quality"
              score={metric(quality.quality_score, "N/A")}
              status={
                quality.rejection_recommended
                  ? "Analysis unreliable - rescan recommended"
                  : "Document quality is acceptable for automated analysis"
              }
              reasons={quality.reasons}
            />

            {Number(contributions.detector_agreement?.contribution ?? 0) > 0 && (
              <DetectorCard
                title="Detector Agreement"
                score={metric(contributions.detector_agreement?.contribution)}
                status="Independent visual detectors agree on possible manipulation"
                reasons={
                  contributions.detector_agreement?.reason
                    ? [contributions.detector_agreement.reason]
                    : []
                }
              />
            )}

            <DetectorCard
              title="Physical Condition"
              score={metric(condition.condition_score)}
              status={
                condition.fold_detected || condition.tear_detected
                  ? `Fold, tear, or damaged-edge indicators found (${condition.condition_confidence ?? "low"})`
                  : `No major fold or tear indicators (${condition.condition_confidence ?? "low"})`
              }
              reasons={condition.reasons}
            />

            <DetectorCard
              title="Photo Replacement"
              score={metric(photo.replacement_score)}
              status={
                photo.photo_replacement_detected
                  ? "Possible photo/image region integrity signal"
                  : photo.printed_photo_likely
                  ? "Photo signal suppressed for this run"
                  : "No strong photo replacement signal"
              }
              reasons={
                photo.reasons?.length
                  ? photo.reasons
                  : photo.suppressed_reasons?.length
                  ? photo.suppressed_reasons
                  : photo.supporting_reasons
              }
            />

            <DetectorCard
              title="Forgery Localization"
              score={normalizeScore(forgery.forgery_score)}
              status={
                forgery.model_available === false
                  ? "TruFor unavailable during this run"
                  : contributions.trufor?.contribution === 0
                    && (forgery.suspicious_regions?.length ?? 0) > 0
                  ? "TruFor detected a region, but it overlapped normal/damaged document structure and was downweighted"
                  : forgery.manipulation_detected
                  ? "TruFor detected possible forged/manipulated image regions"
                  : "No strong TruFor forgery localization signal"
              }
              reasons={
                forgery.model_error
                  ? [forgery.model_error]
                  : forgery.reasons
              }
            />

            <DetectorCard
              title="Field Text Consistency"
              score={metric(textConsistency.field_mismatch_score)}
              status={
                textConsistency.font_mismatch_detected
                  ? "Possible local field text style mismatch detected"
                  : "No strong local field-level text mismatch detected"
              }
              reasons={
                textConsistency.error
                  ? [textConsistency.error]
                  : textConsistency.reasons
              }
            />

            <DetectorCard
              title="MVSS Preprocess"
              score={metric(preprocessing.removed_region_count)}
              status={
                preprocessing.qr_removed
                  ? `QR-like region removed before MVSS (${preprocessing.method})`
                  : "No QR region removed before MVSS"
              }
              reasons={
                preprocessing.error
                  ? [preprocessing.error]
                  : preprocessing.qr_removed
                  ? [
                      `${preprocessing.removed_region_count ?? preprocessing.removed_regions?.length ?? 0} region(s) masked before MVSS`
                    ]
                  : []
              }
            />

            <DetectorCard
              title="Visual Consistency"
              score={metric(visual.consistency_score)}
              status={
                visual.inconsistent_regions?.length
                  ? "Region-level noise, lighting, or blur mismatch"
                  : "No major regional inconsistency"
              }
              reasons={visual.reasons}
            />

            <DetectorCard
              title="ELA / Compression Consistency"
              score={metric(ela.ela_score)}
              status={
                (ela.suspicious_regions?.length ?? 0) > 0
                  ? "Supporting compression inconsistency signal"
                  : "No strong compression consistency signal"
              }
              reasons={
                ela.error
                  ? [ela.error]
                  : []
              }
            />

            <DetectorCard
              title="Visual Tampering"
              score={metric(tampering.tampering_score)}
              status={
                tampering.tampering_detected
                  && (tampering.scoring_region_count ?? tampering.suspicious_region_count ?? 0) > 0
                  ? "MVSS detected scoring-eligible suspicious visual manipulation regions"
                  : tampering.raw_region_count > 0
                  ? "Raw MVSS regions were suppressed as small/noisy/normal/damage regions"
                  : "No scoring-eligible MVSS tampering region"
              }
              reasons={
                tampering.error
                  ? [tampering.error]
                  : tampering.reasons?.length
                  ? tampering.reasons
                  : tampering.suppressed_regions?.length
                  ? ["Small/noisy MVSS regions were suppressed"]
                  : []
              }
            />

            <DetectorCard
              title="OCR Masking"
              score={metric(masking.masking_score)}
              status={
                masking.masking_detected
                  ? "Masked fields detected - Critical"
                  : "No masked fields detected"
              }
              reasons={masking.reasons}
            />
          </div>
        </div>

        <div className="component-card">
          <h2>Score Breakdown</h2>
          <div className="component-grid">
            {Object.entries(components).map(([key, value]) => (
              <div
                className="component-row"
                key={key}
              >
                <span>
                  {key.replaceAll("_", " ")}
                </span>
                <strong>{value}</strong>
              </div>
            ))}
          </div>
        </div>

        <div className="image-grid">
          {result.annotated_image_path && (
            <div>
              <h2>Suspicious Regions</h2>
              <img
                src={imageUrl(result.annotated_image_path)}
                alt="Annotated Analysis"
              />
            </div>
          )}

          {result.analysis_image_path && (
            <div>
              <h2>Original Document</h2>

              <img
                src={imageUrl(result.analysis_image_path)}
                alt="Original"
              />
            </div>
          )}
        </div>
      </div>
    </>
  );
}

export default ResultsPage;
