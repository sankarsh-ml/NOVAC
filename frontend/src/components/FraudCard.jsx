function FraudCard({ result }) {

  if (!result) return null;

  return (
    <div
      style={{
        border: "1px solid gray",
        padding: "20px",
        marginTop: "20px",
        borderRadius: "10px"
      }}
    >

      <h2>Analysis Report</h2>

      <p>
        <strong>Case ID:</strong>{" "}
        {result.case_id}
      </p>

      <p>
        <strong>Fraud Score:</strong>{" "}
        {result.fraud_analysis?.fraud_score}
      </p>

      <p>
        <strong>Risk Level:</strong>{" "}
        {result.fraud_analysis?.risk_level}
      </p>

      <p>
        <strong>Average OCR Confidence:</strong>{" "}
        {result.avg_confidence}
      </p>

      <p>
        <strong>Tampering Detected:</strong>{" "}
        {
          result.tampering_analysis?.tampering_detected
            ? "YES"
            : "NO"
        }
      </p>

    </div>
  );
}

export default FraudCard;