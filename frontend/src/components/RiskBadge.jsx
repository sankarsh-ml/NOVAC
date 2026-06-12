function RiskBadge({ level }) {

  const colors = {
    Critical: "#ef4444",
    "High Risk": "#ef4444",
    "Synthetic Document Suspected": "#ef4444",
    "Analysis Inconclusive": "#f59e0b",
    High: "#f97316",
    "Medium Risk": "#eab308",
    Medium: "#eab308",
    "Low Risk": "#22c55e",
    "Likely Authentic": "#22c55e",
    Low: "#22c55e"
  };

  return (
    <span
      style={{
        background: colors[level] || "#64748b",
        color: "white",
        padding: "6px 12px",
        borderRadius: "999px",
        fontWeight: 600,
        fontSize: "0.85rem"
      }}
    >
      {level}
    </span>
  );
}

export default RiskBadge;
