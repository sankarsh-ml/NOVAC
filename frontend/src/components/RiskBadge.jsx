function RiskBadge({ level }) {

  const colors = {
    Critical: "#ef4444",
    High: "#f97316",
    Medium: "#eab308",
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