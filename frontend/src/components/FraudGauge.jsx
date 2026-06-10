import {
  RadialBarChart,
  RadialBar,
  ResponsiveContainer
} from "recharts";

function FraudGauge({ score }) {

  const data = [
    {
      name: "Fraud Score",
      value: score,
      fill:
        score >= 80
          ? "#ef4444"
          : score >= 50
          ? "#f97316"
          : "#22c55e"
    }
  ];

  return (
    <div
      style={{
        width: "100%",
        height: 250
      }}
    >
      <ResponsiveContainer>
        <RadialBarChart
          innerRadius="70%"
          outerRadius="100%"
          data={data}
          startAngle={180}
          endAngle={0}
        >
          <RadialBar
            background
            dataKey="value"
            cornerRadius={10}
          />
        </RadialBarChart>
      </ResponsiveContainer>

      <div
        style={{
          marginTop: "-120px",
          textAlign: "center"
        }}
      >
        <h1>{score}</h1>
        <p>Fraud Score</p>
      </div>
    </div>
  );
}

export default FraudGauge;