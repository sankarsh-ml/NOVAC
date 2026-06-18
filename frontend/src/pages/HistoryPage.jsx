import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

import API from "../services/api";
import Navbar from "../components/Navbar";
import RiskBadge from "../components/RiskBadge";

import "./HistoryPage.css";

function HistoryPage() {

  const [results, setResults] = useState([]);

  const navigate = useNavigate();

  useEffect(() => {
    loadHistory();
  }, []);

  async function loadHistory() {

    try {

      const response =
        await API.get("/results");

      setResults(response.data);

    } catch (error) {

      console.error(error);

      alert("Failed to load history");
    }
  }

  async function deleteCase(caseId) {

    const confirmDelete =
      window.confirm(
        "Delete this analysis?"
      );

    if (!confirmDelete) return;

    try {

      await API.delete(
        `/results/case/${caseId}`
      );

      setResults(
        prev =>
          prev.filter(
            item =>
              item.case_id !== caseId
          )
      );

    } catch (error) {

      console.error(error);

      alert("Delete failed");
    }
  }

  const criticalCount =
    results.filter(
      r =>
        r.risk_level === "Critical"
        || r.risk_level === "Synthetic Document Suspected"
    ).length;

  const highCount =
    results.filter(
      r =>
        r.risk_level === "High"
        || r.risk_level === "High Risk"
    ).length;

  const averageScore =
    results.length > 0
      ? Math.round(
          results.reduce(
            (sum, item) =>
              sum +
              (item.fraud_score || 0),
            0
          ) / results.length
        )
      : 0;

  return (
    <>
      <Navbar />

      <div className="history-page">

        <div className="history-header">

          <h1 className="history-title">
            Analysis History
          </h1>

          <p className="history-subtitle">
            View and manage all
            analyzed documents
          </p>

        </div>

        <div className="stats-grid">

          <div className="stat-card">

            <h3>Total Cases</h3>

            <h2>
              {results.length}
            </h2>

          </div>

          <div className="stat-card">

            <h3>Critical</h3>

            <h2>
              {criticalCount}
            </h2>

          </div>

          <div className="stat-card">

            <h3>High Risk</h3>

            <h2>
              {highCount}
            </h2>

          </div>

          <div className="stat-card">

            <h3>Avg Score</h3>

            <h2>
              {averageScore}
            </h2>

          </div>

        </div>

        <div className="table-container">

          <table className="history-table">

            <thead>

              <tr>

                <th>Case ID</th>

                <th>Filename</th>

                <th>Risk Level</th>

                <th>Fraud Score</th>

                <th>Field Extraction</th>

                <th>Actions</th>

              </tr>

            </thead>

            <tbody>

              {results.length === 0 ? (

                <tr>

                  <td
                    colSpan="6"
                    style={{
                      textAlign:
                        "center",
                      padding:
                        "30px"
                    }}
                  >
                    No analyses found
                  </td>

                </tr>

              ) : (

                results.map(result => (

                  <tr
                    key={result.case_id}
                  >

                    <td>
                      {result.case_id}
                    </td>

                    <td>
                      {result.filename}
                    </td>

                    <td>

                      <RiskBadge
                        level={
                          result.risk_level
                        }
                      />

                    </td>

                    <td>
                      {
                        result.fraud_score
                      }
                    </td>

                    <td>
                      <span className={`extraction-status ${result.field_extraction_status ?? "not_run"}`}>
                        {
                          (result.field_extraction_status ?? "not_run")
                            .replaceAll("_", " ")
                        }
                      </span>
                    </td>

                    <td>

                      <div
                        style={{
                          display:
                            "flex",
                          gap: "8px"
                        }}
                      >

                        <button
                          className="view-btn"
                          onClick={() => {

                            console.log(
                              "View",
                              result.case_id
                            );

                            navigate(
                              `/results/case/${result.case_id}`
                            );
                          }}
                        >
                          View
                        </button>

                        <button
                          className="delete-btn"
                          onClick={() =>
                            deleteCase(
                              result.case_id
                            )
                          }
                        >
                          Delete
                        </button>

                      </div>

                    </td>

                  </tr>

                ))

              )}

            </tbody>

          </table>

        </div>

      </div>

    </>
  );
}

export default HistoryPage;
