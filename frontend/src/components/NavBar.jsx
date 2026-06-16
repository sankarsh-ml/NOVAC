import { useEffect, useState } from "react";
import { Link, useLocation } from "react-router-dom";
import useAnalysis from "../context/useAnalysis";
import "./Navbar.css";

function Navbar() {
  const location = useLocation();
  const { analysisInProgress } = useAnalysis();
  const [noticeVisible, setNoticeVisible] = useState(false);

  useEffect(() => {
    if (!noticeVisible) return undefined;

    const timeoutId = window.setTimeout(
      () => setNoticeVisible(false),
      3000
    );

    return () => window.clearTimeout(timeoutId);
  }, [noticeVisible]);

  function handleHistoryClick(event) {
    if (!analysisInProgress) return;

    event.preventDefault();
    setNoticeVisible(true);
  }

  return (
    <nav className="navbar">

      <div className="navbar-logo">
        NOVAC
      </div>

      <div className="navbar-links">

        <Link
          className={
            location.pathname === "/"
              ? "active"
              : ""
          }
          to="/"
        >
          Upload
        </Link>

        <Link
          className={
            location.pathname.includes(
              "/history"
            )
              ? "active"
              : ""
          }
          to="/history"
          onClick={handleHistoryClick}
          aria-disabled={analysisInProgress}
        >
          History
        </Link>

      </div>

      {noticeVisible && (
        <div
          className="navbar-notice"
          role="status"
          aria-live="polite"
        >
          Please wait, analysis in progress
        </div>
      )}

    </nav>
  );
}

export default Navbar;
