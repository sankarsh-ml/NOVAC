import { Link, useLocation } from "react-router-dom";
import "./Navbar.css";

function Navbar() {
  const location = useLocation();

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
        >
          History
        </Link>

      </div>

    </nav>
  );
}

export default Navbar;