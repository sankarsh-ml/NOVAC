import { BrowserRouter, Routes, Route } from "react-router-dom";

import UploadPage from "./pages/UploadPage";
import HistoryPage from "./pages/HistoryPage";
import ResultsPage from "./pages/ResultsPage";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<UploadPage />} />
        <Route path="/history" element={<HistoryPage />} />
        <Route path="/result/:caseId" element={<ResultsPage />} />
        <Route path="/results/case/:caseId" element={<ResultsPage />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
