import { BrowserRouter, Routes, Route } from "react-router-dom";

import UploadPage from "./pages/UploadPage";
import HistoryPage from "./pages/HistoryPage";
import ResultsPage from "./pages/ResultsPage";
import ExtractionResultsPage from "./pages/ExtractionResultsPage";
import AnalysisProvider from "./context/AnalysisProvider";

function App() {
  return (
    <AnalysisProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<UploadPage />} />
          <Route path="/history" element={<HistoryPage />} />
          <Route path="/result/:caseId" element={<ResultsPage />} />
          <Route path="/results/case/:caseId" element={<ResultsPage />} />
          <Route path="/extraction-results/:caseId" element={<ExtractionResultsPage />} />
        </Routes>
      </BrowserRouter>
    </AnalysisProvider>
  );
}

export default App;
