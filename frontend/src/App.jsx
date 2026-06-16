import { BrowserRouter, Routes, Route } from "react-router-dom";

import UploadPage from "./pages/UploadPage";
import HistoryPage from "./pages/HistoryPage";
import ResultsPage from "./pages/ResultsPage";
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
        </Routes>
      </BrowserRouter>
    </AnalysisProvider>
  );
}

export default App;
