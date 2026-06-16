import { useMemo, useState } from "react";
import AnalysisContext from "./analysisContext";

function AnalysisProvider({ children }) {
  const [analysisInProgress, setAnalysisInProgress] =
    useState(false);

  const value = useMemo(
    () => ({
      analysisInProgress,
      setAnalysisInProgress
    }),
    [analysisInProgress]
  );

  return (
    <AnalysisContext.Provider value={value}>
      {children}
    </AnalysisContext.Provider>
  );
}

export default AnalysisProvider;
