import { useContext } from "react";
import AnalysisContext from "./analysisContext";

function useAnalysis() {
  const context = useContext(AnalysisContext);

  if (!context) {
    throw new Error(
      "useAnalysis must be used within AnalysisProvider"
    );
  }

  return context;
}

export default useAnalysis;
