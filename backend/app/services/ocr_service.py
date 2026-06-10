from paddleocr import PaddleOCR

ocr = PaddleOCR(
    use_angle_cls=True,
    lang="en"
)


def extract_text(image_path):

    result = ocr.ocr(image_path, cls=True)

    texts = []
    confidences = []
    line_results = []

    for line in result[0]:

        # line format from PaddleOCR:
        # line[0] -> bbox (4 points)
        # line[1] -> (text, confidence)
        bbox = line[0]
        text = line[1][0]
        confidence = float(line[1][1])

        texts.append(text)
        confidences.append(confidence)

        line_results.append({
            "text": text,
            "confidence": round(confidence, 3),
            "bbox": bbox,
        })

    avg_confidence = (
        sum(confidences) / len(confidences)
        if confidences else 0
    )

    return {
        "text": "\n".join(texts),
        "avg_confidence": round(avg_confidence, 3),
        "lines": line_results
    }

