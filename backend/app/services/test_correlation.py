from correlation_service import correlate

# Fake OCR output
ocr_result = {
    "lines": [
        {
            "text": "DOB: 01/01/1998",
            "bbox": [
                [350, 180],
                [470, 180],
                [470, 220],
                [350, 220]
            ]
        },
        {
            "text": "NAME: SANKARSH",
            "bbox": [
                [50, 50],
                [250, 50],
                [250, 100],
                [50, 100]
            ]
        }
    ]
}

# Fake ELA output
ela_result = {
    "suspicious_regions": [
        {
            "x": 361,
            "y": 189,
            "w": 112,
            "h": 34
        }
    ]
}

# Fake MVSS output
tampering_result = {
    "suspicious_regions": [
        {
            "x": 350,
            "y": 180,
            "w": 120,
            "h": 40,
            "area": 4800
        }
    ]
}

result = correlate(
    ocr_result,
    ela_result,
    tampering_result
)

print("\n===== CORRELATION RESULT =====\n")
print(result)