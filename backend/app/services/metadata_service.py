import fitz
from PIL import Image
from PIL.ExifTags import TAGS


SUSPICIOUS_SOFTWARE = [
    "photoshop",
    "gimp",
    "canva",
    "pixlr",
    "paint",
]

def convert_keys_to_strings(obj):

    if isinstance(obj, dict):

        return {
            str(k): convert_keys_to_strings(v)
            for k, v in obj.items()
        }

    elif isinstance(obj, list):

        return [
            convert_keys_to_strings(item)
            for item in obj
        ]

    return obj


def analyze_metadata(file_path):

    result = {
        "file_type": None,
        "metadata": {},
        "flags": [],
        "risk_score": 0
    }

    # PDF
    if file_path.lower().endswith(".pdf"):

        doc = fitz.open(file_path)

        metadata = doc.metadata

        doc.close()

        result["file_type"] = "pdf"
        result["metadata"] = metadata

        creator = str(metadata.get("creator", "")).lower()
        producer = str(metadata.get("producer", "")).lower()

        for software in SUSPICIOUS_SOFTWARE:

            if software in creator:
                result["flags"].append(
                    f"Creator software: {software}"
                )
                result["risk_score"] += 20

            if software in producer:
                result["flags"].append(
                    f"Producer software: {software}"
                )
                result["risk_score"] += 20

    # IMAGE
    else:

        result["file_type"] = "image"

        try:

            image = Image.open(file_path)

            exif = image.getexif()

            metadata = {}

            for tag_id, value in exif.items():

                tag = TAGS.get(tag_id, tag_id)

                metadata[tag] = str(value)

            result["metadata"] = metadata

            software = str(
                metadata.get("Software", "")
            ).lower()

            for suspicious in SUSPICIOUS_SOFTWARE:

                if suspicious in software:

                    result["flags"].append(
                        f"Edited using {suspicious}"
                    )

                    result["risk_score"] += 20

        except Exception:

            result["flags"].append(
                "No image metadata available"
            )

    metadata_result = convert_keys_to_strings(
    result
    )

    return metadata_result