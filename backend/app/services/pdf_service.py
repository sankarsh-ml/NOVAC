import os

try:
    import fitz
except Exception:
    fitz = None

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None



def extract_pdf_text(pdf_path: str):
    """
    Extract text directly from PDF.
    Returns empty string if PDF is scanned.
    """

    if fitz is None:
        if PdfReader is None:
            return ""

        reader = PdfReader(pdf_path)

        return "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        ).strip()

    doc = fitz.open(pdf_path)

    text = ""

    for page in doc:
        text += page.get_text()

    doc.close()

    return text.strip()


def pdf_to_image(pdf_path: str):
    """
    Convert first page of PDF to image.
    """

    if fitz is None:
        raise RuntimeError(
            "PDF rendering unavailable: PyMuPDF is not installed"
        )

    pdf = fitz.open(pdf_path)

    page = pdf[0]

    pix = page.get_pixmap(dpi=150, alpha=False)

    image_path = os.path.splitext(pdf_path)[0] + ".png"

    pix.save(image_path)

    pdf.close()

    return image_path
