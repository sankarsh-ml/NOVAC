import fitz
import os



def extract_pdf_text(pdf_path: str):
    """
    Extract text directly from PDF.
    Returns empty string if PDF is scanned.
    """

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

    pdf = fitz.open(pdf_path)

    page = pdf[0]

    pix = page.get_pixmap(dpi=300, alpha=False)

    image_path = os.path.splitext(pdf_path)[0] + ".png"

    pix.save(image_path)

    pdf.close()

    return image_path