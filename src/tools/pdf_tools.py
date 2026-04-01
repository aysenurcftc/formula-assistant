import threading
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import save_output
from typing import Union
from pathlib import Path

_model_dict = None
_model_lock = threading.Lock()

def get_model_dict():
    global _model_dict
    with _model_lock:
        if _model_dict is None:
            _model_dict = create_model_dict()
    return _model_dict

def convert_pdf_to_md(pdf_path: Union[str, Path], output_dir: Union[str, Path] = "./output") -> str:
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    md_path = output_dir / f"{pdf_path.stem}.md"

    if md_path.exists():
        print(f"[PDF→MD] Using existing file: {md_path}")
        return str(md_path)

    converter = PdfConverter(artifact_dict=get_model_dict())
    rendered = converter(str(pdf_path))
    save_output(rendered, str(output_dir), pdf_path.stem)
    return str(md_path)