import os
from pathlib import Path
from fpdf import FPDF

PARENT=Path(__file__).resolve().parent
MISC=PARENT.parent
SRC=MISC.parent
ROOT=SRC.parent

GoNotoK_path=ROOT/'src'/'misc'/'resources'/'Fonts'/'GoNotoKurrentRegular'/'GoNotoKurrent-Regular.ttf'
GoNotoC_path=ROOT/'src'/'misc'/'resources'/'Fonts'/'GoNotoCurrentRegular'/'GoNotoCurrent-Regular.ttf'


def export_pdf(input_text,export_path):

    pdf=FPDF()
    pdf.add_page()

    pdf.add_font('GoNotoKurrentRegular',fname=GoNotoK_path)
    pdf.add_font('GoNotoCurrentRegular',fname=GoNotoC_path)

    pdf.set_font('GoNotoKurrentRegular',size=12)
    pdf.set_fallback_fonts(['GoNotoCurrentRegular'])

    input_text=input_text.expandtabs(4)
    pdf.multi_cell(w=0,h=10,text=input_text)

    pdf.output(export_path)