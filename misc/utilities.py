from fpdf import FPDF

def export_pdf(input_text,export_path):

    pdf=FPDF()
    pdf.add_page()

    pdf.add_font('GoNotoKurrentRegular',fname=r"D:\RAG Project\misc\resources\Fonts\GoNotoKurrentRegular\GoNotoKurrent-Regular.ttf")
    pdf.add_font('GoNotoCurrentRegular',fname=r"D:\RAG Project\misc\resources\Fonts\GoNotoCurrentRegular\GoNotoCurrent-Regular.ttf")

    pdf.set_font('GoNotoKurrentRegular',size=12)
    pdf.set_fallback_fonts(['GoNotoCurrentRegular'])

    input_text=input_text.expandtabs(4)
    pdf.multi_cell(w=0,h=10,text=input_text)

    pdf.output(export_path)