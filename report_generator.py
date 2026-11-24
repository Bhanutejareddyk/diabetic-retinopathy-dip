"""
report_generator.py
===================
Generates clean, professional ophthalmology diagnostic PDF reports
incorporating original and DIP processed intermediate images.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch


def generate_dr_pdf_report(report_data, output_pdf_path):
    """
    Builds a multi-section clinical PDF report for Diabetic Retinopathy screening.
    """
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#0F766E'),  # Teal-700
        spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#64748B')
    )
    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#0F172A'),
        spaceBefore=10,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#334155')
    )
    bold_label = ParagraphStyle(
        'BoldLabel',
        parent=styles['Normal'],
        fontSize=9,
        leading=12,
        fontName='Helvetica-Bold',
        textColor=colors.HexColor('#0F172A')
    )

    story = []

    # Header Banner
    header_table_data = [
        [
            Paragraph("<b>RETINAL IMAGING & DIP DIAGNOSTICS</b><br/><font size=8 color='#0D9488'>Automated Diabetic Retinopathy Staging System (ETDRS Compliant)</font>", title_style),
            Paragraph(f"<b>Report ID:</b> {report_data.get('report_id', 'N/A')}<br/><b>Date:</b> {report_data.get('timestamp', 'N/A')}<br/><b>Status:</b> Completed", subtitle_style)
        ]
    ]
    header_table = Table(header_table_data, colWidths=[4.2 * inch, 3.0 * inch])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ALIGN', (1, 0), (1, 0), 'RIGHT')
    ]))
    story.append(header_table)
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#0D9488'), spaceBefore=8, spaceAfter=12))

    # Diagnosis Summary Box
    pred = report_data.get("prediction", {})
    class_id = pred.get("class_id", 0)
    stage_name = pred.get("name", "Unknown")
    severity = pred.get("severity", "Unknown")
    confidence = pred.get("confidence", 0.0)
    urgency = pred.get("urgency", "Routine")
    color_hex = pred.get("color", "#10B981")

    diag_summary = [
        [
            Paragraph(f"<font size=11><b>DIAGNOSTIC STAGE:</b></font><br/><font size=16 color='{color_hex}'><b>{stage_name}</b></font><br/><font size=9 color='#475569'>Severity Grade: <b>{severity}</b></font>", body_style),
            Paragraph(f"<b>Classification Confidence:</b> {confidence}%<br/><b>Urgency Level:</b> <font color='{color_hex}'><b>{urgency}</b></font><br/><b>Primary Modality:</b> Digital Image Processing (DIP) + Ensemble ML", body_style)
        ]
    ]
    diag_table = Table(diag_summary, colWidths=[4.2 * inch, 3.0 * inch])
    diag_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8FAFC')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#CBD5E1')),
        ('PADDING', (0, 0), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
    ]))
    story.append(diag_table)
    story.append(Spacer(1, 10))

    # Clinical Recommendation
    rec_text = pred.get("recommendation", "")
    notes_text = pred.get("clinical_notes", "")
    rec_content = [
        [Paragraph(f"<b>Clinical Action Required:</b> {rec_text}", body_style)],
        [Paragraph(f"<b>Pathological Findings:</b> {notes_text}", body_style)]
    ]
    rec_table = Table(rec_content, colWidths=[7.2 * inch])
    rec_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F0FDFA')),
        ('LINELEFT', (0, 0), (0, -1), 3, colors.HexColor('#0D9488')),
        ('PADDING', (0, 0), (-1, -1), 6)
    ]))
    story.append(rec_table)
    story.append(Spacer(1, 12))

    # Intermediate DIP Processing Visuals (2x2 Grid)
    story.append(Paragraph("<b>DIP Pipeline Visual Evidence</b>", section_heading))
    processed_dir = report_data.get("processed_dir", "")
    images = report_data.get("images", {})

    cap_style = ParagraphStyle(
        'CapStyle',
        parent=styles['Normal'],
        fontSize=8,
        leading=10,
        alignment=1,  # 1 = TA_CENTER
        textColor=colors.HexColor('#0F172A')
    )

    def get_img_flowable(rel_path, caption):
        if not rel_path:
            return Paragraph(f"<i>{caption} (N/A)</i>", body_style)
        filename = rel_path.replace("\\", "/").rstrip("/").split("/")[-1]
        candidate = os.path.join(processed_dir, filename)
        if os.path.exists(candidate):
            try:
                img_f = Image(candidate, width=1.45 * inch, height=1.45 * inch)
                cap_f = Paragraph(f"<b>{caption}</b>", cap_style)
                return [img_f, cap_f]
            except Exception as e:
                print(f"[PDF] Error loading image {candidate}: {e}")
        return Paragraph(f"<i>{caption}</i>", body_style)

    img_grid_data = [
        [
            get_img_flowable(images.get("original"), "1. Original Fundus"),
            get_img_flowable(images.get("green"), "2. Green Channel (Max Contrast)"),
            get_img_flowable(images.get("clahe"), "3. CLAHE Enhancement"),
            get_img_flowable(images.get("vessels"), "4. Vessel Segmentation")
        ],
        [
            get_img_flowable(images.get("optic_disc"), "5. Optic Disc Mask"),
            get_img_flowable(images.get("lesions"), "6. Lesions (Red: MA, Yel: EX)"),
            get_img_flowable(images.get("overlay"), "7. Clinical Overlay"),
            Paragraph("<font size=8 color='#475569'><b>DIP Steps Explanation:</b><br/>"
                      "• Green channel absorbs hemoglobin peak.<br/>"
                      "• CLAHE enhances local micro-contrast.<br/>"
                      "• Top-hat morphological filter extracts vessels.<br/>"
                      "• Masking excludes optic disc from exudates.<br/>"
                      "• Contour bounding highlights microaneurysms.</font>", body_style)
        ]
    ]

    img_table = Table(img_grid_data, colWidths=[1.8 * inch, 1.8 * inch, 1.8 * inch, 1.8 * inch])
    img_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8)
    ]))
    story.append(img_table)
    story.append(Spacer(1, 10))

    # Quantitative DIP Extracted Features Table
    story.append(Paragraph("<b>Quantitative Retinal Biomarkers (DIP Feature Vector)</b>", section_heading))
    feats = report_data.get("features", {})
    feat_table_data = [
        [
            Paragraph("<b>DIP Biomarker</b>", bold_label),
            Paragraph("<b>Extracted Value</b>", bold_label),
            Paragraph("<b>Reference Normal Range</b>", bold_label),
            Paragraph("<b>Diagnostic Significance</b>", bold_label)
        ],
        [
            Paragraph("Vessel Density %", body_style),
            Paragraph(f"<b>{feats.get('vessel_density', 0.0)}%</b>", body_style),
            Paragraph("9.0% - 14.0%", body_style),
            Paragraph("Reflects vascular perfusion vs neovascular proliferation", body_style)
        ],
        [
            Paragraph("Microaneurysms (MA)", body_style),
            Paragraph(f"<b>{feats.get('microaneurysm_count', 0)}</b>", body_style),
            Paragraph("0 (None)", body_style),
            Paragraph("Earliest hallmark of diabetic capillary wall outpouching", body_style)
        ],
        [
            Paragraph("Hard Exudates (EX)", body_style),
            Paragraph(f"<b>{feats.get('exudate_count', 0)}</b>", body_style),
            Paragraph("0 (None)", body_style),
            Paragraph("Lipid and lipoprotein leakage from broken retinal barrier", body_style)
        ],
        [
            Paragraph("Exudate Area Ratio", body_style),
            Paragraph(f"<b>{feats.get('exudate_area_pct', 0.0)}%</b>", body_style),
            Paragraph("< 0.01%", body_style),
            Paragraph("Indicates severity of vascular permeability and edema risk", body_style)
        ],
        [
            Paragraph("Vessel Tortuosity", body_style),
            Paragraph(f"<b>{feats.get('vessel_tortuosity', 0.0)}</b>", body_style),
            Paragraph("10.0 - 35.0", body_style),
            Paragraph("Abnormal curvature induced by hypoxia and venous dilation", body_style)
        ]
    ]

    feat_table = Table(feat_table_data, colWidths=[1.8 * inch, 1.2 * inch, 1.6 * inch, 2.6 * inch])
    feat_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E2E8F0')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('PADDING', (0, 0), (-1, -1), 4),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
    ]))
    story.append(feat_table)
    story.append(Spacer(1, 12))

    # Footer Disclaimer
    footer_text = (
        "<b>Notice:</b> This automated screening report is generated by a Digital Image Processing (DIP) "
        "and Machine Learning pipeline for educational and triage support. Confirmation by a certified "
        "ophthalmologist via dilated biomicroscopy / OCT is required."
    )
    story.append(Paragraph(footer_text, ParagraphStyle('Footer', parent=styles['Italic'], fontSize=7.5, leading=10, textColor=colors.HexColor('#64748B'))))

    doc.build(story)
    return output_pdf_path
