"""
=============================================================================
CLINICAL DIAGNOSTIC PDF REPORT GENERATOR
MCA Final-Year Capstone Project
=============================================================================
Generates professional clinical laboratory reports with patient vitals,
multimodal breakdown, uploaded radiograph/CT, and Grad-CAM saliency maps.
=============================================================================
"""

import io
import json
from pathlib import Path
from typing import Dict, Any

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    KeepTogether,
    HRFlowable,
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_UPLOADS = BASE_DIR / "static" / "uploads"


def generate_diagnostic_pdf(record: Dict[str, Any]) -> io.BytesIO:
    """
    Generate an official diagnostic PDF report from a patient database record.
    Returns a BytesIO stream ready for HTTP download response.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    # Custom Color Palette
    navy = colors.HexColor("#0f172a")
    teal = colors.HexColor("#0f766e")
    slate_dark = colors.HexColor("#1e293b")
    slate_light = colors.HexColor("#f8fafc")
    border_color = colors.HexColor("#cbd5e1")
    danger_color = colors.HexColor("#b91c1c")
    normal_color = colors.HexColor("#047857")

    # Typography Styles
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=navy,
        alignment=1,  # Center
    )

    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=teal,
        alignment=1,
    )

    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=teal,
        spaceBefore=8,
        spaceAfter=4,
    )

    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=slate_dark,
    )

    bold_label = ParagraphStyle(
        "BoldLabel",
        parent=body_style,
        fontName="Helvetica-Bold",
    )

    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#64748b"),
        alignment=0,
    )

    story = []

    # 1. HEADER BANNER
    header_data = [
        [
            Paragraph("<b>MEDISCAN AI CLINICAL DIAGNOSTICS & RESEARCH LAB</b>", title_style),
        ],
        [
            Paragraph("AI-Based Multimodal Approach for Early Disease Detection · MCA Final-Year Capstone Project", subtitle_style),
        ]
    ]
    header_table = Table(header_data, colWidths=[540])
    header_table.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1.5, color=teal, spaceBefore=4, spaceAfter=8))

    # 2. PATIENT DEMOGRAPHICS & METADATA
    patient_id = record.get("patient_id", "PT-UNKNOWN")
    patient_name = record.get("patient_name", "Anonymous Patient")
    patient_age = str(int(record.get("patient_age"))) if record.get("patient_age") else "N/A"
    patient_gender = record.get("patient_gender", "Unspecified")
    created_at = record.get("created_at", "N/A")
    disease_module = record.get("disease_module", "General Evaluation")
    modality = record.get("imaging_modality", "Radiography")

    patient_meta = [
        [
            Paragraph("<b>Patient Name:</b>", bold_label), Paragraph(patient_name, body_style),
            Paragraph("<b>Patient ID:</b>", bold_label), Paragraph(patient_id, body_style),
        ],
        [
            Paragraph("<b>Age:</b>", bold_label), Paragraph(f"{patient_age} yrs", body_style),
            Paragraph("<b>Gender:</b>", bold_label), Paragraph(patient_gender, body_style),
        ],
        [
            Paragraph("<b>Disease Module:</b>", bold_label), Paragraph(disease_module, body_style),
            Paragraph("<b>Modality:</b>", bold_label), Paragraph(modality, body_style),
        ],
        [
            Paragraph("<b>Date & Time:</b>", bold_label), Paragraph(created_at, body_style),
            Paragraph("<b>Report Status:</b>", bold_label), Paragraph("Complete (AI Verified)", body_style),
        ],
    ]

    meta_table = Table(patient_meta, colWidths=[90, 180, 90, 180])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), slate_light),
        ("GRID", (0, 0), (-1, -1), 0.5, border_color),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # 3. PRIMARY DIAGNOSTIC CONCLUSION BOX
    prediction = record.get("prediction", "UNKNOWN")
    confidence = record.get("confidence_score", 0.0)

    is_danger = prediction in ["PNEUMONIA", "COVID", "TUBERCULOSIS", "Malignant", "Viral Pneumonia"]
    badge_bg = colors.HexColor("#fee2e2") if is_danger else colors.HexColor("#d1fae5")
    badge_text_color = danger_color if is_danger else normal_color

    finding_title = ParagraphStyle(
        "FindingTitle",
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=badge_text_color,
        alignment=1,
    )

    finding_sub = ParagraphStyle(
        "FindingSub",
        fontName="Helvetica",
        fontSize=9.5,
        leading=13,
        textColor=slate_dark,
        alignment=1,
    )

    conclusion_data = [
        [Paragraph(f"<b>PRIMARY AI FINDING: {prediction}</b>", finding_title)],
        [Paragraph(f"Confidence Score: <b>{confidence:.2f}%</b> (Uncalibrated Softmax Score)", finding_sub)]
    ]
    conclusion_table = Table(conclusion_data, colWidths=[540])
    conclusion_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), badge_bg),
        ("BOX", (0, 0), (-1, -1), 1.5, badge_text_color),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))
    story.append(conclusion_table)
    story.append(Spacer(1, 10))

    # 4. MULTIMODAL VITAL SIGNS & SYMPTOMS (IF AVAILABLE)
    vitals_dict = {}
    if record.get("vitals_json"):
        try:
            vitals_dict = json.loads(record["vitals_json"])
        except Exception:
            pass

    symptoms_text = record.get("symptoms_text", "").strip()

    if vitals_dict or symptoms_text:
        story.append(Paragraph("Clinical Observations & Physiological Vitals", heading_style))

        if vitals_dict:
            temp = vitals_dict.get("temperature", "N/A")
            hr = vitals_dict.get("heart_rate", "N/A")
            rr = vitals_dict.get("respiratory_rate", "N/A")
            spo2 = vitals_dict.get("spo2", "N/A")
            cough = "Positive" if vitals_dict.get("cough") in [1, "1"] else "Negative"
            dyspnea = "Positive" if vitals_dict.get("breathing_difficulty") in [1, "1"] else "Negative"

            vitals_table_data = [
                [
                    Paragraph("<b>Temperature:</b>", bold_label), Paragraph(f"{temp} °C", body_style),
                    Paragraph("<b>Heart Rate:</b>", bold_label), Paragraph(f"{hr} bpm", body_style),
                    Paragraph("<b>Resp. Rate:</b>", bold_label), Paragraph(f"{rr} /min", body_style),
                ],
                [
                    Paragraph("<b>SpO2 Saturation:</b>", bold_label), Paragraph(f"{spo2} %", body_style),
                    Paragraph("<b>Cough Status:</b>", bold_label), Paragraph(cough, body_style),
                    Paragraph("<b>Dyspnea:</b>", bold_label), Paragraph(dyspnea, body_style),
                ],
            ]
            v_table = Table(vitals_table_data, colWidths=[80, 100, 80, 100, 80, 100])
            v_table.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.5, border_color),
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            story.append(v_table)
            story.append(Spacer(1, 6))

        if symptoms_text:
            story.append(Paragraph(f"<b>Reported Symptoms Narrative:</b> <i>\"{symptoms_text}\"</i>", body_style))
            story.append(Spacer(1, 8))

    # 5. CLASS PROBABILITY / SCORE BREAKDOWN TABLE
    scores_dict = {}
    if record.get("scores_json"):
        try:
            scores_dict = json.loads(record["scores_json"])
        except Exception:
            pass

    if scores_dict:
        story.append(Paragraph("Differential Diagnosis Confidence Breakdown", heading_style))
        score_rows = [["Condition / Output Class", "Model Confidence Percentage"]]
        for cls_name, cls_score in scores_dict.items():
            score_rows.append([cls_name, f"{float(cls_score):.2f}%"])

        scores_table = Table(score_rows, colWidths=[340, 200])
        scores_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), teal),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("GRID", (0, 0), (-1, -1), 0.5, border_color),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ]))
        story.append(scores_table)
        story.append(Spacer(1, 10))

    # 6. RADIOLOGICAL SCANS & EXPLAINABLE AI (GRAD-CAM)
    image_file = record.get("image_file", "")
    gradcam_file = record.get("gradcam_file", "")

    img_path = STATIC_UPLOADS / image_file if image_file else None
    grad_path = STATIC_UPLOADS / gradcam_file if gradcam_file else None

    has_img = img_path and img_path.exists()
    has_grad = grad_path and grad_path.exists()

    if has_img or has_grad:
        story.append(Paragraph("Radiological Imaging & Explainable AI (Grad-CAM Saliency)", heading_style))

        image_cells = []
        if has_img and has_grad:
            img_rl = RLImage(str(img_path), width=180, height=180)
            grad_rl = RLImage(str(grad_path), width=180, height=180)
            image_cells = [
                [
                    Paragraph("<b>Uploaded Input Radiograph</b>", ParagraphStyle("C", parent=body_style, alignment=1)),
                    Paragraph("<b>Grad-CAM Attention Heatmap</b>", ParagraphStyle("C", parent=body_style, alignment=1)),
                ],
                [img_rl, grad_rl]
            ]
            img_table = Table(image_cells, colWidths=[270, 270])
        elif has_img:
            img_rl = RLImage(str(img_path), width=200, height=200)
            image_cells = [
                [Paragraph("<b>Uploaded Medical Image</b>", ParagraphStyle("C", parent=body_style, alignment=1))],
                [img_rl]
            ]
            img_table = Table(image_cells, colWidths=[540])
        else:
            img_table = None

        if img_table:
            img_table.setStyle(TableStyle([
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(KeepTogether(img_table))
            story.append(Spacer(1, 8))

    # 7. CLINICAL DISCLAIMER & PHYSICIAN SIGNATURE BLOCK
    footer_data = [
        [
            Paragraph(
                "<b>RESEARCH DISCLAIMER:</b> This report is generated by an educational machine learning decision-support "
                "prototype for MCA final-year research. The displayed scores represent experimental statistical approximations "
                "and are not certified medical diagnoses. Confirmation by a certified medical radiologist is mandatory.",
                disclaimer_style,
            ),
            Paragraph(
                "<br/><br/>__________________________________<br/>"
                "<b>Consulting Radiologist / Physician</b><br/>"
                "Signature & Medical Reg. No.",
                ParagraphStyle("Sign", parent=body_style, alignment=1),
            )
        ]
    ]
    footer_table = Table(footer_data, colWidths=[360, 180])
    footer_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(KeepTogether([HRFlowable(width="100%", thickness=1, color=border_color, spaceAfter=6), footer_table]))

    doc.build(story)
    buffer.seek(0)
    return buffer
