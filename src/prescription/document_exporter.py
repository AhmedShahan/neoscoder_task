"""
Document Exporter Module
Exports documents to various formats
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict
from ..schemas import Prescription


class DocumentExporter:
    """Exports medical documents to various formats"""

    def export_to_json(self, data: Dict, filepath: Path) -> str:
        """Export to JSON"""
        # Ensure parent directory exists
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        return str(filepath)

    def export_prescription_to_latex(
        self,
        prescription: Prescription,
        filepath: Path
    ) -> str:
        """Export prescription to LaTeX format"""
        
        # Ensure parent directory exists
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Escape LaTeX special characters
        def escape_latex(text):
            if text is None:
                return 'Not provided'
            replacements = {
                '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#',
                '_': r'\_', '{': r'\{', '}': r'\}',
                '~': r'\textasciitilde', '^': r'\textasciicircum',
                '\\': r'\textbackslash'
            }
            for char, replacement in replacements.items():
                text = str(text).replace(char, replacement)
            return text

        # Prepare sections
        patient_info = f"""
\\begin{{itemize}}
\\item Patient: {escape_latex(prescription.patient_name)}
\\item Age: {escape_latex(str(prescription.age))}
\\item Gender: {escape_latex(prescription.gender)}
\\end{{itemize}}
"""

        diagnosis = "\n".join([f"\\item {escape_latex(d)}" for d in prescription.diseases])
        diagnosis = f"\\begin{{itemize}}\n{diagnosis}\n\\end{{itemize}}"

        medications = "\n".join([f"\\item {escape_latex(med['medicine'])}" for med in prescription.medicines])
        medications = f"\\begin{{itemize}}\n{medications}\n\\end{{itemize}}"

        suggestions = "\n".join([f"\\item {escape_latex(s)}" for s in prescription.doctor_suggestions])
        suggestions = f"\\begin{{itemize}}\n{suggestions}\n\\end{{itemize}}"

        # LaTeX template
        latex_content = r"""\documentclass[a4paper,11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage{geometry}
\geometry{margin=1in}
\begin{document}
\begin{center}
{\LARGE\bfseries MEDICAL PRESCRIPTION} \\
\vspace{0.5em}
{\large Generated: %s}
\end{center}

\section*{Patient Information}
%s

\section*{Diagnosis}
%s

\section*{Prescribed Medications}
%s

\section*{Doctor's Suggestions}
%s

\end{document}
""" % (
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            patient_info,
            diagnosis,
            medications,
            suggestions
        )

        with open(filepath, 'w') as f:
            f.write(latex_content)

        return str(filepath)