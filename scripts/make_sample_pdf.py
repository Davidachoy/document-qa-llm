"""
Generate sample_docs/legal/contract.pdf
A realistic-looking software services contract for testing the experiments.
Uses only Python stdlib — no external dependencies.
"""

import os
import sys
import zlib
import struct

# ──────────────────────────────────────────────
# Minimal PDF writer (stdlib only)
# ──────────────────────────────────────────────

class _PDF:
    """
    Minimal PDF 1.4 writer (stdlib only).

    Object layout (fixed IDs make cross-references trivial):
        1  — Courier font
        2  — Pages tree  (written last, after all pages are known)
        3  — Catalog
        4, 6, 8, … — content streams
        5, 7, 9, … — page objects
    """

    FONT_ID  = 1
    PAGES_ID = 2
    CAT_ID   = 3
    FIRST_CONTENT_ID = 4   # content stream for page 1

    def __init__(self):
        self._objects: dict[int, bytes] = {}  # id → raw bytes between obj/endobj
        self._pages: list[int] = []           # page object IDs in order
        self._next_id = self.FIRST_CONTENT_ID

    def _alloc(self) -> int:
        oid = self._next_id
        self._next_id += 1
        return oid

    @staticmethod
    def _escape(text: str) -> str:
        return (text
                .replace("\\", "\\\\")
                .replace("(", "\\(")
                .replace(")", "\\)")
                .replace("\r", ""))

    def add_page(self, lines: list[str], font_size: int = 10,
                 margin_x: int = 55, start_y: int = 745,
                 line_height: int = 14):
        """Append one page built from a list of text lines."""

        # Build content stream
        parts = [b"BT\n", f"/F1 {font_size} Tf\n".encode()]
        y = start_y
        for raw in lines:
            safe = self._escape(raw)
            parts.append(f"{margin_x} {y} Td ({safe}) Tj T* 0 {-line_height} Td\n".encode())
            y -= line_height
        parts.append(b"ET\n")
        content = b"".join(parts)

        c_id = self._alloc()
        self._objects[c_id] = (
            f"<< /Length {len(content)} >>\nstream\n".encode()
            + content
            + b"\nendstream"
        )

        p_id = self._alloc()
        self._objects[p_id] = (
            f"<< /Type /Page\n"
            f"   /Parent {self.PAGES_ID} 0 R\n"
            f"   /MediaBox [0 0 612 792]\n"
            f"   /Contents {c_id} 0 R\n"
            f"   /Resources << /Font << /F1 {self.FONT_ID} 0 R >> >>\n"
            f">>"
        ).encode()

        self._pages.append(p_id)

    def save(self, path: str):
        """Serialise all objects, build xref, write file."""

        # Fixed objects
        self._objects[self.FONT_ID] = (
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>"
        )
        kids = " ".join(f"{pid} 0 R" for pid in self._pages)
        self._objects[self.PAGES_ID] = (
            f"<< /Type /Pages /Kids [{kids}] /Count {len(self._pages)} >>".encode()
        )
        self._objects[self.CAT_ID] = (
            f"<< /Type /Catalog /Pages {self.PAGES_ID} 0 R >>".encode()
        )

        # Serialise
        buf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets: dict[int, int] = {}

        for oid in sorted(self._objects):
            offsets[oid] = len(buf)
            buf += f"{oid} 0 obj\n".encode()
            buf += self._objects[oid]
            buf += b"\nendobj\n"

        # xref
        max_id = max(self._objects)
        xref_pos = len(buf)
        buf += f"xref\n0 {max_id + 1}\n".encode()
        buf += b"0000000000 65535 f \n"
        for oid in range(1, max_id + 1):
            off = offsets.get(oid, 0)
            buf += f"{off:010d} 00000 n \n".encode()

        buf += (
            f"trailer\n<< /Size {max_id + 1} /Root {self.CAT_ID} 0 R >>\n"
            f"startxref\n{xref_pos}\n%%EOF\n"
        ).encode()

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(buf)
        print(f"✓  Written: {path}  ({len(buf):,} bytes)")


# ──────────────────────────────────────────────
# Contract content
# ──────────────────────────────────────────────

PAGE_1 = [
    "SOFTWARE SERVICES CONTRACT",
    "",
    "Contract No.: SSC-2026-0047",
    "Date: March 1, 2026",
    "",
    "PARTIES",
    "",
    "CLIENT:",
    "  Acme Corporation, a Delaware corporation",
    "  123 Business Ave, Suite 400",
    "  San Francisco, CA 94107",
    "  (hereinafter referred to as 'Client')",
    "",
    "SERVICE PROVIDER:",
    "  TechSolve LLC, a California limited liability company",
    "  456 Innovation Drive",
    "  Palo Alto, CA 94301",
    "  (hereinafter referred to as 'Provider')",
    "",
    "RECITALS",
    "",
    "WHEREAS, Client desires to obtain certain software development and",
    "consulting services; and WHEREAS, Provider has the expertise and",
    "capacity to provide such services;",
    "",
    "NOW, THEREFORE, in consideration of the mutual covenants and",
    "agreements herein, the parties agree as follows:",
    "",
    "ARTICLE 1 — SCOPE OF SERVICES",
    "",
    "1.1 Provider shall deliver the following services (the 'Services'):",
    "    (a) Custom software development for Client's inventory platform.",
    "    (b) API integration with Client's existing ERP system (SAP S/4HANA).",
    "    (c) Quality assurance testing and bug remediation.",
    "    (d) Technical documentation and user training (up to 8 hours).",
    "",
    "1.2 Provider shall follow Client's brand guidelines and technical",
    "    standards as specified in Exhibit A attached hereto.",
    "",
    "ARTICLE 2 — CONTRACT DURATION",
    "",
    "2.1 This contract shall commence on March 1, 2026 (the 'Start Date')",
    "    and shall continue for a period of twelve (12) months, expiring",
    "    on February 28, 2027 (the 'End Date'), unless earlier terminated",
    "    pursuant to Article 8 of this Agreement.",
    "",
    "2.2 Upon mutual written agreement no later than 30 days before the",
    "    End Date, the parties may renew this contract for successive",
    "    12-month periods under the same terms and conditions.",
]

PAGE_2 = [
    "ARTICLE 3 — COMPENSATION AND PAYMENT",
    "",
    "3.1 In consideration for the Services, Client shall pay Provider a",
    "    monthly retainer fee of USD $18,500 (eighteen thousand five hundred",
    "    dollars), payable within 15 calendar days of each invoice date.",
    "",
    "3.2 For work exceeding the monthly scope defined in Exhibit A,",
    "    additional hours shall be billed at USD $175 per hour.",
    "",
    "3.3 All invoices not paid within 15 days shall accrue interest at",
    "    1.5% per month (18% per annum) on the outstanding balance.",
    "",
    "3.4 Travel expenses pre-approved in writing by Client shall be",
    "    reimbursed within 30 days of submission of receipts.",
    "",
    "ARTICLE 4 — OBLIGATIONS OF THE PROVIDER",
    "",
    "4.1 Provider shall assign a dedicated project manager as the primary",
    "    point of contact for all communications with Client.",
    "",
    "4.2 Provider shall deliver a written status report every two weeks",
    "    covering: progress, blockers, and upcoming milestones.",
    "",
    "4.3 Provider shall maintain the confidentiality of all Client data",
    "    and shall not disclose any proprietary information to third",
    "    parties without prior written consent from Client.",
    "",
    "4.4 Provider shall comply with SOC 2 Type II security controls and",
    "    shall notify Client within 24 hours of any data security incident.",
    "",
    "ARTICLE 5 — OBLIGATIONS OF THE CLIENT",
    "",
    "5.1 Client shall provide Provider with timely access to necessary",
    "    systems, data, and personnel required to perform the Services.",
    "",
    "5.2 Client shall designate a Product Owner who shall be available",
    "    for at least 10 hours per week to review deliverables and",
    "    provide feedback within 5 business days of delivery.",
    "",
    "5.3 Client shall pay all invoices in accordance with Article 3.",
    "",
    "ARTICLE 6 — INTELLECTUAL PROPERTY",
    "",
    "6.1 All custom software, code, and deliverables created by Provider",
    "    specifically for Client under this contract shall be considered",
    "    'work for hire' and shall become the sole property of Client",
    "    upon receipt of full payment.",
    "",
    "6.2 Provider retains ownership of any pre-existing tools, libraries,",
    "    or frameworks used in delivery (the 'Provider IP'). Provider",
    "    grants Client a perpetual, royalty-free license to use Provider IP",
    "    solely within the deliverables provided under this contract.",
]

PAGE_3 = [
    "ARTICLE 7 — CONFIDENTIALITY",
    "",
    "7.1 Both parties agree to keep confidential all non-public information",
    "    exchanged under this contract, including technical specifications,",
    "    business processes, pricing, and customer data.",
    "",
    "7.2 The confidentiality obligation shall survive termination of this",
    "    contract for a period of three (3) years.",
    "",
    "7.3 Exceptions: information is not confidential if it is (a) already",
    "    publicly known, (b) independently developed without use of",
    "    confidential information, or (c) disclosed by court order.",
    "",
    "ARTICLE 8 — TERMINATION",
    "",
    "8.1 Either party may terminate this contract without cause upon",
    "    sixty (60) days written notice to the other party.",
    "",
    "8.2 Client may terminate immediately for cause if Provider:",
    "    (a) Fails to deliver a milestone within 30 days of the agreed date.",
    "    (b) Commits a material breach not cured within 14 days of notice.",
    "    (c) Becomes insolvent or files for bankruptcy protection.",
    "",
    "8.3 Upon termination, Provider shall deliver all completed work",
    "    product to Client within 10 business days. Client shall pay for",
    "    all work satisfactorily completed up to the termination date.",
    "",
    "ARTICLE 9 — LIABILITY AND WARRANTIES",
    "",
    "9.1 Provider warrants that the Services will be performed in a",
    "    professional manner consistent with industry standards.",
    "",
    "9.2 Provider's total liability under this contract shall not exceed",
    "    the total fees paid in the 3 months preceding the claim.",
    "",
    "9.3 Neither party shall be liable for indirect, incidental, or",
    "    consequential damages arising from this contract.",
    "",
    "ARTICLE 10 — GOVERNING LAW AND DISPUTES",
    "",
    "10.1 This contract shall be governed by the laws of the State of",
    "     California, without regard to conflict-of-law principles.",
    "",
    "10.2 Any dispute shall first be submitted to mediation. If not",
    "     resolved within 30 days, the dispute shall be resolved by",
    "     binding arbitration in San Francisco, CA, under AAA rules.",
    "",
    "SIGNATURES",
    "",
    "For Acme Corporation:                For TechSolve LLC:",
    "",
    "____________________________         ____________________________",
    "Jane Smith, CEO                      Carlos Rivera, Managing Partner",
    "Date: March 1, 2026                  Date: March 1, 2026",
]


def main():
    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "sample_docs", "legal", "contract.pdf"
    )

    pdf = _PDF()
    pdf.add_page(PAGE_1, font_size=10, line_height=14)
    pdf.add_page(PAGE_2, font_size=10, line_height=14)
    pdf.add_page(PAGE_3, font_size=10, line_height=14)
    pdf.save(out_path)


if __name__ == "__main__":
    main()
