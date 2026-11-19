"""Generate comprehensive, realistic synthetic dataset and sample PDFs for DocVane."""

import os
import sys
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fpdf import FPDF
import pandas as pd

from src.data.validation import validate_dataset
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


INVOICE_TEMPLATES = [
    """INVOICE
Acme Solutions Corp
123 Innovation Drive, Tech Park, CA 94016
Tax ID / GSTIN: 27AABCT3518Q1Z8
Bill To: {client}
Invoice No: INV-{number}
Date: {date}
Due Date: {due_date}

Item Description                     Qty    Unit Price    Amount
-----------------------------------------------------------------
{item_1}                              {qty_1}    ${price_1:.2f}      ${amt_1:.2f}
{item_2}                              {qty_2}    ${price_2:.2f}      ${amt_2:.2f}
{item_3}                              {qty_3}    ${price_3:.2f}      ${amt_3:.2f}

Subtotal: ${subtotal:.2f}
Tax (10%): ${tax:.2f}
Grand Total: ${total:.2f}

Payment Terms: Net 30 days. Remit payment to Silicon Valley Bank, Acc: 9876543210.
Thank you for your business!""",

    """TAX INVOICE
Global Tech Supplies Ltd.
Invoice #: INV-{number}
Billing Date: {date}
Due Date: {due_date}

Customer: {client}
Address: 456 Enterprise Blvd, Suite 200

Particulars:
1. Cloud Infrastructure & Hosting Services - Qty: {qty_1} - ${amt_1:.2f}
2. Enterprise Maintenance Contract (SLA 99.9%) - Qty: {qty_2} - ${amt_2:.2f}
3. Dedicated DevOps Engineering Support - Qty: {qty_3} - ${amt_3:.2f}

Total Amount Due: ${total:.2f}
Payment Status: UNPAID
Please make cheques payable to Global Tech Supplies Ltd.""",

    """BILL OF SUPPLY / COMMERICAL INVOICE
Apex Digital Media LLC
Invoice Number: INV-{number}
Issued: {date}
Payment Due: {due_date}
Billed Client: {client}

Deliverables:
- {item_1} (${amt_1:.2f})
- {item_2} (${amt_2:.2f})
- {item_3} (${amt_3:.2f})

Discount: $0.00
Tax: ${tax:.2f}
Final Total: ${total:.2f}
Wire Transfer Instructions: Routing #021000021, Account #{number}890"""
]

RESUME_TEMPLATES = [
    """{name}
Email: {email} | Phone: {phone}
LinkedIn: linkedin.com/in/{handle} | GitHub: github.com/{handle}
Location: {city}, {state}

PROFESSIONAL SUMMARY
Dynamic and results-driven Software Engineer with {years}+ years of experience in architecting scalable distributed systems, microservices, and end-to-end machine learning pipelines. Demonstrated mastery in Python, Docker, Kubernetes, and cloud platforms.

TECHNICAL SKILLS
Languages: Python, Go, SQL, Bash, JavaScript, C++
Frameworks & Libraries: FastAPI, Flask, Scikit-Learn, PyTorch, Pandas, NumPy, TensorFlow
DevOps & Cloud: Docker, Kubernetes, AWS, GCP, CI/CD, GitHub Actions, Terraform
Databases: PostgreSQL, Redis, MongoDB, Elasticsearch

WORK EXPERIENCE
Senior Machine Learning Engineer | {company} ({start_yr} - Present)
- Designed and deployed end-to-end document classification pipeline processing 50k requests/day.
- Integrated MLflow experiment tracking and automated model registry deployments.
- Implemented automated CI/CD workflows using GitHub Actions and Docker containerization.

Software Developer | {prev_company} ({prev_yr} - {start_yr})
- Developed low-latency RESTful microservices with FastAPI and PostgreSQL.
- Reduced inference latency by 40% utilizing quantized linear models.

EDUCATION
Bachelor of Science in Computer Science | {university} ({grad_yr})
GPA: 3.8/4.0""",

    """CURRICULUM VITAE - {name}
Contact: {email} | {phone}
Address: {city}, {state}

OBJECTIVE
Motivated Data Scientist with deep expertise in predictive modeling, NLP text extraction, and MLOps lifecycle engineering seeking to build intelligent AI platforms.

CORE COMPETENCIES:
Skills: Python, Scikit-Learn, MLflow, Docker, Pytest, PyMuPDF, Pandas, Git, Linux
Specialties: Natural Language Processing, Feature Engineering, Classification Algorithms

PROFESSIONAL EXPERIENCE:
MLOps Specialist | {company} (2021 - Present)
- Built automated continuous training pipelines with scikit-learn and MLflow.
- Developed automated model performance monitoring with drift detection metrics.
- Authored unit and integration test suites achieving 92% code coverage.

Junior Data Analyst | {prev_company} (2019 - 2021)
- Cleaned and prepared large tabular and text datasets.
- Generated executive performance dashboards and statistical reports.

EDUCATION:
Master of Science in Artificial Intelligence | {university} (2019)""",

    """RESUME: {name}
Full Stack & ML Engineer
Contact Details: {email} | {phone}

PROFILE:
Enthusiastic developer focused on end-to-end full stack and machine learning systems. Experienced in building clean APIs with FastAPI, containerizing services with Docker, and writing clean maintainable code.

SKILLS & TECHNOLOGIES:
Technical Skills: Python, Scikit-Learn, FastAPI, Docker, Pytest, SQL, HTML/CSS, Git, AWS
Tools: Jupyter, Git, Postman, Linux, VS Code

EXPERIENCE:
Software Engineer | {company}
- Created automated document ingestion pipeline with PyMuPDF text parsing.
- Served calibrated classifier models behind REST endpoints.
- Managed container builds and automated deployment workflows.

EDUCATION:
B.Tech in Computer Science and Engineering | {university}"""
]

CERTIFICATE_TEMPLATES = [
    """CERTIFICATE OF ACHIEVEMENT
This certificate is proudly presented to
{name}

for successfully completing the rigorous industry examination and requirements for
{course}

Credential ID: CERT-{cert_id}
Issue Date: {date}
Expiry Date: Does Not Expire

Issued by:
{institution} Academy
Authorized Signatory: Dr. Marcus Vance, Chief Academic Officer""",

    """CERTIFICATE OF COMPLETION
This is to certify that
{name}

has demonstrated exemplary competence and successfully completed the professional credential program:
{course}

Given this day: {date}
Certificate ID: CERT-{cert_id}
Accredited By: Global Tech Council & Standards Institute
Verification Link: https://verify.certificates.org/id/{cert_id}""",

    """PROFESSIONAL CREDENTIAL & CERTIFICATE
Presented to:
{name}

In recognition of outstanding performance and completion of:
{course}

Issued on: {date}
Certificate Number: CERT-{cert_id}
Awarded By: {institution} Online Learning Platform
Status: Verified & Valid"""
]

BANK_STATEMENT_TEMPLATES = [
    """NATIONAL RESERVE BANK
Monthly Account Statement
Statement Period: {period}

Account Holder: {name}
Address: 789 Oakridge Ave, {city}
Account Number: {account_no}
Account Type: Premium Business Checking

SUMMARY:
Opening Balance: ${opening_bal:.2f}
Total Deposits & Credits: ${deposits:.2f}
Total Withdrawals & Debits: ${withdrawals:.2f}
Closing Balance: ${closing_bal:.2f}

TRANSACTION DETAILS:
Date        Description                     Debit       Credit      Balance
-------------------------------------------------------------------------
01/10/2026  Payroll Direct Deposit                      ${deposits:.2f}  ${mid_bal:.2f}
15/10/2026  Vendor Wire Transfer            $2,150.00               ${mid_bal_2:.2f}
28/10/2026  Utility & Operations AutoPay    $500.00                 ${closing_bal:.2f}

Available Balance as of {date}: ${closing_bal:.2f}
For customer service inquiries call 1-800-555-0199.""",

    """FIRST METROPOLITAN BANK
Official Statement of Account
Account Holder Name: {name}
Account No: {account_no}
Period: {period}
Generated On: {date}

Opening Balance: ${opening_bal:.2f}
Interest Earned: $14.20
Total Service Charges: $0.00
Ending Balance: ${closing_bal:.2f}

Transaction History:
- Electronic Transfer IN: +${deposits:.2f}
- Debit Card Merchant Purchase: -$340.50
- ATM Cash Withdrawal: -$200.00

Current Balance: ${closing_bal:.2f}
FDIC Insured up to $250,000""",

    """HORIZON COMMERCIAL BANK
Account Statement
Account Number: {account_no}
Client Name: {name}
Statement Date: {date}
Period Covered: {period}

Balances:
- Previous Balance: ${opening_bal:.2f}
- Total Credits: ${deposits:.2f}
- Total Debits: ${withdrawals:.2f}
- Closing Balance: ${closing_bal:.2f}

Notice: Please verify all posted transactions immediately. Report discrepancies within 30 days."""
]


def make_pdf(text: str, output_path: Path):
    """Render plain text content into a standard PDF using FPDF."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Helvetica", size=10)
    for line in text.split("\n"):
        # FPDF handles latin-1 characters safely
        sanitized_line = line.encode("latin-1", "replace").decode("latin-1")
        pdf.cell(0, 5, txt=sanitized_line, ln=True)
    pdf.output(str(output_path))


def generate_dataset(num_samples_per_class: int = 40):
    """Generate sample files and dataset.csv across all 4 categories."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    records = []

    clients = ["Starlight Logistics", "Nexus Health Inc", "Alpha Dynamics", "Vanguard Media", "Orion Cloud LLC", "Quantum Core", "Crestview Partners"]
    items = ["Full-Stack Engineering Consultation", "Serverless Infrastructure Setup", "Cloud Database Migration", "Security Audit & Vulnerability Assessment", "Machine Learning Pipeline Development", "API Gateway Integration", "Frontend UX/UI Optimization"]
    names = ["Alex Mercer", "Sarah Jenkins", "Michael Chang", "Emily Davis", "David Miller", "Jessica Watson", "Robert Taylor", "Sophia Patel", "Liam Henderson", "Emma Garcia"]
    cities = ["San Francisco", "Austin", "Seattle", "New York", "Boston", "Chicago", "Denver"]
    companies = ["MetaTech Systems", "Innovatech Labs", "CloudVibe", "DataWorks", "ByteForge", "HyperScale Inc"]
    courses = ["Machine Learning & MLOps Engineering", "Deep Neural Networks with PyTorch", "Advanced Cloud Architecture & Kubernetes", "FastAPI Microservices & Containerization", "Data Engineering with Apache Spark", "Applied Natural Language Processing"]
    institutions = ["Stanford Center for Professional Development", "MIT Professional Education", "Google Cloud Training", "DeepLearning.AI", "Linux Foundation"]

    # 1. Generate Invoices
    invoice_dir = RAW_DIR / "invoice"
    invoice_dir.mkdir(parents=True, exist_ok=True)
    for i in range(1, num_samples_per_class + 1):
        num = 1000 + i
        date = f"{random.randint(1, 28):02d}/10/2026"
        due_date = f"{random.randint(1, 28):02d}/11/2026"
        q1, q2, q3 = random.randint(1, 5), random.randint(1, 3), random.randint(5, 20)
        p1, p2, p3 = random.choice([150.0, 250.0, 500.0]), random.choice([200.0, 450.0]), random.choice([45.0, 80.0])
        a1, a2, a3 = q1 * p1, q2 * p2, q3 * p3
        sub = a1 + a2 + a3
        tax = sub * 0.10
        tot = sub + tax

        tmpl = INVOICE_TEMPLATES[i % len(INVOICE_TEMPLATES)]
        content = tmpl.format(
            client=random.choice(clients),
            number=num,
            date=date,
            due_date=due_date,
            item_1=random.choice(items), qty_1=q1, price_1=p1, amt_1=a1,
            item_2=random.choice(items), qty_2=q2, price_2=p2, amt_2=a2,
            item_3=random.choice(items), qty_3=q3, price_3=p3, amt_3=a3,
            subtotal=sub, tax=tax, total=tot,
        )
        fname = f"invoice_{i:03d}"
        if i <= 5:
            # Generate both PDF and TXT for first 5
            pdf_path = invoice_dir / f"{fname}.pdf"
            make_pdf(content, pdf_path)
            txt_path = invoice_dir / f"{fname}.txt"
            txt_path.write_text(content, encoding="utf-8")
        records.append({"document_text": content, "label": "invoice", "source_file": f"{fname}.txt"})

    # 2. Generate Resumes
    resume_dir = RAW_DIR / "resume"
    resume_dir.mkdir(parents=True, exist_ok=True)
    for i in range(1, num_samples_per_class + 1):
        name = random.choice(names)
        handle = name.lower().replace(" ", "")
        email = f"{handle}{random.randint(10, 99)}@gmail.com"
        phone = f"+1 (555) {random.randint(100, 999)}-{random.randint(1000, 9999)}"
        tmpl = RESUME_TEMPLATES[i % len(RESUME_TEMPLATES)]
        content = tmpl.format(
            name=name,
            email=email,
            phone=phone,
            handle=handle,
            city=random.choice(cities),
            state="CA",
            years=random.randint(3, 10),
            company=random.choice(companies),
            prev_company=random.choice(companies),
            start_yr=random.choice([2021, 2022, 2023]),
            prev_yr=random.choice([2018, 2019, 2020]),
            university="University of California, Berkeley",
            grad_yr=random.choice([2018, 2019, 2020]),
        )
        fname = f"resume_{i:03d}"
        if i <= 5:
            pdf_path = resume_dir / f"{fname}.pdf"
            make_pdf(content, pdf_path)
            txt_path = resume_dir / f"{fname}.txt"
            txt_path.write_text(content, encoding="utf-8")
        records.append({"document_text": content, "label": "resume", "source_file": f"{fname}.txt"})

    # 3. Generate Certificates
    cert_dir = RAW_DIR / "certificate"
    cert_dir.mkdir(parents=True, exist_ok=True)
    for i in range(1, num_samples_per_class + 1):
        name = random.choice(names)
        cert_id = f"84920{i:03d}"
        date = f"{random.randint(1, 28):02d}/09/2026"
        tmpl = CERTIFICATE_TEMPLATES[i % len(CERTIFICATE_TEMPLATES)]
        content = tmpl.format(
            name=name,
            course=random.choice(courses),
            cert_id=cert_id,
            date=date,
            institution=random.choice(institutions),
        )
        fname = f"certificate_{i:03d}"
        if i <= 5:
            pdf_path = cert_dir / f"{fname}.pdf"
            make_pdf(content, pdf_path)
            txt_path = cert_dir / f"{fname}.txt"
            txt_path.write_text(content, encoding="utf-8")
        records.append({"document_text": content, "label": "certificate", "source_file": f"{fname}.txt"})

    # 4. Generate Bank Statements
    bank_dir = RAW_DIR / "bank_statement"
    bank_dir.mkdir(parents=True, exist_ok=True)
    for i in range(1, num_samples_per_class + 1):
        name = random.choice(names)
        acc_no = f"49201938{i:04d}"
        period = "October 01, 2026 to October 31, 2026"
        date = "11/01/2026"
        op_bal = random.uniform(5000.0, 25000.0)
        dep = random.uniform(4000.0, 12000.0)
        withd = random.uniform(2000.0, 8000.0)
        cl_bal = op_bal + dep - withd
        mid_bal = op_bal + dep
        mid_bal_2 = closing_bal = cl_bal
        tmpl = BANK_STATEMENT_TEMPLATES[i % len(BANK_STATEMENT_TEMPLATES)]
        content = tmpl.format(
            name=name,
            city=random.choice(cities),
            account_no=acc_no,
            period=period,
            date=date,
            opening_bal=op_bal,
            deposits=dep,
            withdrawals=withd,
            closing_bal=cl_bal,
            mid_bal=mid_bal,
            mid_bal_2=mid_bal_2 + 500.0,
        )
        fname = f"bank_statement_{i:03d}"
        if i <= 5:
            pdf_path = bank_dir / f"{fname}.pdf"
            make_pdf(content, pdf_path)
            txt_path = bank_dir / f"{fname}.txt"
            txt_path.write_text(content, encoding="utf-8")
        records.append({"document_text": content, "label": "bank_statement", "source_file": f"{fname}.txt"})

    # Save to CSV
    df = pd.DataFrame(records)
    csv_path = PROCESSED_DIR / "dataset.csv"
    df.to_csv(csv_path, index=False)
    print(f"Dataset successfully created with {len(df)} samples across 4 classes at {csv_path}")

    # Validate
    validation_res = validate_dataset(csv_path)
    print("Dataset Validation Result:", validation_res)


if __name__ == "__main__":
    generate_dataset()
