from fhir.resources.patient import Patient
from fhir.resources.documentreference import DocumentReference
from fhir.resources.condition import Condition
from fhir.resources.bundle import Bundle, BundleEntry
from datetime import datetime, timezone, date
import uuid
import json
import sys
import base64

class FHIRBuilder:
    def __init__(self):
        pass

    def create_patient(self, name: str, dob: str, gender: str) -> Patient:
        parts = name.strip().split()

        if len(parts) == 0:
            first = ""
            middle = []
            last = "Unknown"
        elif len(parts) == 1:
            first = parts[0]
            middle = []
            last = "Unknown"
        elif len(parts) == 2:
            first, last = parts
            middle = []
        else:
            first, *middle, last = parts

        first = first or ""
        last = last or "Unknown"
        middle = [m or "" for m in middle]

        # Validate gender against FHIR ValueSet
        valid_genders = ["male", "female", "other", "unknown"]
        if gender.lower() not in valid_genders:
            gender = "unknown"

        # Validate birth date format
        try:
            datetime.strptime(dob, "%Y-%m-%d")
        except ValueError:
            dob = "1900-01-01"  # Default if invalid

        patient_id = str(uuid.uuid4())
        
        return Patient(
            id=patient_id,
            name=[{
                "use": "official",
                "family": last,
                "given": [first] + [m for m in middle if m]
            }],
            gender=gender.lower(),
            birthDate=dob,
            # Add meta information for FHIR compliance
            meta={
                "versionId": "1",
                "lastUpdated": datetime.now(timezone.utc).isoformat(),
                "profile": ["http://hl7.org/fhir/StructureDefinition/Patient"]
            }
        )

    def create_document_reference(self, patient_id: str, text: str) -> DocumentReference:
        # Use base64 encoding instead of hex for standard compliance
        encoded_content = base64.b64encode(text.encode("utf-8")).decode("utf-8")
        
        return DocumentReference(
            id=str(uuid.uuid4()),
            status="current",
            subject={"reference": f"Patient/{patient_id}"},
            date=datetime.now(timezone.utc).isoformat(),
            # Add meta information
            meta={
                "versionId": "1",
                "lastUpdated": datetime.now(timezone.utc).isoformat(),
                "profile": ["http://hl7.org/fhir/StructureDefinition/DocumentReference"]
            },
            # Add document type for better categorization
            type={
                "coding": [{
                    "system": "http://loinc.org",
                    "code": "11506-3",
                    "display": "Progress note"
                }]
            },
            content=[{
                "attachment": {
                    "contentType": "text/plain",
                    "data": encoded_content
                }
            }]
        )

    def create_condition(self, patient_id: str, condition_text: str, diagnosis: str = None) -> Condition:
        condition_id = str(uuid.uuid4())
        
        # Create code structure with both text and potential coding
        code_structure = {"text": condition_text}
        
        # If diagnosis is provided, try to add structured coding
        if diagnosis:
            # This is a simplified example - in real implementation, 
            # you'd want to map to proper coding systems like ICD-10, SNOMED CT
            code_structure["coding"] = [{
                "system": "http://snomed.info/sct",
                "display": diagnosis
            }]

        return Condition(
            id=condition_id,
            subject={"reference": f"Patient/{patient_id}"},
            code=code_structure,
            recordedDate=datetime.now(timezone.utc).isoformat(),
            # Add clinical status (required in FHIR R4)
            clinicalStatus={
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                    "code": "active",
                    "display": "Active"
                }]
            },
            # Add verification status (required in FHIR R4)
            verificationStatus={
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
                    "code": "confirmed",
                    "display": "Confirmed"
                }]
            },
            # Add meta information
            meta={
                "versionId": "1",
                "lastUpdated": datetime.now(timezone.utc).isoformat(),
                "profile": ["http://hl7.org/fhir/StructureDefinition/Condition"]
            }
        )

    def create_bundle(self, patient: Patient, document: DocumentReference, condition: Condition) -> Bundle:
        """Create a standard FHIR Bundle containing all resources"""
        bundle_id = str(uuid.uuid4())
        
        entries = [
            BundleEntry(
                fullUrl=f"Patient/{patient.id}",
                resource=patient
            ),
            BundleEntry(
                fullUrl=f"DocumentReference/{document.id}",
                resource=document
            ),
            BundleEntry(
                fullUrl=f"Condition/{condition.id}",
                resource=condition
            )
        ]

        return Bundle(
            id=bundle_id,
            type="collection",
            timestamp=datetime.now(timezone.utc).isoformat(),
            total=len(entries),
            entry=entries,
            # Add meta information
            meta={
                "versionId": "1",
                "lastUpdated": datetime.now(timezone.utc).isoformat(),
                "profile": ["http://hl7.org/fhir/StructureDefinition/Bundle"]
            }
        )

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return super().default(obj)

def bytes_to_str(obj):
    if isinstance(obj, dict):
        return {k: bytes_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [bytes_to_str(i) for i in obj]
    elif isinstance(obj, bytes):
        return obj.decode('utf-8', errors='ignore')
    else:
        return obj

if __name__ == "__main__":
    input_json = json.loads(sys.stdin.readline())
    patient_info = input_json["patient"]
    condition_info = input_json["condition"]
    text = sys.stdin.read()

    builder = FHIRBuilder()
    
    # Create FHIR resources
    patient = builder.create_patient(**patient_info)
    document = builder.create_document_reference(patient.id, text)

    # Create condition with structured information
    condition_text = f"Diagnosis: {condition_info['diagnosis']}\nHistory: {condition_info['history']}\n" \
                     f"Recommendations: {condition_info['recommendations']}\nPhysical Exam: {condition_info['physical_exam']}"
    condition = builder.create_condition(patient.id, condition_text, condition_info.get('diagnosis'))

    # OPTION 1: Maintain backward compatibility with your existing code
    output = {
        "patient": patient.dict(),
        "document": document.dict(),
        "condition": condition.dict(),
        # Also include the FHIR Bundle for full compliance
        "fhir_bundle": builder.create_bundle(patient, document, condition).dict()
    }

    # OPTION 2: Full FHIR Bundle only (uncomment if you want to update your main code)
    # bundle = builder.create_bundle(patient, document, condition)
    # output = bundle.dict()

    output = bytes_to_str(output)
    print(json.dumps(output, cls=DateTimeEncoder, indent=2))