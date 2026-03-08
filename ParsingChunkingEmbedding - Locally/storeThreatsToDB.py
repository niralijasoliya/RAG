import boto3
import psycopg2
from datetime import datetime, timedelta
from collections import defaultdict
import uuid

# DB connection
conn = psycopg2.connect(
    dbname="XXX",
    user="XXX",
    password="XXXXXXXXXX",
    host="XXXXXXXXXX",
    port="5432"
)
cursor = conn.cursor()

# Initialize AWS clients
securityhub = boto3.client('securityhub', region_name='eu-west-1')
guardduty = boto3.client('guardduty', region_name='eu-west-1')
sts = boto3.client('sts')

def get_aws_identity():
    identity = sts.get_caller_identity()
    print(f"\n🧾 AWS Account ID: {identity['Account']}")
    print(f"User ARN: {identity['Arn']}")

def insert_security_task(data):
    allowed_sources = {'GuardDuty', 'ScoutSuite', 'securityhub'}
    allowed_severities = {'critical', 'high', 'medium', 'low'}
    allowed_authority = {'executing_by_ai', 'executing_by_human'}

    (
        _, name, severity, source, authority_level,
        *_rest, risk_score, _
    ) = data

    if severity not in allowed_severities:
        print(f"⛔ Skipping: severity '{severity}' not allowed")
        return
    if source not in allowed_sources:
        print(f"⛔ Skipping: source '{source}' not allowed")
        return
    if authority_level not in allowed_authority:
        print(f"⛔ Skipping: authority level '{authority_level}' not allowed")
        return

    query = """
    INSERT INTO public.security_tasks (
        security_task_id, name, severity, source, authority_level,
        resource_id, region, impact_description, score, compliance
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    try:
        cursor.execute(query, data)
        conn.commit()
        print(f"✅ Inserted: {name}")
    except psycopg2.errors.UniqueViolation:
        conn.rollback()
        print(f"⚠️ Skipped duplicate: {name} [{source}]")
    except Exception as e:
        conn.rollback()
        print(f"❌ Unexpected error while inserting '{name}': {e}")


def get_securityhub_findings():
    print("\n📌 Security Hub Findings (Last 24 hrs, High+ Severity):")
    start_time = (datetime.utcnow() - timedelta(days=1)).isoformat() + 'Z'
    end_time = datetime.utcnow().isoformat() + 'Z'

    findings = securityhub.get_findings(
        Filters={
            'UpdatedAt': [{'Start': start_time, 'End': end_time}],
            'SeverityLabel': [{'Value': 'HIGH', 'Comparison': 'EQUALS'}, {'Value': 'CRITICAL', 'Comparison': 'EQUALS'}]
        },
        MaxResults=100
    )['Findings']

    for f in findings:
        title = f['Title']
        description = f.get('Description', '')
        severity = f['Severity']['Label'].lower()
        if severity not in ['critical', 'high', 'medium', 'low']:
            continue  # skip unsupported
        source = 'securityhub'  # matches the constraint enum
        risk_score = float(f['Severity']['Normalized'])
        authority_level = "executing_by_human"
        resource_id = f['Resources'][0]['Id'] if f['Resources'] else 'unknown'
        region = 'eu-west-1'
        compliance = ''
        if isinstance(f.get('Compliance'), dict):
            related = f['Compliance'].get('RelatedRequirements', [])
            compliance = ','.join([c.get('Status', '') for c in related if isinstance(c, dict)])
        elif isinstance(f.get('Compliance'), str):
            compliance = f['Compliance']


        insert_security_task((
            str(uuid.uuid4()),                     # security_task_id
            title,                                  # name
            severity,                               # severity
            source,  # source
            authority_level,
            resource_id,                            # resource_id
            region,                                 # region
            description,                            # impact_description
            risk_score,                             # score
            compliance                              # compliance
        ))


def get_guardduty_findings():
    print("\n📌 GuardDuty Findings (Severity > 4):")
    detector_ids = guardduty.list_detectors()['DetectorIds']
    if not detector_ids:
        print("❌ No GuardDuty detector found.")
        return

    for detector_id in detector_ids:
        finding_ids = guardduty.list_findings(
            DetectorId=detector_id,
            FindingCriteria={'Criterion': {'severity': {'Gt': 4}}},
            MaxResults=50
        )['FindingIds']

        if finding_ids:
            findings = guardduty.get_findings(DetectorId=detector_id, FindingIds=finding_ids)
            for f in findings['Findings']:
                title = f['Title']
                description = f['Description']
                raw_severity = f['Severity']
                resource_id = f['Resource']['ResourceType']
                region = 'eu-west-1'
                authority_level = "executing_by_human"
                source = 'GuardDuty'

                # Normalize severity
                if raw_severity >= 7:
                    severity = 'critical'
                elif raw_severity >= 4:
                    severity = 'high'
                else:
                    severity = 'medium'

                score = round(raw_severity * 10, 2)
                compliance = ''  # GuardDuty does not provide compliance mapping

                insert_security_task((
                    str(uuid.uuid4()),
                    title,
                    severity,
                    source,
                    authority_level,
                    resource_id,
                    region,
                    description,
                    score,
                    compliance
                ))
        else:
            print("✅ No high severity findings found.")

# Execute
if __name__ == '__main__':
    get_aws_identity()
    get_securityhub_findings()
    get_guardduty_findings()
    cursor.close()
    conn.close()
