import boto3
from datetime import datetime, timedelta
from collections import defaultdict

# Initialize AWS clients with correct region
securityhub = boto3.client('securityhub', region_name='eu-west-1')
guardduty = boto3.client('guardduty', region_name='eu-west-1')
sts = boto3.client('sts')

# Get current AWS identity (account & user)
def get_aws_identity():
    identity = sts.get_caller_identity()
    print(f"\n🧾 AWS Account ID: {identity['Account']}")
    print(f"User ARN: {identity['Arn']}")

# Get recent Security Hub findings
def get_securityhub_findings():
    print("\n📌 Security Hub Findings (Last 24 hrs, High+ Severity):")
    start_time = (datetime.utcnow() - timedelta(days=1)).isoformat() + 'Z'
    end_time = datetime.utcnow().isoformat() + 'Z'

    findings = securityhub.get_findings(
        Filters={
            'UpdatedAt': [{
                'Start': start_time,
                'End': end_time
            }],
            'SeverityLabel': [{
                'Value': 'HIGH',
                'Comparison': 'EQUALS'
            }, {
                'Value': 'CRITICAL',
                'Comparison': 'EQUALS'
            }]
        },
        MaxResults=10
    )['Findings']
    summary = defaultdict(list)
    for f in findings:
        print({
            'Threat Name': f['Title'],
            'Description': f.get('Description', ''),
            'Severity': f['Severity']['Label'],
            'Risk Score': f['Severity']['Normalized'],
            'Product': f['ProductArn'].split('/')[-1],
            'Resource': f['Resources'][0]['Id'] if f['Resources'] else 'Unknown',
            'Source': 'Security Hub'
        })

# Get GuardDuty findings (High severity only)
def get_guardduty_findings():
    print("\n📌 GuardDuty Findings (Severity > 4):")
    detector_ids = guardduty.list_detectors()['DetectorIds']
    if not detector_ids:
        print("❌ No GuardDuty detector found.")
        return

    for detector_id in detector_ids:
        finding_ids = guardduty.list_findings(
            DetectorId=detector_id,
            FindingCriteria={
                'Criterion': {
                    'severity': {'Gt': 4}
                }
            },
            MaxResults=10
        )['FindingIds']

        if finding_ids:
            findings = guardduty.get_findings(DetectorId=detector_id, FindingIds=finding_ids)
            for finding in findings['Findings']:
                print({
                    'Threat Name': finding['Title'],
                    'Description': finding['Description'],
                    'Severity': finding['Severity'],
                    'Type': finding['Type'],
                    'Resource': finding['Resource']['ResourceType'],
                    'Risk Score': round(finding['Severity'] * 10),
                    'Source': 'GuardDuty'
                })
        else:
            print("✅ No high severity findings found.")

# Execute
if __name__ == '__main__':
    get_aws_identity()
    get_securityhub_findings()
    get_guardduty_findings()