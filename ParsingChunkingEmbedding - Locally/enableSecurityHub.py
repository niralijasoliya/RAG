import boto3
from botocore.exceptions import ClientError

def ensure_securityhub_enabled():
    securityhub = boto3.client('securityhub')
    try:
        # Check current subscription status
        response = securityhub.describe_hub()
        print("✅ Security Hub is already enabled in this region.")
    except ClientError as e:
        if e.response['Error']['Code'] == 'InvalidAccessException':
            print("⚠️ Security Hub is NOT enabled. Enabling now...")
            try:
                securityhub.enable_security_hub()
                print("✅ Security Hub has been successfully enabled.")
            except ClientError as enable_err:
                print(f"❌ Failed to enable Security Hub: {enable_err}")
        else:
            print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    ensure_securityhub_enabled()
